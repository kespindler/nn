import numpy as np
from PIL import Image
import pickle
import gzip


def read_file(fname):
    fpath = 'MNIST_data/' + fname
    return gzip.GzipFile(fpath)


def save_image(pixels, fname):
    """Takes ndarray with dtype=int8

    :param pixels:
    :param fname:
    :return:
    """
    pixels = np.tile(pixels, (3, 1, 1)).T
    pixels = np.rot90(pixels)
    pixels = np.flipud(pixels)
    img = Image.fromarray(pixels, 'RGB')
    img.save('vis/%s.png' % fname)


def read_label_file(fname):
    dt = np.dtype('int32').newbyteorder('>')
    with read_file(fname) as f:
        f.read(4)
        buf = f.read(4)
        n_items = np.frombuffer(buf, dt)[0]
        buf = f.read()
        y = np.frombuffer(buf, 'uint8')
    assert y.size == n_items
    y_one_hot = np.zeros((y.size, np.max(y) + 1))
    y_one_hot[np.arange(y.size), y] = 1
    return y_one_hot


def read_image_file(fname):
    dt = np.dtype('int32').newbyteorder('>')
    with read_file(fname) as f:
        f.read(4)
        buf = f.read(4)
        n_items = np.frombuffer(buf, dt)[0]
        buf = f.read(4)
        n_rows = np.frombuffer(buf, dt)[0]
        buf = f.read(4)
        n_cols = np.frombuffer(buf, dt)[0]
        buf = f.read()
        x = np.frombuffer(buf, 'uint8')
    assert x.size == n_items * n_rows * n_cols
    x = x.astype('float32') / 255.0
    x = x.reshape(n_items, n_rows * n_cols)
    for i in range(n_rows * n_cols):
        x[:, i] -= x[:, i].mean()
    return x


class DataSet:
    def __init__(self):
        self.test_x = read_image_file('t10k-images-idx3-ubyte.gz')
        self.test_y = read_label_file('t10k-labels-idx1-ubyte.gz')
        self.x = read_image_file('train-images-idx3-ubyte.gz')
        self.y = read_label_file('train-labels-idx1-ubyte.gz')
        self.batch_size = 100
        assert self.x.shape[0] >= self.batch_size
        self.i = 0
        self.indexer = np.arange(0, self.test_x.shape[0])
        np.random.shuffle(self.indexer)

    def __iter__(self):
        return self

    def __next__(self):
        slice_indices = self.indexer[self.i:self.i+self.batch_size]
        if slice_indices.size < self.batch_size:
            self.i = 0
            np.random.shuffle(self.indexer)
            return next(self)

        self.i += self.batch_size
        return self.x[slice_indices, :], self.y[slice_indices, :]


def new_w(shape):
    n_inputs = shape[0]
    w = np.random.randn(*shape) * np.sqrt(2.0/n_inputs)
    return w


def numerical_gradient(xs, w, ys):
    h = 1e-5
    grad = np.zeros(w.shape)
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            w[i, j] -= h
            yhat_prime = forward(xs, w)
            lprime = loss(yhat_prime, ys)
            w[i, j] += 2*h
            yhat_prime2 = forward(xs, w)
            lprime2 = loss(yhat_prime2, ys)
            grad[i, j] = (lprime2 - lprime) / (2 * h) / ys.shape[0]
            w[i, j] -= h
    return grad


def loss(yhat, y):
    y = np.argmax(y)
    correct_logprops = -np.log(yhat[:, y])
    return np.sum(correct_logprops) / y.size


def main():
    d = DataSet()
    w = new_w((784, 10))

    print('ag = analytic gradient. ng = numerial gradient')
    for i in range(1000):
        xs, ys = next(d)
        yhat = forward(xs, w)
        # correct_logprops = -np.log(yhat * ys)
        # loss = np.sum(correct_logprops) / xs.shape[0]
        dscores = yhat
        dscores -= ys
        dscores /= xs.shape[0]
        delta = xs.T @ dscores
        # delta2 = numerical_gradient(xs, w, ys)
        rate = 0.01
        yhat_test = forward(d.test_x, w)
        acc = accuracy(yhat_test, d.test_y)
        print('i:%02d acc:%f ag_min:%f ag_max:%f' % ( # ng_min:%f ng_max:%f' % (
            i, acc, delta.min(), delta.max()# , delta2.min(), delta2.max()
        ))
        w -= rate * delta


def accuracy(yhat, y):
    yhat = np.argmax(yhat, axis=1)
    y = np.argmax(y, axis=1)
    correct = (yhat == y)
    return np.sum(correct) / y.size


def forward(xs, w):
    activation = xs @ w
    softmax = np.zeros(activation.shape)
    for i in range(xs.shape[0]):
        act = activation[i, :]
        act -= act.max()
        eact = np.exp(act)
        p = eact / eact.sum()
        softmax[i, :] = p
    return softmax


main()
