import numpy as np
import pickle


# Things to try out
# 1. relu
# 2. analytic gradient
# 3. use bias
# 4. regularization


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(y):
    # y already sigmoided
    return y * (1 - y)


def tanh(x):
    return np.tanh(x)


def dtanh(y):
    return 1 - y * y


def relu(x):
    return np.max(0, x)


def drelu(y):
    # confirm correct
    return int(y > 0)


def save(arr, arr_name):
    with open('data/{}.pkl'.format(arr_name), 'wb') as f:
        pickle.dump(arr, f)


def load(arr_name):
    return pickle.load(open('data/{}.pkl'.format(arr_name), 'rb'))


def load_csv():
    # for train, train[1:, 0] is y value, train[1:, 1:] is data
    train = np.genfromtxt('data/train.csv', delimiter=',')
    test = np.genfromtxt('data/test.csv', delimiter=',')
    return train, test


def load_data():
    # 0 mean center the data
    # one-hot encode y
    x = pickle.load(open('data/x.pkl', 'rb'))
    x -= np.mean(x, 0)
    y = pickle.load(open('data/y.pkl', 'rb'))
    yy = np.zeros((y.size, np.max(y) + 1))
    yy[np.arange(y.size), y.astype('int')] = 1
    test = pickle.load(open('data/test.pkl', 'rb'))
    return x, yy, test


def compute(x, w, n_classes):
    activation = x @ w
    # yhat = np.argmax(activation, 1)
    yhat = tanh(activation)
    return yhat


def predict(y, yhat):
    yy = np.argmax(y, 1)
    yyhat = np.argmax(yhat, 1)
    correct = (yy == yyhat)
    return np.mean(correct)


def update(x, w, y, yhat):
    err = y - yhat  # 10k x 10

    rate = 1e-4

    delta = x.T @ dtanh(yhat) * rate

    w += delta

    return np.sum(err)  # total loss


def align(f):
    return ('%.2f' % f).zfill(5)


def main():
    x, y, test = load_data()
    # x = x[:10000, :]
    # y = y[:10000, :]

    w = load('w')

    # w = np.random.randn(test.shape[1], 10) * np.sqrt(2/test.shape[1])
    # save(w, 'w')

    for i in range(100):
        yhat = compute(x, w, np.max(y) + 1)
        percent_correct = predict(y, yhat)
        loss = update(x, w, y, yhat)
        print('%02d %s%% %.5f' % (
            i, align(percent_correct * 100), loss
        ))

    return x, y, test, w

x, y, test, w = main()
