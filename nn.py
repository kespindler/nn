import numpy as np
import pickle


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
    x = pickle.load(open('data/x.pkl', 'rb'))
    y = pickle.load(open('data/y.pkl', 'rb'))
    test = pickle.load(open('data/test.pkl', 'rb'))
    return x, y, test


def run_test(x, w, y):
    yhat = np.argmax(x @ w, 1).astype('float64')
    correct = (yhat == y).astype('float64')
    return yhat, correct


x, y, test = load_data()
x = x[:10000, :]
y = y[:10000]
w = load('w')
print(x.shape, y.shape, test.shape)

# w = np.random.rand(test.shape[1], 10)

# save(w, 'w')


def update(w, y, correct):
    concat = np.concatenate((
        np.reshape(y, (1, y.shape[0])),
        np.reshape(correct * 2 - 1, (1, correct.shape[0])),
    )).T
    rate = 0.01
    import ipdb;ipdb.set_trace()
    for y_e in concat:
        y_i, e = y_e
        for j in range(w.shape[1]):
            if y_i == j:
                w[:, j] += rate * e
            else:
                w[:, j] -= rate * e


for i in range(10):
    yhat, correct = run_test(x, w, y)
    print(i, np.mean(correct))
    update(w, y, correct)
