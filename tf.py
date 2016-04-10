import tensorflow as tf
from nn import load_data


def slicer(arr, slice_size):
    i = 0
    assert arr.shape[0] >= slice_size
    while True:
        arr_slice = arr[i:i+slice_size, :]
        if arr_slice.shape[0] < slice_size:
            i = 0
            continue
        i += slice_size
        yield arr_slice


X, Y, test = load_data()

sess = tf.Session()

x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
y = tf.nn.softmax(tf.matmul(x, w))
y_ = tf.placeholder(tf.float32, [None, 10])

max_w_op = tf.reduce_max(w)
min_w_op = tf.reduce_min(w)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sq_err = tf.scalar_mul(1/2, tf.reduce_sum(tf.square(y - y_)))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(sq_err)
init = tf.initialize_all_variables()

sess.run(init)
max_w = sess.run(max_w_op)
min_w = sess.run(min_w_op)
print('min_w:%f max_w:%f' % (min_w, max_w))
raise Exception

batch_size = 100
assert X.shape[0] == Y.shape[0]
X_data = slicer(X, batch_size)
Y_data = slicer(Y, batch_size)

for i in range(100):
    x_data = next(X_data)
    y_data = next(Y_data)

    sess.run(train_step, feed_dict={x: x_data, y_: y_data})
    acc = sess.run(accuracy, feed_dict={x: x_data, y_: y_data})
    max_w = sess.run(max_w_op)
    min_w = sess.run(min_w_op)
    print('i:%02d acc:%f min_w:%f max_w:%f' % (
        i, acc, min_w, max_w
    ))
