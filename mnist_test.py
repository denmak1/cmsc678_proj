# sample with mnist data

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.5
epochs = 10
batch_size = 100

def main():
  # input
  x = tf.placeholder(tf.float32, [None, 784])

  # output, as a vector
  y = tf.placeholder(tf.float32, [None, 10])

  # weights for each layer
  w1 = tf.Variable(tf.random_normal([784, 300], stddev = 0.03), name = "w1")
  w2 = tf.Variable(tf.random_normal([300, 10], stddev = 0.03), name = "w2")

  # bias for each layer
  b1 = tf.Variable(tf.random_normal([300]), name = "b1")
  b2 = tf.Variable(tf.random_normal([10]), name = "b2")

  # matrix multiplication = x * w1 + b1
  hidden_out = tf.add(tf.matmul(x, w1), b1)

  # set up neural network
  hidden_out = tf.nn.relu(hidden_out)

  # hidden layer output
  y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, w2), b2))

  # clip output to avoid log(0), log(0.0000000001) at lowest
  y_c = tf.clip_by_value(y_, 1e-10, 0.9999999)

  # cross entropy
  xent = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_c) + \
         (1-y) * tf.log(1 - y_c), axis = 1))

  optimizer = tf.train.GradientDescentOptimizer( \
    learning_rate = learning_rate).minimize(xent)

  # initialize
  init_op = tf.global_variables_initializer()

  # estimate accuracy
  correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



  # run tf session
  with tf.Session() as sess:
    sess.run(init_op)
    total_batch = int(len(mnist.train.labels) / batch_size)

    for epoch in range(epochs):
      avg_cost = 0

      for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size = batch_size)

        b, c = sess.run([optimizer, xent], \
                        feed_dict = {x: batch_x, y: batch_y})

        avg_cost += c / total_batch

      print "Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost)

    print sess.run(accuracy, \
                   feed_dict = {x: mnist.test.images, y: mnist.test.labels})

main()
