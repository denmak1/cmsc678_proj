# RNN sample with mnist data
import time
import numpy as np
import os
import cv2
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

LEARNING_RATE = 0.001
N_EPOCHS = 15
BATCH_SIZE = 32
N_NEURONS = 128

INT_DEV_SIZE_PER_LABEL = 64
INT_DEV_RATIO = 0.2

IMG_SIZE = 256
NUM_CHANNELS = 3

# adjust paths accordingly for linux
WORK_DIR = "H:\\dev\\cmsc678_proj\\"
TRAIN_DIR = WORK_DIR + "data\\train\\"
PATH_SEP = "\\"

class DataSet():
  def __init__(self, _images, _labels, _classes, _fnames):
    self.num_examples = _images.shape[0]

    self.images = _images
    self.labels = _labels
    self.classes = _classes
    self.fnames = _fnames

    self.epochs_done = 0
    self.index_in_epoch = 0

  def next_batch(self, batch_size):
    start = self.index_in_epoch
    self.index_in_epoch += batch_size

    if self.index_in_epoch > self.num_examples:
      self.epochs_done += 1
      start = 0
      self.index_in_epoch = batch_size
      assert batch_size <= self.num_examples
    end = self.index_in_epoch

    return self.images[start:end], \
           self.labels[start:end], \
           self.classes[start:end], \
           self.fnames[start:end]
# END DataSet

def load_data():
  images = []
  labels = []
  classes = []
  fnames = []

  class_list = []      # all possible labels
  for td in os.listdir(TRAIN_DIR):
    class_list.append(td)
  total_num_classes = len(class_list)

  # training images are in ./train/<tag>/*
  # label is <tag>
  label_idx = 0
  for td in os.listdir(TRAIN_DIR):
    file_dir = TRAIN_DIR + td
    for fp in os.listdir(file_dir):
      fpath = file_dir + PATH_SEP + fp
      print(fpath)

      img = cv2.imread(fpath)
      img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), 0, 0, cv2.INTER_LINEAR)
      img = img.astype(np.float32)
      img = np.multiply(img, 1.0 / 255.0)
      images.append(img)

      # use an array for the label where 1 = label class
      label = np.zeros(len(class_list))
      label[label_idx] = 1.0
      labels.append(label)

      classes.append(td)
      fnames.append(fp)

    label_idx += 1

  images = np.array(images)
  labels = np.array(labels)
  classes = np.array(classes)
  fnames = np.array(fnames)

  images, labels, classes, fnames = shuffle(images, labels, classes, fnames)

  dev_size = int(INT_DEV_RATIO * images.shape[0])

  int_train_images  = images[dev_size:]
  int_train_labels  = labels[dev_size:]
  int_train_fnames  = fnames[dev_size:]
  int_train_classes = classes[dev_size:]

  int_dev_images  = images[:dev_size]
  int_dev_labels  = labels[:dev_size]
  int_dev_fnames  = fnames[:dev_size]
  int_dev_classes = classes[:dev_size]

  int_train_set = DataSet(int_train_images,
                          int_train_labels,
                          int_train_classes,
                          int_train_fnames)

  int_dev_set = DataSet(int_dev_images,
                        int_dev_labels,
                        int_dev_classes,
                        int_dev_fnames)

  return total_num_classes, int_train_set, int_dev_set
# END load_data

def main():
  print("loading image data...")
  total_num_classes, int_train_set, int_dev_set = load_data()
  print("done")

  print("int_train_set len = " + str(int_train_set.num_examples))
  print("int_dev_set len = " + str(int_dev_set.num_examples))
  print("total_num_classes = " + str(total_num_classes))

  # parameters
  n_steps = IMG_SIZE
  n_inputs = IMG_SIZE
  n_outputs = total_num_classes

  # build a rnn model
  shape = [None, IMG_SIZE, IMG_SIZE]

  # input
  X = tf.placeholder(tf.float32, shape = shape, name = 'X')

  # labels
  Y_true = tf.placeholder(tf.float32,
                          shape = [None, total_num_classes],
                          name = "Y_true")
  Y_true_class = tf.argmax(Y_true, dimension = 1)

  cell = tf.nn.rnn_cell.LSTMCell(num_units = N_NEURONS)
  output, state = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)
  logits = tf.keras.layers.Dense(state, n_outputs)

  cross_entropy = \
    tf.nn.sparse_softmax_cross_entropy_with_logits(labels = Y_true, logits = logits)
  loss = tf.reduce_mean(cross_entropy)

  optimizer = \
    tf.train.AdamOptimizer(learning_rate = LEARNING_RATE).minimize(loss)

  prediction = tf.nn.in_top_k(logits, Y_true, 1)
  accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

  # initialize the variables
  init = tf.global_variables_initializer()

  # dynamic memory allocation
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  # train the model
  with tf.Session(config = config) as sess:
    sess.run(init)
    n_batches = int_train_set.num_examples // BATCH_SIZE

    for epoch in range(N_EPOCHS):
      s_time = time.time() 
      for batch in range(n_batches):
        train_X_b, train_Y_b, train_classes_b, train_fname_b = \
          int_train_set.next_batch(BATCH_SIZE)

        dev_X_b, dev_Y_b, dev_classes_b, dev_fname_b = \
          int_dev_set.next_batch(BATCH_SIZE)

        train_dict = {X:      train_X_b,
                      Y_true: train_Y_b}

        dev_dict = {X:      dev_X_b,
                    Y_true: dev_Y_b}

        sess.run(optimizer, feed_dict = train_dict)

        loss_train, acc_train = \
          sess.run([loss, accuracy], feed_dict = train_dict)
      e_time = time.time()

      print('Epoch: {}, Train Loss: {:.3f}, Train Acc: {:.3f}, Time: {:.3f}'.format(
        epoch + 1, loss_train, acc_train, e_time - s_time))

    loss_test, acc_test = \
      sess.run([loss, accuracy], feed_dict = dev_dict)

    print('Test Loss: {:.3f}, Test Acc: {:.3f}'.format(loss_test, acc_test))

main()
