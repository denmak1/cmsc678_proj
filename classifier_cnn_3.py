import tensorflow as tf
import numpy as np
import os
import cv2
import gc
from sklearn.utils import shuffle

INT_DEV_SIZE_PER_LABEL = 64
INT_DEV_RATIO = 0.2

BATCH_SIZE = 32

IMG_SIZE = 256
NUM_CHANNELS = 3

# adjust paths accordingly for linux
PATH_SEP = "\\"
TRAIN_DIR = "data" + PATH_SEP + "train" + PATH_SEP

MODEL_NAME = "smile_precure_model_cnn_3"
MODEL_PATH = "model" + PATH_SEP + MODEL_NAME
MODEL_FULL = MODEL_PATH + PATH_SEP + MODEL_NAME

# use class names specified in this file for training
USE_CLASSES_F = "use_classes.txt"

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
  # use only classes specified in the USE_CLASSES_F file
  with open(USE_CLASSES_F) as f:
    class_list = f.readlines()
  class_list = [x.strip() for x in class_list]
  total_num_classes = len(class_list)

  images = []
  labels = []
  classes = []
  fnames = []

  # training images are in ./train/<tag>/*
  # label is <tag>
  label_idx = 0
  for td in class_list:
    file_dir = TRAIN_DIR + td
    for fp in os.listdir(file_dir):

      # ignore gifs
      if fp.split('.')[1] == "gif":
        print("skipping gif")
        continue;

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

def create_weights(shape):
  return tf.Variable(tf.truncated_normal(shape, stddev = 0.05))
# END create_weights

def create_biases(size):
  return tf.Variable(tf.constant(0.05, shape = [size]))
# END create_biases

def create_conv_layer(input_data,
                      num_input_channels,
                      conv_filter_size,
                      num_filters):
  shape = [conv_filter_size,
           conv_filter_size,
           num_input_channels,
           num_filters]

  weights = create_weights(shape)
  biases  = create_biases(num_filters)

  layer = tf.nn.conv2d(input = input_data,
                       filter = weights,
                       strides = [1, 1, 1, 1],
                       padding = "SAME")
  layer += biases

  layer = tf.nn.max_pool(value = layer,
                         ksize = [1, 2, 2, 1],
                         strides = [1, 2, 2, 1],
                         padding = "SAME")

  layer = tf.nn.relu(layer)
  return layer
# END create_conv_layer                                    

def create_conv_layer_no_pool(input_data,
                              num_input_channels,
                              conv_filter_size,
                              num_filters):
  shape = [conv_filter_size,
           conv_filter_size,
           num_input_channels,
           num_filters]

  weights = create_weights(shape)
  biases  = create_biases(num_filters)

  layer = tf.nn.conv2d(input = input_data,
                       filter = weights,
                       strides = [1, 1, 1, 1],
                       padding = "SAME")
  layer += biases

  layer = tf.nn.relu(layer)
  return layer
# END create_conv_layer_no_pool

def flatten_layer(layer):
  layer_shape  = layer.get_shape()
  num_features = layer_shape[1:4].num_elements()
  layer        = tf.reshape(layer, [-1, num_features])

  return layer
# END flatten_layer

def create_fc_layer(input_data,
                    num_inputs,
                    num_outputs,
                    use_relu = True):
  shape = [num_inputs, num_outputs]

  weights = create_weights(shape)
  biases  = create_biases(num_outputs)

  layer = tf.matmul(input_data, weights) + biases

  if use_relu:
    layer = tf.nn.relu(layer)

  return layer
# END create_fc_layer

total_iterations = 0

# important to avoid out of memory errors on GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

session = tf.Session(config = config)

print("loading image data...")
total_num_classes, int_train_set, int_dev_set = load_data()
print("done")

print("int_train_set len = " + str(int_train_set.num_examples))
print("int_dev_set len = " + str(int_dev_set.num_examples))
print("total_num_classes = " + str(total_num_classes))

print("init model...")
shape = [None, IMG_SIZE, IMG_SIZE, NUM_CHANNELS]

# input
X = tf.placeholder(tf.float32, shape = shape, name = 'X')

# labels
Y_true = tf.placeholder(tf.float32,
                        shape = [None, total_num_classes],
                        name = "Y_true")
Y_true_class = tf.argmax(Y_true, dimension = 1)

# layer properties
filter_size_conv0 = 1
num_filters_conv0 = 64

filter_size_conv1 = 3
num_filters_conv1 = 64

filter_size_conv2 = 1
num_filters_conv2 = 256

filter_size_conv3 = 5
num_filters_conv3 = 32

fc_layer_size = 128

# create layers
layer_conv0 = \
  create_conv_layer_no_pool(X,
                            NUM_CHANNELS,
                            filter_size_conv0,
                            num_filters_conv0)

layer_conv1 = \
  create_conv_layer(layer_conv0,
                    num_filters_conv0,
                    filter_size_conv1,
                    num_filters_conv1)

layer_conv2 = \
  create_conv_layer(layer_conv1,
                    num_filters_conv1,
                    filter_size_conv2,
                    num_filters_conv2)

layer_conv3 = \
  create_conv_layer(layer_conv2,
                    num_filters_conv2,
                    filter_size_conv3,
                    num_filters_conv3)

layer_flat = flatten_layer(layer_conv3)

layer_fc1 = \
  create_fc_layer(layer_flat,
                  layer_flat.get_shape()[1:4].num_elements(),
                  fc_layer_size,
                  True)

layer_fc2 = \
  create_fc_layer(layer_fc1,
                  fc_layer_size,
                  total_num_classes,
                  False)

# output
Y_pred = tf.nn.softmax(layer_fc2, name = 'Y_pred')
Y_pred_class = tf.argmax(Y_pred, dimension = 1)

# init
session.run(tf.global_variables_initializer())

cross_entropy = \
  tf.nn.softmax_cross_entropy_with_logits(logits = layer_fc2,
                                          labels = Y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(cost)

# accuracy metrics stuff
correct_prediction = tf.equal(Y_pred_class, Y_true_class)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer())

f = open("train_log.txt", "w")

def print_progress(epoch, train_dict, dev_dict):
  dev_loss = session.run(cost,     feed_dict = dev_dict)
  acc      = session.run(accuracy, feed_dict = train_dict)
  dev_acc  = session.run(accuracy, feed_dict = dev_dict)

  msg = ("Epoch {0:>3} : "
         "Train Acc: {1:>6.1%}, "
         "Dev Acc: {2:>6.1%}, "
         "Dev Loss: {3:.3f}")
  print(msg.format(epoch + 1, acc, dev_acc, dev_loss))
  f.write(msg.format(epoch + 1, acc, dev_acc, dev_loss) + "\n")
# END print_progress

# training
total_iterations = 0

# make director for model saver junk files
if (not os.path.exists(MODEL_PATH)):
  os.makedirs(MODEL_PATH, 0o0755)

saver = tf.train.Saver()

def train(num_iteration):
  global total_iterations

  for i in range(total_iterations, total_iterations + num_iteration):
    train_X_b, train_Y_b, train_classes_b, train_fname_b = \
      int_train_set.next_batch(BATCH_SIZE)

    dev_X_b, dev_Y_b, dev_classes_b, dev_fname_b = \
      int_dev_set.next_batch(BATCH_SIZE)
 
    train_dict = {X:      train_X_b,
                  Y_true: train_Y_b}

    dev_dict = {X:      dev_X_b,
                Y_true: dev_Y_b}

    session.run(optimizer, feed_dict = train_dict)

    if (i % int(int_train_set.num_examples/BATCH_SIZE) == 0):
      epoch = int(i / int(int_train_set.num_examples/BATCH_SIZE))
      print_progress(epoch, train_dict, dev_dict)

      saver.save(session, MODEL_FULL)
      gc.collect()

  total_iterations += num_iteration

train(num_iteration=6000)

f.close()
