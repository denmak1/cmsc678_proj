import cv2
import numpy as np
import os
import sys
import tensorflow as tf

IMG_SIZE = 400
NUM_CHANNELS = 3

PATH_SEP = "\\"

USE_CLASSES_F = "use_classes.txt"

def main():
  if (len(sys.argv) != 3):
    print("usage: %s <model> <input image | directory>" % sys.argv[0])
    return

  MODEL_NAME = sys.argv[1]
  MODEL_PATH = "model" + PATH_SEP + MODEL_NAME
  MODEL_META = MODEL_PATH + PATH_SEP + MODEL_NAME + ".meta"

  img_file = sys.argv[2]

  images = []
  if (len(img_file.split(".")) > 1):             # single image
    print("loading image %s" % (img_file))

    img = cv2.imread(img_file)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), 0, 0, cv2.INTER_LINEAR)
    images.append(img)

  elif (len(img_file.split(".")) == 1):          # directory of images
    for fp in os.listdir(img_file):
      print("loading image %s" % (img_file + PATH_SEP + fp))

      img = cv2.imread(img_file + PATH_SEP + fp)
      img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), 0, 0, cv2.INTER_LINEAR)
      images.append(img)

  # adjust images and stuff
  images = np.array(images, dtype = np.uint8)
  images = images.astype('float32')
  images = np.multiply(images, 1.0/255.0)

  # reshape inputs to match the trained model
  input_X = images #.reshape(1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS)

  # load saved model and checkpoint
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config = config)

  saver = tf.train.import_meta_graph(MODEL_META)
  saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))

  # restore the graph and reload the tensors
  graph = tf.get_default_graph()

  Y_pred = graph.get_tensor_by_name("Y_pred:0")
  X      = graph.get_tensor_by_name("X:0") 
  Y_true = graph.get_tensor_by_name("Y_true:0")

  # read class labels from use_classes.txt file
  with open(USE_CLASSES_F) as f:
    use_classes = f.readlines()
  use_classes = [x.strip() for x in use_classes]

  class_labels = np.zeros((1, len(use_classes)))

  # classify the input image
  pred_dict = {X:      input_X,
               Y_true: class_labels}
  results = sess.run(Y_pred, feed_dict = pred_dict)

  # results correspond to same index in use_classes
  for result in results:
    print(result)
    for i in range(len(use_classes)):
      msg = "{0:>6.1%} :: " + use_classes[i]
      print(msg.format(result[i])) 

main()
