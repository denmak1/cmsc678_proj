import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from preprocess import erode_img, kmeans_img, center_contours

NUM_CHANNELS = 3
PATH_SEP = "\\"
USE_CLASSES_F = "use_classes.txt"

def main():
  if (len(sys.argv) != 3):
    print("usage: %s <model> <input image | directory>" % sys.argv[0])
    return

  # set model path info
  model_name = sys.argv[1]
  model_path = "model" + PATH_SEP + model_name
  model_meta = model_path + PATH_SEP + model_name + ".meta"

  img_file = sys.argv[2]

  # load saved model
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config = config)

  saver = tf.train.import_meta_graph(model_meta)
  saver.restore(sess, tf.train.latest_checkpoint(model_path))

  # restore the graph and reload the tensors
  graph = tf.get_default_graph()
  Y_pred = graph.get_tensor_by_name("Y_pred:0")
  X      = graph.get_tensor_by_name("X:0")
  Y_true = graph.get_tensor_by_name("Y_true:0")

  # read image size from loaded graph
  img_size_x = X.shape[2]
  img_size_y = X.shape[1]

  # read and preprocess image
  images = []
  if (len(img_file.split(".")) > 1):             # single image
    print("loading image %s" % (img_file))

    img      = cv2.imread(img_file)
    img_orig = img.copy()

    #img_pre = erode_img(img)
    #kmeans_img(img_pre)                         # way too slow lmao

    # get image segments
    seg_imgs, seg_pts = center_contours(img)

    for i in range(len(seg_imgs)):
      # ignore regions that are too small since they will throw error
      try: 
        img_temp = cv2.resize(seg_imgs[i], (img_size_x, img_size_y), 0, 0,
                              cv2.INTER_LINEAR)
        images.append(img_temp)
      except cv2.error:
        pass

  elif (len(img_file.split(".")) == 1):          # directory of images
    for fp in os.listdir(img_file):
      print("loading image %s" % (img_file + PATH_SEP + fp))

      img = cv2.imread(img_file + PATH_SEP + fp)
      img = cv2.resize(img, (img_size_x, img_size_y), 0, 0, cv2.INTER_LINEAR)
      images.append(img)

  # adjust images and stuff
  images = np.array(images, dtype = np.uint8)
  images = images.astype('float32')
  images = np.multiply(images, 1.0/255.0)

  # reshape inputs to match the trained model
  input_X = images #.reshape(1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS)

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
  c = 0
  overall_res = {}
  for result in results:
    # print(result)

    # put rectangle around region
    cv2.rectangle(img_orig, seg_pts[c][0], seg_pts[c][1], (0, 0, 255), 3)

    # used for multi line text
    y0, dy = seg_pts[c][0][1] + 14, 14
    for i in range(len(use_classes)):
      # overall result will just be the max of each tag
      overall_res[use_classes[i]] = \
        max([overall_res.get(use_classes[i], 0.0), result[i]])

      msg = "{0:>6.1%} :: " + use_classes[i]
      # print(msg.format(result[i]))

      # move to next line and print text
      y = y0 + i * dy
      cv2.putText(img_orig, msg.format(result[i]), (seg_pts[c][0][0], y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    c += 1

  print(overall_res)

  cv2.imshow("cropped", img_orig)
  cv2.waitKey(0)

main()
