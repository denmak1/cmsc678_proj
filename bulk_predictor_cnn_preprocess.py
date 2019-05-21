import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import time

from preprocess import erode_img, kmeans_img, center_contours

NUM_CHANNELS = 3
PATH_SEP = "\\"
USE_CLASSES_F = "use_classes.txt"

def main():
  if (len(sys.argv) != 6):
    print("usage: %s <model> <directory> <model_name> <dump_csv:y/n> "
          "<save_imgs:y/n>\n" 
          "  <model> - the model name in the model/* directory\n"
          "  <directory> - the directory of all the image files to predict\n"
          "  <model_name> - the model name to save this as (in the csv file)\n"
          "  <dump_csv> - y or n to dump output to a csv file or not\n"
          "  <save_imgs> - if using a preprocessing model, specify to save\n"
          "                boxed and tagged image"
          % sys.argv[0])
    return

  model_name = sys.argv[1]
  model_path = "model" + PATH_SEP + model_name
  model_meta = model_path + PATH_SEP + model_name + ".meta"

  img_dir = sys.argv[2]
  m_name = sys.argv[3]
  dump_csv = sys.argv[4]
  save_imgs = sys.argv[5]

  # load saved model and checkpoint
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

  # read class labels from use_classes.txt file
  with open(USE_CLASSES_F) as f:
    use_classes = f.readlines()
  use_classes = [x.strip() for x in use_classes]

  class_labels = np.zeros((1, len(use_classes)))

  id_cntr = 0

  # results correspond to same index in use_classes
  if (dump_csv == 'y'):
    csv_file = open("output.csv", 'w')
    csv_file.write("pic_id,pic_name,series_name,tag_str,ep_num,time_stamp\n")

  if (save_imgs == 'y'):
    save_img_path = img_dir + "_" + model_name
    if (not os.path.exists(save_img_path)):
      os.makedirs(save_img_path, 0o0755)

  start_time = time.time()
  num_pics = 0
  for fp in os.listdir(img_dir):
    num_pics += 1
    images = []

    print("loading image %s" % (img_dir + PATH_SEP + fp))

    img      = cv2.imread(img_dir + PATH_SEP + fp)
    img_orig = img.copy()

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

    # adjust images and stuff
    images = np.array(images, dtype = np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0/255.0)

    # reshape inputs to match the trained model
    input_X = images #.reshape(1, img_size_x, img_size_y, NUM_CHANNELS)

    # classify the input image
    pred_dict = {X:      input_X,
                 Y_true: class_labels}
    results = sess.run(Y_pred, feed_dict = pred_dict)

    id_cntr += 1

    # results correspond to same index in use_classes
    c = 0
    overall_res = [0.0] * len(use_classes)
    for result in results:
      # put rectangle around region
      cv2.rectangle(img_orig, seg_pts[c][0], seg_pts[c][1], (0, 0, 255), 3)

      # first classify and draw boxes on the image
      y0, dy = seg_pts[c][0][1] + 14, 14
      for i in range(len(use_classes)):
        # overall result will just be the max of each prediction
        overall_res[i] = max([overall_res[i], result[i]])

        msg = "{0:>6.1%} :: " + use_classes[i]
        # print(msg.format(result[i]))

        # move to next line and print text
        y = y0 + i * dy
        cv2.putText(img_orig, msg.format(result[i]), (seg_pts[c][0][0], y),
          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

      c += 1

    # build the csv tag string using the overall results
    tag_str = m_name + "="
    for i in range(len(use_classes)):
      tag_str += ("%s:%.3f|" % (use_classes[i], 100.0*overall_res[i]))

    tag_str = tag_str[:-1]

    csv_row = ("%d,\"%s\",\"smile_precure!\",\"%s\",,\"\"\n" % \
               (id_cntr, fp, tag_str))
    print(csv_row)

    if (dump_csv == 'y'):
      csv_file.write(csv_row)

    if (save_imgs == 'y'):
      save_img_path = img_dir + "_" + model_name + PATH_SEP + fp
      #print(save_img_path)
      cv2.imwrite(save_img_path, img_orig, [cv2.IMWRITE_JPEG_QUALITY, 100])

  end_time = time.time()
  print("total time:", end_time - start_time,
        "avg time:", (end_time - start_time)/num_pics)

  if (dump_csv == 'y'):
    csv_file.close()

main()
