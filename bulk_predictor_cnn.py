import cv2
import numpy as np
import os
import sys
import tensorflow as tf

IMG_SIZE = 256
NUM_CHANNELS = 3

PATH_SEP = "\\"

USE_CLASSES_F = "use_classes.txt"

def main():
  if (len(sys.argv) != 5):
    print("usage: %s <model> <directory> <model_name> <dump_csv:y/n>\n" 
          "  <model> - the model name in the model/* directory\n"
          "  <directory> - the directory of all the image files to predict\n"
          "  <model_name> - the model name to save this as (in the csv file)\n"
          "  <dump_csv> - y or n to dump output to a csv file or not\n"
          % sys.argv[0])
    return

  model_name = sys.argv[1]
  model_path = "model" + PATH_SEP + model_name
  model_meta = model_path + PATH_SEP + model_name + ".meta"

  img_dir = sys.argv[2]
  m_name = sys.argv[3]
  dump_csv = sys.argv[4]

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

  for fp in os.listdir(img_dir):
    images = []

    print("loading image %s" % (img_dir + PATH_SEP + fp))

    img = cv2.imread(img_dir + PATH_SEP + fp)
    img = cv2.resize(img, (img_size_x, img_size_y), 0, 0, cv2.INTER_LINEAR)
    images.append(img)

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
    for result in results:   # should always be 1
      tag_str = m_name + "="

      for i in range(len(use_classes)):
        msg = "{0:>6.1%} :: " + use_classes[i]
        print(msg.format(result[i]))

        tag_str += ("%s:%.3f|" % (use_classes[i], 100.0*result[i]))

      tag_str = tag_str[:-1]
      csv_row = ("%d,\"%s\",\"smile_precure!\",\"%s\",,\"\"\n" % \
                       (id_cntr, fp, tag_str))
      print(csv_row)

      if (dump_csv == 'y'):
        csv_file.write(csv_row)

  if (dump_csv == 'y'):
    csv_file.close()

main()
