# cmsc678_proj
Image Classification - using a CNN to label and detect characters
in screenshots from anime using only fanart as the training data.

Weakly supervised approach so no manually inspecting or marking up of
images to assist in training.


# Notes
I'm using cygwin and windows cmd so the paths coded in the files
will need to be adjusted to work.

You'll need python 3, with TensorFlow GPU acceleration and other stuff.


# About the files
Classifiers:
* classifier_cnn_2r.py - CNN configuration 2, see layer properties in the file
* classifier_cnn_3r.py - CNN configuration 3, see layer properties in the file
* classifier_cnn_4r.py - CNN configuration 4, see layer properties in the file

Predictors:
* bulk_predictor_cnn.py - bulk predicts using the CNN on files in a directory
* bulk_predictor_cnn_preprocess.py - bulk predicts using the CNN on files in a
  directory, incorporates preprocessing and saves the marked up images
* predictor_cnn.py - predicts on an input file without preprocessing
* predictor_cnn_preprocess.py - predicts on an input image file with
  preprocessing

Others:
* kmeans.py - implements kmeans, used for preprocessing
* preprocess.py - preprocessing functions for images
* use_classes.txt - when training a model, this file contains the list of
  classes to include, which should be some or all of the directories in the
  data/train dir or whatever

Scripts:
* scripts/dl_pics.py - downloads pics from a booru given some input tags, used
  for getting training data
* scripts/extract_screens.sh - bash script that uses ffmpeg to extract frames
  from a video file, used for getting testing data

Logs:
* logs/\* - model training logs and classification outputs

Deprecated files (these can be deleted but whatever):
* classifier_cnn.py
* classifier_cnn_2.py
* classifier_cnn_2_sigmoid.py
* classifier_cnn_3.py


# Usage
For usage examples, just invoke the scripts without any arguments.

General usage steps:
* Get training data using dl_pics.py
* Extract test data from video files using extract_screens.sh
* Adjust use_classes.txt file to set classes to use for training
* Train model using the classifier_cnn\*.py files
* Run bulk classifier on extracted tes data screenshots
* Import CSV into DB
