# cmsc678_proj
Image Classification Probably


# Notes
I'm using cygwin and windows cmd so the paths coded in the files
will need to be adjusted to work.

You'll need python 3.


# About the files
Classifiers:
* classifier_cnn_2r.py - CNN configuration 2, see layer properties in the file
* classifier_cnn_3r.py - CNN configuration 3, see layer properties in the file

Predictors:
* bulk_predictor_cnn.py - bulk predicts using the CNN on files in a directory
* predictor_cnn.py - predicts on an input file without preprocessing
* predictor_cnn_preprocess.py - predicts on an input image file with
  preprocessing

Others:
* kmeans.py - implements kmeans, used for preprocessing
* preprocess.py - preprocessing functions for images

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
For usage examples, just invoke the scripts without any arguments


# TODO
Try RNN with our image data.

Try combination of RNN with CNN:
  RNN layers into convolutional layer at the end.
  CNN layers into RNN cell at the end?

Rename repo
