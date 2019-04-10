# cmsc678_proj
Image Classification Probably


Notes:
I'm using cygwin and windows cmd so the paths coded in the files
will need to be adjusted to work.

Run the script downloader from the scrips dir (scripts/dl_scripts.py)
to download training data. It will be saved into ../data/train/<tag>.

Usage example: "python dl_pics.py shibuki_ran solo"
  This will download all of the images tagged with shibuki_ran and solo
  from the booru.

Then run the classifier from the work dir:
  "python classifier_test.py"
Check the paths specified in the .py file because you will want to use
something else probably.

You'll probably need python 3.


TODO:
Try RNN with our image data.

Try combination of RNN with CNN:
  RNN layers into convolutional layer at the end.
  CNN layers into RNN cell at the end?

Try using multi output instead of single:
  Replace the relu with sigmoid function in output layer.
