# cmsc678_proj
Image Classification Probably

I'm using cygwin and windows cmd so the paths coded in the files
will need to be adjusted to work.

Run the script downloader from the scrips dir (scripts/dl_scripts.py)
to download training data. It will be saved into ../data/train/<tag>.

Usage example: "python dl_pics.py shibuki_ran solo"
  This will download all of the images tagged with shibuki_ran and solo
  from the booru.

Then run the classifier from the work dir:
  "python classifier_test.py"
Check the paths specified in the .py file becuase you will want to use
something else probably.

You'll probably need python 3.
