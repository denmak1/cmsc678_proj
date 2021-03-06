from lxml import html

import os
import random
import requests
import sys
import time

def main():
  if (len(sys.argv) < 3):
    msg = ("usage: python %s <tag> <tag2> [excluded ratings]\n"
           "excluded ratings can be (separated by space):\n"
           "  s = safe, q = questionable, e = nsfw")
    print(msg % (sys.argv)[0])
    return

  tag = sys.argv[1]
  solo = sys.argv[2]
  excluded_ratings = sys.argv[3:]

  exts = ["jpg", "png", "bmp", "gif", "jpeg"]

  page_num = 1
  init_url = "https://danbooru.donmai.us/posts?ms=1"

  path = "../data/train/%s" % (tag)
  if (not os.path.exists(path)):
    os.makedirs(path, 0755)

  done = False
  while (not done):
    tag_url = "%s&page=%s&tags=%s+%s" % (init_url, str(page_num), tag, solo)
    print(tag_url)

    page_data = requests.get(tag_url)
    page_tree = html.fromstring(page_data.content)
    print(page_tree)

    # get urls to images
    posts = page_tree.xpath('//article[starts-with(@class, "post-preview")]')
    for post in posts:
      img_url = post.attrib["data-file-url"]

      # s = safe, q = questionable, e = ecchi
      img_rating = post.attrib["data-rating"]
      include = img_rating not in excluded_ratings

      if ((img_url.split('.')[-1] in exts) and include):
        print(img_url)
        fname = img_url.split('/')[-1]
        fpath = "%s/%s" % (path, fname)

        # only dl file if we don't already have it
        if (os.path.isfile(fpath)):
          print("skipping")
        else:
          r = requests.get(img_url)
          with open(fpath, "wb") as f:
            f.write(r.content)

          time.sleep(random.randint(1,3))

    # get pages
    next_page_a = page_tree.xpath('//a[@id="paginator-next"]')
    if (len(next_page_a) == 0):
      done = True

    page_num = page_num + 1

main()
