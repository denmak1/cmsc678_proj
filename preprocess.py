import numpy as np
import os
import sys
import cv2
from matplotlib import pyplot as plt

def alpha_blend(overlay, img):
  print(overlay.shape)
  print(img.shape)
  overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
  overlay = cv2.normalize(overlay, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

  cvt = cv2.multiply(img, overlay)

  #print(overlay.shape)
  #print(img.shape)
  #cvt = cv2.cvtColor(overlay, img, cv2.COLOR_GRAY2BGR)
 
  cv2.imshow("cvt", cvt)
  cv2.waitKey(0)

def adaptive_thresh(img):
  img = cv2.medianBlur(img, 5)
  ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

  th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                              cv2.THRESH_BINARY, 75, 10)

  th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                              cv2.THRESH_BINARY, 75, 10)

  titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
  images = [img, th1, th2, th3]

  for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

  plt.show()

  return th2
# END adaptive_thresh


def main():
  img_path = sys.argv[1]

  # read image
  img = cv2.imread(img_path)
  img_bw = cv2.imread(img_path, 0)

  #t, img = cv2.threshold(img_bw, 200, 255, cv2.THRESH_BINARY_INV)

  cv2.imshow("image", img)
  cv2.waitKey(0)

  th = adaptive_thresh(img_bw)
  alpha_blend(th, img)

  return 0

  # get edges using canny
  edges = cv2.Canny(img, 100, 200, 200)
  cv2.imshow("edges", edges)
  cv2.waitKey(0)

  # invert edges
  edges_inv = cv2.bitwise_not(edges)
  cv2.imshow("edges_inv", edges_inv)
  cv2.waitKey(0)

  # kernel = np.ones((9, 9), np.uint8)
  kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))

  # erosion
  erosion = cv2.erode(edges_inv, kernel, iterations=2)
  cv2.imshow("erosion", erosion)
  cv2.waitKey(0)

  # dilation
  dilation = cv2.dilate(edges, kernel, iterations=10)
  cv2.imshow("dilation", dilation)
  cv2.waitKey(0)

  # opening
  opening = cv2.morphologyEx(img_bw, cv2.MORPH_OPEN, kernel)
  cv2.imshow("opening", opening)
  cv2.waitKey(0)

  # morh grad
  gradient = cv2.morphologyEx(img_bw, cv2.MORPH_GRADIENT, kernel)
  cv2.imshow("gradient", gradient)
  cv2.waitKey(0)

  # morph + flat kernel + thresh
  kernel = np.ones((5, 5), np.uint8)

  morphed = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
  t, morphed = cv2.threshold(morphed, 200, 255, cv2.THRESH_TOZERO_INV)

  cv2.imshow("morph thresh", morphed)
  cv2.waitKey(0)

  # try to fill in edges
  edges_fill = cv2.bitwise_not(erosion)

  h, w = edges.shape[:2]
  mask = np.zeros((h+2, w+2), np.uint8)

  cv2.floodFill(edges_fill, mask, (0, 0), 255)
  edges_fill_inv = cv2.bitwise_not(edges_fill)
  out = edges | edges_fill_inv | gradient

  cv2.imshow("edges_fill", out)
  cv2.waitKey(0)

  cv2.destroyAllWindows()

main()
