import numpy as np
import os
import sys
import cv2
import random
import imutils

from matplotlib import pyplot as plt
from kmeans import KMeans

RAW_IMG_SIZE_X = 1920
RAW_IMG_SIZE_Y = 1080
MAX_EPOCH = 100

def center_contours(img):
  gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  thresh  = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]

  cv2.imshow("eroded", thresh)
  cv2.waitKey(0)

  cnts = cv2.findContours(thresh.copy(),
                          cv2.RETR_EXTERNAL,
                          cv2.CHAIN_APPROX_SIMPLE)

  cnts = imutils.grab_contours(cnts)

  # store list of contour center pts
  contour_pts = []

  for c in cnts:
    # dropout on contours that are too small/noise
    if (cv2.contourArea(c) < 400):
      continue

    # compute the center of the contour
    M = cv2.moments(c)

    if (M["m00"] != 0):
      cX = int(M["m10"] / M["m00"])
      cY = int(M["m01"] / M["m00"])
    else:
      cX = 0
      cY = 0

    contour_pts.append((cX, cY));

    # draw the contour and center of the shape on the image
    cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
    cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(img, "center", (cX - 20, cY - 20),
      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

  # show image
  cv2.imshow("Image", img)
  cv2.waitKey(0)

  # perform kmeans on contour points
  km = KMeans(MAX_EPOCH, False)
  km.add_data_pts(contour_pts)

  km.add_cluster_pt([random.randint(0, RAW_IMG_SIZE_X),
                     random.randint(0, RAW_IMG_SIZE_Y)], "CENTER1")

  km.add_cluster_pt([random.randint(0, RAW_IMG_SIZE_X),
                     random.randint(0, RAW_IMG_SIZE_Y)], "CENTER2")

  km.add_cluster_pt([random.randint(0, RAW_IMG_SIZE_X),
                     random.randint(0, RAW_IMG_SIZE_Y)], "CENTER3")

  km.add_cluster_pt([random.randint(0, RAW_IMG_SIZE_X),
                     random.randint(0, RAW_IMG_SIZE_Y)], "CENTER4")

  km.add_cluster_pt([random.randint(0, RAW_IMG_SIZE_X),
                     random.randint(0, RAW_IMG_SIZE_Y)], "CENTER5")

  km.run_alg()
  km.print_cluster_pts()

  for cp in km.cluster_pts:
    cN = cp[1]
    cX = int(cp[0][0])
    cY = int(cp[0][1])

    # draw the contour and center of the shape on the image
    cv2.circle(img, (cX, cY), 10, (255, 0, 0), -1)
    cv2.putText(img, cN, (cX - 20, cY - 20),
      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

  # show image
  cv2.imshow("Image", img)
  cv2.waitKey(0)

def kmeans_img(img):
  km = KMeans(MAX_EPOCH, False)

  pts = []
  for i in range(RAW_IMG_SIZE_X):
    for j in range(RAW_IMG_SIZE_Y):
      if (img[j][i] == 255):
        pts.append((i, j))

  km.add_data_pts(pts)

  km.add_cluster_pt([random.randint(0, RAW_IMG_SIZE_X),
                     random.randint(0, RAW_IMG_SIZE_Y)], "red")

  km.add_cluster_pt([random.randint(0, RAW_IMG_SIZE_X),
                     random.randint(0, RAW_IMG_SIZE_Y)], "green")

  km.add_cluster_pt([random.randint(0, RAW_IMG_SIZE_X),
                     random.randint(0, RAW_IMG_SIZE_Y)], "blue")

  km.add_cluster_pt([random.randint(0, RAW_IMG_SIZE_X),
                     random.randint(0, RAW_IMG_SIZE_Y)], "yellow")
  km.run_alg()
  km.print_cluster_pts()
# END kmeans_img

def erode_img(img):
  # inv edges
  edges = cv2.Canny(img, 100, 200, 200)
  edges_inv = cv2.bitwise_not(edges)

  # kernel = np.ones((9, 9), np.uint8)
  kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (6,6))

  # erosion to thicken outlines
  erosion = cv2.erode(edges_inv, kernel, iterations=3)
  erosion = cv2.bitwise_not(erosion)

  return erosion
# END erode_img

def draw_lines(img, edges, color=[0, 0, 255], thickness=3):

  lines = cv2.HoughLinesP(
    edges,
    rho=6,
    theta=np.pi / 60,
    threshold=160,
    lines=np.array([]),
    minLineLength=40,
    maxLineGap=25
  )

  if lines is None:
    return

  img = np.copy(img)
  line_img = np.zeros((img.shape[0], img.shape[1], 3),  dtype=np.uint8,)

  for line in lines:
    for x1, y1, x2, y2 in line:
      cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)

  print(line_img.shape)
  print(img.shape)

  img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

  # combine img
  img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

  cv2.imshow("lines", img)
  cv2.waitKey(0)

  return img
# END draw_lines

def blob_detect(img):
  params = cv2.SimpleBlobDetector_Params()

  params.minThreshold = 50
  params.maxThreshold = 200
  params.filterByArea = True
  params.minArea = 15
  params.filterByCircularity = True
  params.minCircularity = 0.1
  params.filterByConvexity = True
  params.minConvexity = 0.87
  params.filterByInertia = True
  params.minInertiaRatio = 0.001

  detector = cv2.SimpleBlobDetector_create(params)
 
  keypoints = detector.detect(img)
 
  img_keypoints = \
    cv2.drawKeypoints(img,
                      keypoints,
                      np.array([]),
                      (0, 0, 255),
                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
  cv2.imshow("Keypoints", img_keypoints)
  cv2.waitKey(0)
# END blob_detect

# overlay should be grayscale, img should be rgb
def alpha_blend(overlay, img):
  img = np.float32(img)
  print(overlay.shape)
  print(img.shape)

  overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
  overlay = np.float32(overlay) / 255.0

  print(img)

  cv2.imshow("over", overlay)
  cv2.waitKey(0)

  print(overlay.shape)
  print(img.shape)

  cvt = np.uint8(cv2.multiply(img, overlay))
  print(cvt)

  #print(overlay.shape)
  #print(img.shape)
  #cvt = cv2.cvtColor(overlay, img, cv2.COLOR_GRAY2BGR)
 
  cv2.imshow("cvt", cvt)
  cv2.waitKey(0)

  return cvt
# END alpha_blend

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
  img_bw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

  blob_detect(img_bw)

  #t, img = cv2.threshold(img_bw, 200, 255, cv2.THRESH_BINARY_INV)

  cv2.imshow("image", img)
  cv2.waitKey(0)

  th = adaptive_thresh(img_bw)
  alpha_blend(th, img)

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
  erosion = cv2.bitwise_not(erosion)

  cv2.imshow("erosion", erosion)
  cv2.waitKey(0)

  draw_lines(img_bw, erosion)

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

if __name__ == "__main__":
  main()
