import numpy as np
import os
import sys
import cv2
from matplotlib import pyplot as plt

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

  # If there are no lines to draw, exit.
  if lines is None:
    return

  # Make a copy of the original image.
  img = np.copy(img)

  # Create a blank image that matches the original in size.
  line_img = np.zeros((img.shape[0], img.shape[1], 3),  dtype=np.uint8,)

  # Loop over all lines and draw them on the blank image.
  for line in lines:
    for x1, y1, x2, y2 in line:
      cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)

  print(line_img.shape)
  print(img.shape)

  img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

  # Merge the image with the lines onto the original.
  img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

  # Show keypoints
  cv2.imshow("lines", img)
  cv2.waitKey(0)

  # Return the modified image.
  return img
# END draw_lines

def blob_detect(img):
  # Setup SimpleBlobDetector parameters.
  params = cv2.SimpleBlobDetector_Params()

  # Change thresholds
  params.minThreshold = 50
  params.maxThreshold = 200

  # Filter by Area.
  params.filterByArea = True
  params.minArea = 15

  # Filter by Circularity
  params.filterByCircularity = True
  params.minCircularity = 0.1

  # Filter by Convexity
  params.filterByConvexity = True
  params.minConvexity = 0.87

  # Filter by Inertia
  params.filterByInertia = True
  params.minInertiaRatio = 0.001

  # Create a detector with the parameters
  detector = cv2.SimpleBlobDetector_create(params)
 
  # Detect blobs.
  keypoints = detector.detect(img)
 
  img_keypoints = \
    cv2.drawKeypoints(img,
                      keypoints,
                      np.array([]),
                      (0, 0, 255),
                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
  # Show keypoints
  cv2.imshow("Keypoints", img_keypoints)
  cv2.waitKey(0)

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

main()
