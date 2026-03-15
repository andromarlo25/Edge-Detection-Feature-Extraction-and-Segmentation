import cv2 as cv
import numpy as np

# 1. Load image
img = cv.imread('Photos/dogs4.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# 2. Gaussian Blur (The "Blurred Image" step)
# This removes noise so the detector doesn't pick up "fake" edges
blur = cv.GaussianBlur(gray, (5, 5), 1.4)
cv.imshow('1. Blurred Image', blur)

# 3. Sobel Operator (Gradient Calculation)
# This finds the intensity gradient of the image in x and y directions
sobelx = cv.Sobel(blur, cv.CV_64F, 1, 0, ksize=3) # Horizontal
sobely = cv.Sobel(blur, cv.CV_64F, 0, 1, ksize=3) # Vertical
mag = np.sqrt(sobelx**2 + sobely**2)
mag = np.uint8(mag) # Convert back to 8-bit to display
cv.imshow('2. Sobel Gradient Magnitude', mag)

# 4. Canny (Includes Non-Maxima Suppression & Hysteresis)
# OpenCv's Canny function performs NMS and Hysteresis internally
# threshold1 = Low Threshold (Hysteresis)
# threshold2 = High Threshold (Hysteresis)
edges = cv.Canny(blur, 100, 200)
cv.imshow('3.Canny Edge Detection', edges)

cv.waitKey(0)
cv.destroyAllWindows()