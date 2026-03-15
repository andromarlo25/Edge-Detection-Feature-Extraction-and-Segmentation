
import cv2 as cv
import numpy as np
 
# Load image in grayscale
img = cv.imread('Photos/bigdog.jpg')
cv.imshow('bigdog', img)
 
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Apply Sobel operator
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)  # Horizontal edges
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)  # Vertical edges
 
# Compute gradient magnitude
gradient_magnitude = cv.magnitude(sobelx, sobely)
 
# Convert to uint8
gradient_magnitude = cv.convertScaleAbs(gradient_magnitude)
 
# Display result
cv.imshow("Sobel Edge Detection", gradient_magnitude)
 
cv.waitKey(0)
cv.destroyAllWindows()