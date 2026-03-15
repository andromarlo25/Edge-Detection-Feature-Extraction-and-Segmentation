import cv2 as cv
import numpy as np
 
# Load image in grayscale
img = cv.imread('Photos/bigdog.jpg')
cv.imshow('bigdog', img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
 
# Apply Laplacian operator
laplacian = cv.Laplacian(img, cv.CV_64F)
 
# Convert to uint8
laplacian_abs = cv.convertScaleAbs(laplacian)
 
# Display result
cv.imshow("Laplacian Edge Detection", laplacian_abs)
 
cv.waitKey(0)
cv.destroyAllWindows()