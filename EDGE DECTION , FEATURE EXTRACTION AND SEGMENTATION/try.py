import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 1. Load image
img = cv.imread('Photos/block.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 2. Detect corners
# Parameters: (image, maxCorners, qualityLevel, minDistance)
corners = cv.goodFeaturesToTrack(gray, 25, 0.01, 10)

# 3. Convert corners to integers (Fixed typo: int0)
corners = np.int64(corners)

# 4. Draw the corners
for i in corners:
    x, y = i.ravel()
    cv.circle(img, (x, y), 3, (0, 255, 0), -1) # Green dots

# 5. Convert BGR to RGB for Matplotlib display
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title('Shi-Tomasi Corner Detection')
plt.axis('off') # Hide axes for a cleaner look
plt.show()