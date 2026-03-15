import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display

def imshow(img):
    # Safety Check: Did the image actually load?
    if img is None:
        print("❌ ERROR: The image file was not found. Check your folder path!")
        return
    cv2.imshow('Coin Image', img)
    cv2.waitKey(0) # This keeps the window open until you press a key
    cv2.destroyAllWindows()
    
    ret, encoded = cv2.imencode(".jpg", img)
    display(Image(encoded))

# Try to load the image
img = cv2.imread("Photos/coins.jpg")

# If it's not in the Photos folder, try the main folder
if img is None:
    img = cv2.imread("coins.jpg")
#image grayscale conversion
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imshow(gray)

#Threshold Processing
ret, bin_img = cv2.threshold(gray,
                             0, 255, 
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
imshow(bin_img)

# noise removal
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
bin_img = cv2.morphologyEx(bin_img, 
                           cv2.MORPH_OPEN,
                           kernel,
                           iterations=2)
imshow(bin_img)

# Run the function
imshow(img)