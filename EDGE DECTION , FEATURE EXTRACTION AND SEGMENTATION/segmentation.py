import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD THE IMAGE
img = cv2.imread("Photos/fruit1.jpg") 

if img is None:
    print("Error: Could not load image. Check the file path!")
else:
    # 2. PRE-PROCESSING
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)

    # 3. WATERSHED STEPS (Calculation)
    sure_bg = cv2.dilate(bin_img, kernel, iterations=3)
    dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)  
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 4. MARKER LABELLING
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 5. WATERSHED ALGORITHM
    # This modifies the markers image
    markers_copy = markers.copy() # Keep a copy for visualization
    markers = cv2.watershed(img, markers)

    # 6. CONTOUR EXTRACTION
    labels = np.unique(markers)
    coins = []
    for label in labels:
        # 0 is boundary, 1 is background, so we want labels 2 and above
        if label < 2:
            continue
        
        target = np.where(markers == label, 255, 0).astype(np.uint8)
        contours, hierarchy = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            coins.append(contours[0])

    # Draw the final outlines
    img_outlines = cv2.drawContours(img.copy(), coins, -1, color=(0, 23, 223), thickness=2)

    # ###########################################################
    # 7. FINAL VISUALIZATION (Everything in order)
    # ###########################################################

    # Figure 1: The 4-Step Process
    fig1, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    axes[0, 0].imshow(sure_bg, cmap='gray')
    axes[0, 0].set_title('Sure Background')
    axes[0, 1].imshow(dist, cmap='gray')
    axes[0, 1].set_title('Distance Transform')
    axes[1, 0].imshow(sure_fg, cmap='gray')
    axes[1, 0].set_title('Sure Foreground')
    axes[1, 1].imshow(unknown, cmap='gray')
    axes[1, 1].set_title('Unknown')

    # Figure 2: The Marker Map
    fig2, ax_markers = plt.subplots(figsize=(6, 6))
    ax_markers.imshow(markers, cmap="tab20b")
    ax_markers.set_title('Watershed Markers')
    ax_markers.axis('off')

    # Figure 3: Final Result with Outlines
    fig3, ax_final = plt.subplots(figsize=(6, 6))
    # Convert BGR (OpenCV) to RGB (Matplotlib)
    ax_final.imshow(cv2.cvtColor(img_outlines, cv2.COLOR_BGR2RGB))
    ax_final.set_title(f'Final Result: {len(coins)} Coins Detected')
    ax_final.axis('off')

    plt.show()