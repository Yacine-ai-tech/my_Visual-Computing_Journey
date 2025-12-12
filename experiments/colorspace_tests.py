# Random test - trying different color spaces
# Not sure which is best for segmentation yet

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image (need to find a colorful one)
# Using the shapes image for now
img = cv2.imread('../Contour_detection/shape_for_test.jpeg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to different color spaces
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display all
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title('Original (RGB)')
axes[0, 0].axis('off')

# HSV channels
axes[0, 1].imshow(img_hsv[:,:,0], cmap='hsv')  # Hue
axes[0, 1].set_title('HSV - Hue')
axes[0, 1].axis('off')

axes[0, 2].imshow(img_hsv[:,:,1], cmap='gray')  # Saturation
axes[0, 2].set_title('HSV - Saturation')
axes[0, 2].axis('off')

axes[1, 0].imshow(img_hsv[:,:,2], cmap='gray')  # Value
axes[1, 0].set_title('HSV - Value')
axes[1, 0].axis('off')

# LAB - not sure what L, A, B stand for yet... need to look up
axes[1, 1].imshow(img_lab[:,:,0], cmap='gray')  # L
axes[1, 1].set_title('LAB - L channel')
axes[1, 1].axis('off')

axes[1, 2].imshow(img_gray, cmap='gray')
axes[1, 2].set_title('Grayscale')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('colorspace_comparison.png')
plt.show()

# Notes:
# HSV seems useful for color-based segmentation
# Hue channel isolates color from brightness
# Saturation shows how "colorful" each pixel is
# Value is similar to grayscale but not quite the same
# 
# TODO: Try color-based segmentation using HSV
# Example: isolate all red objects or blue objects
# Should be easier in HSV than RGB
