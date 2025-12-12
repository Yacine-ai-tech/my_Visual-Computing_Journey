# Import necessary libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image - using relative path
img = cv2.imread('./shape_for_test.jpeg')

# OpenCV loads images in BGR, but matplotlib expects RGB
# This was a gotcha that took me forever to figure out!
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to grayscale for processing
# Contour detection works on single-channel images
gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
# This is crucial! Without blurring, you get tons of tiny contours from noise
# Kernel size must be odd - tried (3,3) and (7,7), settled on (5,5)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Adaptive thresholding - much better than global threshold
# Global threshold failed on images with varying lighting
# The parameters (11, 2) were found through trial and error:
# - 11 is the neighborhood size for adaptive calculation
# - 2 is the constant subtracted from mean
thresh = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)

# Detect contours
# RETR_TREE gets all contours and builds hierarchy
# CHAIN_APPROX_SIMPLE compresses contours to save memory
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on a copy of the original image
copy_img = img.copy()
# -1 means draw all contours, (0, 0, 255) is red in RGB, 2 is thickness
cv2.drawContours(copy_img, contours, -1, (0, 0, 255), 2)

# Add text annotation showing number of contours found
# This helps verify the detection is working correctly
cv2.putText(copy_img, f"Contours Detected: {len(contours)}", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Display original and processed images side by side
titles = ['Original', 'Contours']
imgs = [img, copy_img]

plt.figure(figsize=(10, 5))
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.xticks([])  # Remove tick marks for cleaner look
    plt.yticks([])
    plt.title(titles[i])
    plt.imshow(imgs[i])
plt.tight_layout()
plt.show()

# Debug info
print(f"Found {len(contours)} contours")
print(f"Image shape: {img.shape}")

# TODO: Try filtering contours by area to remove very small ones
# Could use cv2.contourArea() for this
# Also want to try cv2.boundingRect() to draw boxes around shapes
