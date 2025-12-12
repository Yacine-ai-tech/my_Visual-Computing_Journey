# Work in progress - Feature Detection
# Started: December 2024
# Status: Learning phase, code is messy

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Trying to understand SIFT (Scale-Invariant Feature Transform)
# Reading: https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html

# Load two images for matching (will find good test images later)
img1 = cv2.imread('../morphological_operations/cameraman.tif', 0)

# Create SIFT detector
# Note: SIFT is patented, for commercial use might need ORB instead
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
# keypoints: locations of interest points
# descriptors: mathematical description of the region around keypoint
keypoints, descriptors = sift.detectAndCompute(img1, None)

print(f"Found {len(keypoints)} keypoints")
print(f"Descriptor shape: {descriptors.shape}")

# Draw keypoints on image
# flags options:
# - cv2.DRAW_MATCHES_FLAGS_DEFAULT: just location
# - cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS: size and orientation
img_keypoints = cv2.drawKeypoints(img1, keypoints, None, 
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=(10, 8))
plt.imshow(img_keypoints, cmap='gray')
plt.title(f'SIFT Keypoints ({len(keypoints)} found)')
plt.axis('off')
plt.savefig('sift_keypoints.png')
plt.show()

# Things to explore:
# - What do the descriptors actually represent?
# - How to match keypoints between two images?
# - How does this compare to ORB? (ORB is faster, patent-free)
# - Use case: image stitching, object recognition

# Next: Try matching keypoints between two images of same scene
# Then: Build a simple panorama stitcher

# QUESTIONS:
# - How many keypoints is "good"? 100? 1000?
# - Does image size affect number of keypoints?
# - What's the difference between different feature detectors?
#   SIFT, SURF, ORB, AKAZE, BRISK...

# TODO:
# [ ] Try ORB detector (faster, free)
# [ ] Match features between two images
# [ ] Filter matches (distance threshold)
# [ ] Draw matches visualization
# [ ] Calculate homography
# [ ] Image alignment/stitching

print("\nWork in progress... more to come!")
