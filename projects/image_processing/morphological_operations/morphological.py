import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the cameraman image - classic test image in image processing
imgpath = "cameraman.tif"
img = cv2.imread(imgpath, 0)  # Read as grayscale

# Define kernel for morphological operations
# Started with 3x3 but the effects were too subtle
# 5x5 with 2 iterations gives visible but not excessive results
kernel_size = (5, 5)
iterations = 2
k = np.ones(kernel_size, np.uint8)

# Apply morphological operations
# Erosion: shrinks bright regions, useful for removing small noise
erosion = cv2.erode(img, k, iterations=iterations)

# Dilation: expands bright regions, opposite of erosion
dilation = cv2.dilate(img, k, iterations=iterations)

# Gradient: difference between dilation and erosion
# This basically gives us the edges/boundaries
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, k)

# Note to self: there's also MORPH_OPENING and MORPH_CLOSING
# Opening = erosion followed by dilation (removes noise)
# Closing = dilation followed by erosion (fills holes)
# Should try those next

# Setup for visualization
titles = ['Original', 'Erosion', 'Dilation', 'Gradient']
imgs = [img, erosion, dilation, gradient]

# Display all results in a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.ravel()  # Flatten to 1D array for easier iteration

for i in range(4):
    axes[i].imshow(imgs[i], cmap='gray')
    axes[i].set_title(titles[i])
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# Save results for comparison
# Useful to see the effects without re-running
cv2.imwrite('erosion_result.tif', erosion)
cv2.imwrite('dilation_result.tif', dilation)
cv2.imwrite('gradient_result.tif', gradient)

print(f"Processed image with kernel size {kernel_size} and {iterations} iterations")
print("Results saved to disk")

# Observations:
# - Erosion makes the cameraman look thinner, removes small details
# - Dilation makes him look "fatter", fills in small gaps
# - Gradient highlights the edges/contours nicely
# - Increasing iterations amplifies the effect
