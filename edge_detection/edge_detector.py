import cv2
import numpy as np
import matplotlib.pyplot as plt

# Edge Detection Comparison
# Trying out different edge detection methods to see which works best

# Load image - using the cameraman image again since it's handy
img = cv2.imread('../morphological_operations/cameraman.tif', 0)

if img is None:
    print("Error: Could not load image")
    exit()

# Method 1: Canny Edge Detection
# This is supposed to be the most popular edge detector
# Parameters: low_threshold, high_threshold
# Tried different values: (50,150), (100,200), settled on 50,150
edges_canny = cv2.Canny(img, 50, 150)

# Method 2: Sobel Edge Detection
# Computes gradient in x and y directions
# Then combines them
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # Horizontal edges
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # Vertical edges

# Combine x and y gradients
# Using magnitude: sqrt(gx^2 + gy^2)
edges_sobel = np.sqrt(sobelx**2 + sobely**2)
edges_sobel = np.uint8(edges_sobel)  # Convert back to uint8 for display

# Method 3: Laplacian Edge Detection
# Second derivative method - detects edges where intensity changes rapidly
edges_laplacian = cv2.Laplacian(img, cv2.CV_64F)
edges_laplacian = np.uint8(np.abs(edges_laplacian))  # Take absolute value and convert

# TODO: Try Scharr operator - supposedly better than Sobel for small kernels

# Display all results
titles = ['Original', 'Canny', 'Sobel', 'Laplacian']
images = [img, edges_canny, edges_sobel, edges_laplacian]

plt.figure(figsize=(12, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.show()

# Save results for comparison
cv2.imwrite('canny_edges.png', edges_canny)
cv2.imwrite('sobel_edges.png', edges_sobel)
cv2.imwrite('laplacian_edges.png', edges_laplacian)

print("Edge detection complete!")
print(f"Canny - clean edges, less noise")
print(f"Sobel - captures directional edges well")
print(f"Laplacian - very sensitive, picks up noise too")

# My observations:
# - Canny gives the cleanest results, good edge continuity
# - Sobel shows directional information, useful for gradient analysis
# - Laplacian is very sensitive to noise, maybe need more preprocessing?
# - For most applications, Canny seems like the best starting point
