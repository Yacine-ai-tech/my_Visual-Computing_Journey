# Testing different blur kernels to see effect on edge detection
# Spoiler: it matters A LOT

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../morphological_operations/cameraman.tif', 0)

# Try different blur kernel sizes before Canny edge detection
# Hypothesis: larger kernel = smoother edges, less noise

kernel_sizes = [3, 5, 7, 9, 11]
results = []

for k in kernel_sizes:
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (k, k), 0)
    
    # Apply Canny edge detection with same parameters
    edges = cv2.Canny(blurred, 50, 150)
    results.append(edges)

# Display all results
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

for i, (k, edges) in enumerate(zip(kernel_sizes, results)):
    axes[i+1].imshow(edges, cmap='gray')
    axes[i+1].set_title(f'Kernel: {k}x{k}')
    axes[i+1].axis('off')

plt.tight_layout()
plt.savefig('blur_kernel_comparison.png')
plt.show()

# Results:
# 3x3: Still quite noisy, picks up lots of small edges
# 5x5: Good balance - clean edges, preserves important details (MY CHOICE)
# 7x7: Smoother, starting to lose some fine details
# 9x9: Too smooth, missing some edges
# 11x11: Way too blurry, important details lost

# Takeaway: 5x5 or 7x7 usually work best
# Depends on image resolution and noise level
