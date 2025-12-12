# Quick experiments and tests
# Not polished code - just trying things out

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Experiment 1: Different threshold methods on the same image
# Trying to figure out which works best

img = cv2.imread('../morphological_operations/cameraman.tif', 0)

# Simple threshold - requires manual value
ret1, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Otsu's method - automatically finds optimal threshold
# Supposed to work well for bimodal images
ret2, thresh2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptive threshold - what I ended up using in contour detection
thresh3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)

# Display
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
images = [img, thresh1, thresh2, thresh3]
titles = ['Original', 'Simple (127)', f'Otsu ({ret2:.0f})', 'Adaptive']

for i, (ax, image, title) in enumerate(zip(axes, images, titles)):
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.savefig('threshold_comparison.png')
plt.show()

print(f"Otsu's method found optimal threshold: {ret2}")

# Observations:
# - Simple threshold at 127: too dark, loses detail
# - Otsu's method: found threshold around 120-130, looks better
# - Adaptive: best for uneven lighting, which is common in real images
# Conclusion: Use adaptive for most cases, Otsu's for uniform lighting
