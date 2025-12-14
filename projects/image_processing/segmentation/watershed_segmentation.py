"""
Watershed Segmentation Implementation
Demonstrates marker-based watershed algorithm for image segmentation
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

def watershed_segmentation(image_path):
    """
    Performs watershed segmentation on an image
    
    Args:
        image_path: Path to input image
        
    Returns:
        Tuple of (original, markers, segmented) images
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        # Create a synthetic image with shapes for demonstration
        img = create_test_image()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Noise removal using morphological opening
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area - dilation
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area using distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    
    # Mark the region of unknown with zero
    markers[unknown == 255] = 0
    
    # Apply watershed algorithm
    markers = cv2.watershed(img, markers)
    
    # Create visualization
    img_result = img.copy()
    img_result[markers == -1] = [0, 0, 255]  # Mark boundaries in red
    
    # Create colored segmentation map
    segmented = np.zeros_like(img)
    for label in range(2, markers.max() + 1):
        mask = markers == label
        color = np.random.randint(0, 255, 3).tolist()
        segmented[mask] = color
    
    return img, markers, img_result, segmented

def create_test_image():
    """Creates a test image with multiple shapes"""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Draw several circles
    cv2.circle(img, (100, 100), 50, (0, 255, 0), -1)
    cv2.circle(img, (200, 100), 50, (0, 255, 0), -1)
    cv2.circle(img, (300, 100), 50, (0, 255, 0), -1)
    cv2.circle(img, (100, 250), 50, (0, 255, 0), -1)
    cv2.circle(img, (200, 250), 50, (0, 255, 0), -1)
    cv2.circle(img, (300, 250), 50, (0, 255, 0), -1)
    
    # Draw rectangles
    cv2.rectangle(img, (400, 50), (550, 150), (255, 0, 0), -1)
    cv2.rectangle(img, (400, 200), (550, 300), (255, 0, 0), -1)
    
    return img

def kmeans_segmentation(image_path, k=3):
    """
    Performs K-means clustering for image segmentation
    
    Args:
        image_path: Path to input image
        k: Number of clusters
        
    Returns:
        Tuple of (original, segmented) images
    """
    img = cv2.imread(image_path)
    if img is None:
        img = create_test_image()
    
    # Reshape image to 2D array of pixels
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Define criteria and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, 
                                     cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to 8-bit values
    centers = np.uint8(centers)
    
    # Flatten labels array
    labels = labels.flatten()
    
    # Convert all pixels to their nearest center value
    segmented_image = centers[labels]
    
    # Reshape back to original image dimensions
    segmented_image = segmented_image.reshape(img.shape)
    
    return img, segmented_image

def grabcut_segmentation(image_path):
    """
    Performs GrabCut segmentation for foreground extraction
    
    Args:
        image_path: Path to input image
        
    Returns:
        Tuple of (original, masked, foreground) images
    """
    img = cv2.imread(image_path)
    if img is None:
        img = create_test_image()
    
    # Create a mask
    mask = np.zeros(img.shape[:2], np.uint8)
    
    # Define rectangle containing the object
    # Using 10% margin from edges
    height, width = img.shape[:2]
    rect = (int(width * 0.1), int(height * 0.1), 
            int(width * 0.8), int(height * 0.8))
    
    # Background and foreground models (required by GrabCut)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Apply GrabCut algorithm
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Modify mask: pixels with value 0 or 2 are set to 0 (background)
    # pixels with value 1 or 3 are set to 1 (foreground)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Multiply image with mask to get foreground
    foreground = img * mask2[:, :, np.newaxis]
    
    # Create visualization with mask overlay
    masked_img = img.copy()
    masked_img[mask2 == 0] = [0, 0, 255]  # Red background
    
    return img, masked_img, foreground

def main():
    """Main function demonstrating all segmentation techniques"""
    print("=" * 60)
    print("Image Segmentation Techniques")
    print("=" * 60)
    
    # Test with synthetic image
    image_path = None
    
    # 1. Watershed Segmentation
    print("\n1. Performing Watershed Segmentation...")
    original, markers, boundaries, segmented = watershed_segmentation(image_path)
    
    plt.figure(figsize=(16, 4))
    plt.subplot(141)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(142)
    plt.imshow(markers, cmap='nipy_spectral')
    plt.title('Watershed Markers')
    plt.axis('off')
    
    plt.subplot(143)
    plt.imshow(cv2.cvtColor(boundaries, cv2.COLOR_BGR2RGB))
    plt.title('Boundaries Marked')
    plt.axis('off')
    
    plt.subplot(144)
    plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    plt.title('Segmented Regions')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('watershed_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("   ✓ Watershed segmentation complete")
    print(f"   - Found {markers.max() - 1} distinct regions")
    
    # 2. K-means Segmentation
    print("\n2. Performing K-means Segmentation...")
    plt.figure(figsize=(15, 5))
    
    for i, k in enumerate([2, 3, 5], 1):
        original, segmented = kmeans_segmentation(image_path, k)
        
        plt.subplot(1, 3, i)
        plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
        plt.title(f'K-means (k={k})')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('kmeans_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("   ✓ K-means segmentation complete")
    print("   - Tested with k=2, 3, and 5 clusters")
    
    # 3. GrabCut Segmentation
    print("\n3. Performing GrabCut Segmentation...")
    original, masked, foreground = grabcut_segmentation(image_path)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
    plt.title('Mask Overlay')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
    plt.title('Extracted Foreground')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('grabcut_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("   ✓ GrabCut segmentation complete")
    print("   - Foreground successfully extracted")
    
    print("\n" + "=" * 60)
    print("All segmentation techniques completed successfully!")
    print("Results saved as PNG files.")
    print("=" * 60)

if __name__ == "__main__":
    main()
