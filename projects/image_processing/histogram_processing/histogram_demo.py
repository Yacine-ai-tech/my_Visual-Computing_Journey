"""
Histogram Processing and Color Analysis
Comprehensive histogram techniques for image enhancement and analysis
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_test_images():
    """Create test images with different lighting conditions"""
    # Dark image
    dark_img = np.ones((300, 400, 3), dtype=np.uint8) * 50
    cv2.circle(dark_img, (200, 150), 80, (100, 100, 100), -1)
    cv2.rectangle(dark_img, (50, 50), (150, 250), (80, 80, 80), -1)
    
    # Bright image
    bright_img = np.ones((300, 400, 3), dtype=np.uint8) * 200
    cv2.circle(bright_img, (200, 150), 80, (220, 220, 220), -1)
    cv2.rectangle(bright_img, (50, 50), (150, 250), (230, 230, 230), -1)
    
    # Low contrast
    low_contrast = np.ones((300, 400, 3), dtype=np.uint8) * 128
    cv2.circle(low_contrast, (200, 150), 80, (140, 140, 140), -1)
    cv2.rectangle(low_contrast, (50, 50), (150, 250), (150, 150, 150), -1)
    
    # Normal image with colors
    normal_img = np.ones((300, 400, 3), dtype=np.uint8) * 128
    cv2.circle(normal_img, (200, 150), 80, (0, 0, 255), -1)
    cv2.rectangle(normal_img, (50, 50), (150, 250), (0, 255, 0), -1)
    cv2.ellipse(normal_img, (320, 230), (60, 40), 0, 0, 360, (255, 0, 0), -1)
    
    return dark_img, bright_img, low_contrast, normal_img

def plot_histogram(img, title="Histogram"):
    """Calculate and plot histogram for BGR channels"""
    colors = ('b', 'g', 'r')
    plt.figure(figsize=(10, 4))
    
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, label=f'{color.upper()} channel')
    
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 256])
    
def histogram_equalization_gray(img):
    """
    Histogram equalization for grayscale images
    Enhances contrast by spreading out intensity values
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization
    equalized = cv2.equalizeHist(gray)
    
    # Convert back to BGR for display
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    equalized_bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    
    return gray_bgr, equalized_bgr

def histogram_equalization_color(img):
    """
    Histogram equalization for color images using different methods
    """
    results = {}
    
    # Method 1: Equalize each channel independently (not recommended)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    equalized_yuv = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    results['YUV Method'] = equalized_yuv
    
    # Method 2: Equalize in HSV (V channel only)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[:,:,2] = cv2.equalizeHist(img_hsv[:,:,2])
    equalized_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    results['HSV Method'] = equalized_hsv
    
    # Method 3: Equalize each BGR channel (for comparison)
    equalized_bgr = img.copy()
    for i in range(3):
        equalized_bgr[:,:,i] = cv2.equalizeHist(img[:,:,i])
    results['BGR Method'] = equalized_bgr
    
    return results

def adaptive_histogram_equalization(img):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)
    Better than global equalization for images with varying lighting
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Apply CLAHE to L channel
    cl = clahe.apply(l)
    
    # Merge channels
    merged = cv2.merge([cl, a, b])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    # Also create versions with different parameters
    clahe_strong = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    cl_strong = clahe_strong.apply(l)
    merged_strong = cv2.merge([cl_strong, a, b])
    enhanced_strong = cv2.cvtColor(merged_strong, cv2.COLOR_LAB2BGR)
    
    clahe_fine = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    cl_fine = clahe_fine.apply(l)
    merged_fine = cv2.merge([cl_fine, a, b])
    enhanced_fine = cv2.cvtColor(merged_fine, cv2.COLOR_LAB2BGR)
    
    return {
        'CLAHE Standard': enhanced,
        'CLAHE Strong': enhanced_strong,
        'CLAHE Fine Grid': enhanced_fine
    }

def histogram_matching(source, reference):
    """
    Match histogram of source image to reference image
    Useful for color transfer and style matching
    """
    # Convert to LAB color space
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
    
    # Split channels
    source_l, source_a, source_b = cv2.split(source_lab)
    ref_l, ref_a, ref_b = cv2.split(reference_lab)
    
    # Match histograms for each channel
    matched_l = match_histograms_channel(source_l, ref_l)
    matched_a = match_histograms_channel(source_a, ref_a)
    matched_b = match_histograms_channel(source_b, ref_b)
    
    # Merge and convert back
    matched_lab = cv2.merge([matched_l, matched_a, matched_b])
    matched_bgr = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)
    
    return matched_bgr

def match_histograms_channel(source, reference):
    """Match histogram of one channel"""
    # Calculate histograms
    source_hist, _ = np.histogram(source.flatten(), 256, [0, 256])
    reference_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])
    
    # Calculate cumulative distributions
    source_cdf = source_hist.cumsum()
    reference_cdf = reference_hist.cumsum()
    
    # Normalize
    source_cdf = source_cdf / source_cdf[-1]
    reference_cdf = reference_cdf / reference_cdf[-1]
    
    # Create lookup table
    lookup_table = np.zeros(256, dtype=np.uint8)
    
    for i in range(256):
        # Find closest match in reference CDF
        diff = np.abs(reference_cdf - source_cdf[i])
        lookup_table[i] = np.argmin(diff)
    
    # Apply lookup table
    matched = cv2.LUT(source, lookup_table)
    
    return matched

def color_histogram_analysis(img):
    """
    Analyze color distribution in image
    """
    # Convert to different color spaces
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Calculate 2D histogram (Hue vs Saturation)
    hist_2d = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    
    # Normalize for visualization
    hist_2d_norm = cv2.normalize(hist_2d, None, 0, 255, cv2.NORM_MINMAX)
    
    # Calculate color moments
    mean_bgr = img.mean(axis=(0, 1))
    std_bgr = img.std(axis=(0, 1))
    
    return hist_2d_norm, mean_bgr, std_bgr

def gamma_correction(img, gamma=1.0):
    """
    Adjust image brightness using gamma correction
    gamma < 1: brighten
    gamma > 1: darken
    """
    # Build lookup table
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                      for i in range(256)]).astype("uint8")
    
    # Apply gamma correction
    corrected = cv2.LUT(img, table)
    
    return corrected

def contrast_stretching(img):
    """
    Stretch contrast to use full dynamic range
    """
    # Calculate min and max for each channel
    result = np.zeros_like(img)
    
    for i in range(3):
        channel = img[:,:,i]
        min_val = channel.min()
        max_val = channel.max()
        
        # Stretch to [0, 255]
        if max_val > min_val:
            result[:,:,i] = ((channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            result[:,:,i] = channel
    
    return result

def main():
    """Main function demonstrating histogram processing techniques"""
    print("=" * 70)
    print("Histogram Processing and Color Analysis")
    print("=" * 70)
    
    # Create test images
    print("\nCreating test images with different characteristics...")
    dark, bright, low_contrast, normal = create_test_images()
    print("   ✓ Test images created")
    
    # 1. Histogram Visualization
    print("\n1. Histogram Analysis...")
    
    fig = plt.figure(figsize=(16, 10))
    
    test_images = [
        (dark, "Dark Image"),
        (bright, "Bright Image"),
        (low_contrast, "Low Contrast"),
        (normal, "Normal Image")
    ]
    
    for idx, (img, title) in enumerate(test_images):
        # Show image
        plt.subplot(4, 3, idx * 3 + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        
        # Show histogram
        plt.subplot(4, 3, idx * 3 + 2)
        for i, color in enumerate(['b', 'g', 'r']):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
        plt.title(f'{title} - Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Show cumulative histogram
        plt.subplot(4, 3, idx * 3 + 3)
        for i, color in enumerate(['b', 'g', 'r']):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max() / cdf.max()
            plt.plot(cdf_normalized, color=color)
        plt.title(f'{title} - CDF')
        plt.xlabel('Pixel Value')
        plt.ylabel('Cumulative Frequency')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('histogram_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("   ✓ Histogram analysis complete")
    
    # 2. Histogram Equalization
    print("\n2. Histogram Equalization...")
    
    fig = plt.figure(figsize=(16, 8))
    
    # Apply to dark image
    gray_dark, eq_dark = histogram_equalization_gray(dark)
    
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(dark, cv2.COLOR_BGR2RGB))
    plt.title('Original Dark Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(eq_dark, cv2.COLOR_BGR2RGB))
    plt.title('After Equalization')
    plt.axis('off')
    
    # Histograms
    plt.subplot(2, 3, 3)
    hist_before = cv2.calcHist([cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY)], 
                               [0], None, [256], [0, 256])
    hist_after = cv2.calcHist([cv2.cvtColor(eq_dark, cv2.COLOR_BGR2GRAY)], 
                              [0], None, [256], [0, 256])
    plt.plot(hist_before, 'b-', label='Before')
    plt.plot(hist_after, 'r-', label='After')
    plt.title('Histogram Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Color equalization methods
    color_eq_results = histogram_equalization_color(normal)
    
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(normal, cv2.COLOR_BGR2RGB))
    plt.title('Original Color Image')
    plt.axis('off')
    
    for idx, (method, result) in enumerate(list(color_eq_results.items())[:2], 5):
        plt.subplot(2, 3, idx)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(method)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('histogram_equalization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("   ✓ Histogram equalization complete")
    
    # 3. CLAHE (Adaptive Equalization)
    print("\n3. CLAHE (Adaptive Histogram Equalization)...")
    
    clahe_results = adaptive_histogram_equalization(dark)
    
    fig = plt.figure(figsize=(16, 4))
    
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(dark, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')
    
    for idx, (method, result) in enumerate(clahe_results.items(), 2):
        plt.subplot(1, 4, idx)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(method)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('clahe_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("   ✓ CLAHE processing complete")
    
    # 4. Gamma Correction
    print("\n4. Gamma Correction...")
    
    fig = plt.figure(figsize=(16, 4))
    
    gamma_values = [0.5, 1.0, 1.5, 2.0]
    
    for idx, gamma in enumerate(gamma_values, 1):
        corrected = gamma_correction(dark, gamma)
        plt.subplot(1, 4, idx)
        plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
        plt.title(f'Gamma = {gamma}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('gamma_correction.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("   ✓ Gamma correction complete")
    
    # 5. Contrast Stretching
    print("\n5. Contrast Stretching...")
    
    stretched = contrast_stretching(low_contrast)
    
    fig = plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(low_contrast, cv2.COLOR_BGR2RGB))
    plt.title('Low Contrast Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(stretched, cv2.COLOR_BGR2RGB))
    plt.title('After Contrast Stretching')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('contrast_stretching.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("   ✓ Contrast stretching complete")
    
    # 6. Color Distribution Analysis
    print("\n6. Color Distribution Analysis...")
    
    hist_2d, mean_bgr, std_bgr = color_histogram_analysis(normal)
    
    fig = plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(normal, cv2.COLOR_BGR2RGB))
    plt.title('Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(hist_2d, cmap='hot', aspect='auto')
    plt.xlabel('Saturation')
    plt.ylabel('Hue')
    plt.title('2D Histogram (H-S)')
    plt.colorbar(label='Frequency')
    
    plt.tight_layout()
    plt.savefig('color_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"   ✓ Mean BGR: {mean_bgr}")
    print(f"   ✓ Std BGR: {std_bgr}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Histogram Processing Summary")
    print("=" * 70)
    print("\nHistogram Equalization:")
    print("  • Spreads out intensity values")
    print("  • Enhances global contrast")
    print("  • Works best on low-contrast images")
    print("  • May over-enhance noise")
    
    print("\nCLAHE:")
    print("  • Adaptive local equalization")
    print("  • Prevents over-amplification")
    print("  • Better for images with varying lighting")
    print("  • Parameters: clipLimit, tileGridSize")
    
    print("\nGamma Correction:")
    print("  • gamma < 1: brighten image")
    print("  • gamma > 1: darken image")
    print("  • Non-linear transformation")
    print("  • Good for display calibration")
    
    print("\nContrast Stretching:")
    print("  • Linear transformation")
    print("  • Uses full dynamic range")
    print("  • Simple and fast")
    print("  • May amplify noise")
    
    print("\nApplications:")
    print("  • Medical image enhancement")
    print("  • Satellite image processing")
    print("  • Photography enhancement")
    print("  • Video quality improvement")
    print("  • Low-light image enhancement")
    
    print("\n" + "=" * 70)
    print("All histogram processing demonstrations completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
