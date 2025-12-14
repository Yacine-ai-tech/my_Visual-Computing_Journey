"""
Advanced Image Filtering and Enhancement
Comprehensive collection of filtering techniques for image quality improvement
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def create_test_image_with_noise():
    """Create test image with various types of noise"""
    # Create base image with shapes
    img = np.ones((400, 600, 3), dtype=np.uint8) * 128
    
    # Add shapes
    cv2.circle(img, (150, 150), 80, (255, 255, 255), -1)
    cv2.rectangle(img, (300, 50), (500, 250), (200, 200, 200), -1)
    cv2.ellipse(img, (400, 320), (100, 60), 0, 0, 360, (180, 180, 180), -1)
    
    # Add Gaussian noise
    gaussian_noise = np.random.normal(0, 25, img.shape)
    noisy_img = np.clip(img.astype(np.float64) + gaussian_noise, 0, 255).astype(np.uint8)
    
    # Add salt and pepper noise
    salt_pepper_img = noisy_img.copy()
    salt_pepper_ratio = 0.02
    
    # Salt noise (white)
    salt_coords = np.random.random(img.shape[:2]) < salt_pepper_ratio / 2
    salt_pepper_img[salt_coords] = 255
    
    # Pepper noise (black)
    pepper_coords = np.random.random(img.shape[:2]) < salt_pepper_ratio / 2
    salt_pepper_img[pepper_coords] = 0
    
    return img, noisy_img, salt_pepper_img

def gaussian_filtering(img):
    """
    Gaussian blur - removes Gaussian noise
    Good for: Smoothing, noise reduction
    """
    results = {}
    
    # Different kernel sizes
    for ksize in [3, 5, 9]:
        blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
        results[f'Gaussian {ksize}x{ksize}'] = blurred
    
    return results

def bilateral_filtering(img):
    """
    Bilateral filter - edge-preserving smoothing
    Good for: Noise reduction while preserving edges
    """
    results = {}
    
    # Different parameters
    params = [
        (9, 75, 75, "Standard"),
        (9, 150, 150, "Strong"),
        (15, 75, 75, "Large kernel")
    ]
    
    for d, sigmaColor, sigmaSpace, name in params:
        filtered = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
        results[f'Bilateral {name}'] = filtered
    
    return results

def median_filtering(img):
    """
    Median filter - excellent for salt and pepper noise
    Good for: Removing impulse noise
    """
    results = {}
    
    for ksize in [3, 5, 7]:
        filtered = cv2.medianBlur(img, ksize)
        results[f'Median {ksize}x{ksize}'] = filtered
    
    return results

def morphological_filtering(img):
    """
    Morphological filters - opening and closing
    Good for: Noise removal, hole filling
    """
    results = {}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    
    # Opening - removes small bright spots
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    results['Opening'] = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
    
    # Closing - removes small dark spots
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    results['Closing'] = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)
    
    # Morphological gradient - edge detection
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    results['Gradient'] = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
    
    # Top hat - bright features
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    results['Top Hat'] = cv2.cvtColor(tophat, cv2.COLOR_GRAY2BGR)
    
    # Black hat - dark features
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    results['Black Hat'] = cv2.cvtColor(blackhat, cv2.COLOR_GRAY2BGR)
    
    return results

def non_local_means_filtering(img):
    """
    Non-local means denoising - advanced noise reduction
    Good for: Texture preservation while denoising
    """
    results = {}
    
    # Fast version
    fast = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    results['NLM Fast'] = fast
    
    # Strong denoising
    strong = cv2.fastNlMeansDenoisingColored(img, None, 20, 20, 7, 21)
    results['NLM Strong'] = strong
    
    return results

def wiener_filtering(img):
    """
    Wiener filter approximation using frequency domain
    Good for: Deblurring, noise reduction
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Fourier transform
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    
    # Create Wiener-like filter
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create meshgrid
    x = np.arange(-ccol, cols - ccol)
    y = np.arange(-crow, rows - crow)
    X, Y = np.meshgrid(x, y)
    
    # Gaussian filter in frequency domain
    sigma = 30
    H = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    
    # Apply filter
    fshift_filtered = fshift * H
    
    # Inverse transform
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_filtered = np.fft.ifft2(f_ishift)
    img_filtered = np.abs(img_filtered)
    
    result = cv2.cvtColor(img_filtered.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    return {'Wiener-like': result}

def anisotropic_diffusion(img, num_iter=20, kappa=50, gamma=0.1):
    """
    Anisotropic diffusion (Perona-Malik) - edge-preserving smoothing
    Good for: Smoothing while preserving important edges
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    
    for _ in range(num_iter):
        # Calculate gradients
        north = np.roll(gray, -1, axis=0) - gray
        south = np.roll(gray, 1, axis=0) - gray
        east = np.roll(gray, -1, axis=1) - gray
        west = np.roll(gray, 1, axis=1) - gray
        
        # Calculate diffusion coefficients
        c_north = np.exp(-(north / kappa) ** 2)
        c_south = np.exp(-(south / kappa) ** 2)
        c_east = np.exp(-(east / kappa) ** 2)
        c_west = np.exp(-(west / kappa) ** 2)
        
        # Update image
        gray += gamma * (
            c_north * north + c_south * south +
            c_east * east + c_west * west
        )
    
    result = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return {'Anisotropic Diffusion': result}

def guided_filtering(img):
    """
    Guided filter - edge-preserving smoothing
    Good for: Detail enhancement, haze removal
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # OpenCV doesn't have built-in guided filter, using bilateral as alternative
    # In practice, you'd use cv2.ximgproc.guidedFilter if available
    result = cv2.bilateralFilter(img, 9, 75, 75)
    
    return {'Guided Filter (approx)': result}

def unsharp_masking(img):
    """
    Unsharp mask - sharpening filter
    Good for: Enhancing edges and details
    """
    # Gaussian blur
    blurred = cv2.GaussianBlur(img, (9, 9), 10.0)
    
    # Unsharp mask
    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    
    # Strong sharpening
    strong_sharpened = cv2.addWeighted(img, 2.0, blurred, -1.0, 0)
    
    return {
        'Unsharp Mask': sharpened,
        'Strong Sharpen': strong_sharpened
    }

def frequency_domain_filtering(img):
    """
    Frequency domain filters - high-pass and low-pass
    Good for: Frequency-based image enhancement
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Fourier transform
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    
    # Low-pass filter (blur)
    mask_low = np.zeros((rows, cols), np.uint8)
    r = 30
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask_low[mask_area] = 1
    
    fshift_low = fshift * mask_low
    f_ishift_low = np.fft.ifftshift(fshift_low)
    img_low = np.fft.ifft2(f_ishift_low)
    img_low = np.abs(img_low)
    
    # High-pass filter (sharpen)
    mask_high = np.ones((rows, cols), np.uint8)
    mask_high[mask_area] = 0
    
    fshift_high = fshift * mask_high
    f_ishift_high = np.fft.ifftshift(fshift_high)
    img_high = np.fft.ifft2(f_ishift_high)
    img_high = np.abs(img_high)
    
    return {
        'Low-pass (Freq)': cv2.cvtColor(img_low.astype(np.uint8), cv2.COLOR_GRAY2BGR),
        'High-pass (Freq)': cv2.cvtColor(img_high.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    }

def main():
    """Main function demonstrating all filtering techniques"""
    print("=" * 70)
    print("Advanced Image Filtering and Enhancement")
    print("=" * 70)
    
    # Create test images
    print("\nCreating test images with various noise types...")
    clean, gaussian_noisy, sp_noisy = create_test_image_with_noise()
    print("   ✓ Clean image created")
    print("   ✓ Gaussian noise added")
    print("   ✓ Salt & pepper noise added")
    
    # Display original images
    fig = plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
    plt.title('Clean Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(gaussian_noisy, cv2.COLOR_BGR2RGB))
    plt.title('Gaussian Noise')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(sp_noisy, cv2.COLOR_BGR2RGB))
    plt.title('Salt & Pepper Noise')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('noise_types.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Test all filters
    print("\n" + "=" * 70)
    print("Testing Filters")
    print("=" * 70)
    
    # 1. Gaussian filtering
    print("\n1. Gaussian Filtering...")
    gaussian_results = gaussian_filtering(gaussian_noisy)
    print(f"   ✓ Tested {len(gaussian_results)} variations")
    
    # 2. Bilateral filtering
    print("\n2. Bilateral Filtering...")
    bilateral_results = bilateral_filtering(gaussian_noisy)
    print(f"   ✓ Tested {len(bilateral_results)} variations")
    
    # 3. Median filtering
    print("\n3. Median Filtering...")
    median_results = median_filtering(sp_noisy)
    print(f"   ✓ Tested {len(median_results)} variations")
    
    # 4. Morphological filtering
    print("\n4. Morphological Filtering...")
    morph_results = morphological_filtering(gaussian_noisy)
    print(f"   ✓ Tested {len(morph_results)} variations")
    
    # 5. Non-local means
    print("\n5. Non-Local Means Filtering...")
    nlm_results = non_local_means_filtering(gaussian_noisy)
    print(f"   ✓ Tested {len(nlm_results)} variations")
    
    # 6. Frequency domain
    print("\n6. Frequency Domain Filtering...")
    freq_results = frequency_domain_filtering(gaussian_noisy)
    print(f"   ✓ Tested {len(freq_results)} variations")
    
    # 7. Unsharp masking
    print("\n7. Unsharp Masking...")
    unsharp_results = unsharp_masking(clean)
    print(f"   ✓ Tested {len(unsharp_results)} variations")
    
    # 8. Anisotropic diffusion
    print("\n8. Anisotropic Diffusion...")
    aniso_results = anisotropic_diffusion(gaussian_noisy)
    print(f"   ✓ Tested {len(aniso_results)} variations")
    
    # Visualize Gaussian filters
    print("\n9. Visualizing Results...")
    
    fig = plt.figure(figsize=(15, 10))
    
    # Gaussian filtering results
    for idx, (name, img) in enumerate(gaussian_results.items(), 1):
        plt.subplot(2, 3, idx)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('gaussian_filtering_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Bilateral filtering results
    fig = plt.figure(figsize=(15, 5))
    
    for idx, (name, img) in enumerate(bilateral_results.items(), 1):
        plt.subplot(1, 3, idx)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('bilateral_filtering_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Median filtering for salt & pepper
    fig = plt.figure(figsize=(16, 4))
    
    plt.subplot(141)
    plt.imshow(cv2.cvtColor(sp_noisy, cv2.COLOR_BGR2RGB))
    plt.title('Salt & Pepper Noise')
    plt.axis('off')
    
    for idx, (name, img) in enumerate(median_results.items(), 2):
        plt.subplot(1, 4, idx)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('median_filtering_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Morphological operations
    fig = plt.figure(figsize=(18, 6))
    
    for idx, (name, img) in enumerate(morph_results.items(), 1):
        plt.subplot(2, 3, idx)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('morphological_filtering_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Comparison of advanced techniques
    fig = plt.figure(figsize=(16, 8))
    
    all_advanced = {}
    all_advanced.update(nlm_results)
    all_advanced.update(freq_results)
    all_advanced.update(unsharp_results)
    all_advanced.update(aniso_results)
    
    for idx, (name, img) in enumerate(all_advanced.items(), 1):
        plt.subplot(2, 4, idx)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('advanced_filtering_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("   ✓ All visualizations complete")
    
    # Summary
    print("\n" + "=" * 70)
    print("Filter Summary and Recommendations")
    print("=" * 70)
    print("\nGaussian Blur:")
    print("  • Best for: General smoothing, Gaussian noise")
    print("  • Speed: Very fast")
    print("  • Edge preservation: No")
    
    print("\nBilateral Filter:")
    print("  • Best for: Noise reduction with edge preservation")
    print("  • Speed: Medium")
    print("  • Edge preservation: Yes")
    
    print("\nMedian Filter:")
    print("  • Best for: Salt & pepper noise removal")
    print("  • Speed: Fast")
    print("  • Edge preservation: Good")
    
    print("\nNon-Local Means:")
    print("  • Best for: Texture preservation, high-quality denoising")
    print("  • Speed: Slow")
    print("  • Edge preservation: Excellent")
    
    print("\nAnisotropic Diffusion:")
    print("  • Best for: Edge-preserving smoothing")
    print("  • Speed: Medium")
    print("  • Edge preservation: Excellent")
    
    print("\nUnsharp Masking:")
    print("  • Best for: Sharpening, detail enhancement")
    print("  • Speed: Fast")
    print("  • Edge preservation: N/A (enhancement)")
    
    print("\n" + "=" * 70)
    print("All filtering demonstrations completed successfully!")
    print("Results saved as PNG files.")
    print("=" * 70)

if __name__ == "__main__":
    main()
