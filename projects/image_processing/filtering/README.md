# Advanced Image Filtering

Comprehensive collection of image filtering techniques for noise removal and enhancement.

## Overview

This module implements multiple filtering approaches:
- **Spatial Filters**: Gaussian, Bilateral, Median
- **Morphological Filters**: Opening, Closing, Gradient
- **Advanced Denoising**: Non-local Means, Anisotropic Diffusion
- **Frequency Domain**: Low-pass, High-pass, Wiener-like
- **Enhancement**: Unsharp Masking, Guided Filtering

## Filters Implemented

### 1. Gaussian Filtering
- Simple convolution with Gaussian kernel
- Effective for Gaussian noise
- Multiple kernel sizes: 3x3, 5x5, 9x9

### 2. Bilateral Filtering
- Edge-preserving smoothing
- Considers both spatial and intensity similarity
- Parameters: diameter, sigmaColor, sigmaSpace

### 3. Median Filtering
- Excellent for salt & pepper noise
- Non-linear filter
- Kernel sizes: 3x3, 5x5, 7x7

### 4. Morphological Filtering
- **Opening**: Erosion followed by dilation (removes noise)
- **Closing**: Dilation followed by erosion (fills holes)
- **Gradient**: Difference between dilation and erosion (edges)
- **Top Hat**: Original minus opening (bright features)
- **Black Hat**: Closing minus original (dark features)

### 5. Non-Local Means
- Compares patches instead of single pixels
- Excellent texture preservation
- Fast and strong variants

### 6. Anisotropic Diffusion
- Perona-Malik diffusion
- Edge-preserving smoothing
- Iterative refinement

### 7. Frequency Domain
- FFT-based filtering
- Low-pass for smoothing
- High-pass for edge enhancement

### 8. Unsharp Masking
- Sharpening technique
- Enhances edges and details
- Adjustable strength

## Usage

```bash
python advanced_filters.py
```

The script will:
1. Create test images with various noise types
2. Apply all filtering techniques
3. Compare results side-by-side
4. Save comprehensive visualizations

## Noise Types Demonstrated

### Gaussian Noise
- Random noise from normal distribution
- Common in image acquisition
- Standard deviation: 25

### Salt & Pepper Noise
- Random white and black pixels
- Impulse noise from sensor errors
- Density: 2%

## Algorithm Comparison

| Filter | Speed | Edge Preserving | Best For |
|--------|-------|-----------------|----------|
| Gaussian | Very Fast | No | General smoothing |
| Bilateral | Medium | Yes | Edge-preserving smoothing |
| Median | Fast | Good | Salt & pepper noise |
| NLM | Slow | Excellent | Texture preservation |
| Anisotropic | Medium | Excellent | Edge preservation |
| Morphological | Fast | Varies | Structural filtering |
| Frequency | Fast | No | Frequency analysis |
| Unsharp | Fast | N/A | Sharpening |

## Parameters Guide

### Gaussian Blur
```python
cv2.GaussianBlur(img, ksize, sigmaX)
# ksize: (width, height) - must be odd
# sigmaX: standard deviation in X direction
```

### Bilateral Filter
```python
cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
# d: diameter of pixel neighborhood
# sigmaColor: filter in color space
# sigmaSpace: filter in coordinate space
```

### Median Filter
```python
cv2.medianBlur(img, ksize)
# ksize: aperture size (must be odd)
```

### Non-Local Means
```python
cv2.fastNlMeansDenoisingColored(img, None, h, hColor, templateWindowSize, searchWindowSize)
# h: filter strength for luminance
# hColor: filter strength for color
# templateWindowSize: patch size (odd)
# searchWindowSize: search area size (odd)
```

## Applications

### Noise Reduction
- Medical imaging cleanup
- Photo restoration
- Video denoising
- Low-light image enhancement

### Image Enhancement
- Detail enhancement
- Sharpening
- Contrast improvement
- Edge enhancement

### Pre-processing
- Before segmentation
- Before feature detection
- Before object recognition
- Before compression

## Output Files

- `noise_types.png`: Original images with different noise
- `gaussian_filtering_results.png`: Gaussian blur variations
- `bilateral_filtering_results.png`: Bilateral filter results
- `median_filtering_results.png`: Median filter on salt & pepper
- `morphological_filtering_results.png`: All morphological operations
- `advanced_filtering_results.png`: Advanced techniques comparison

## Tips for Best Results

### For Gaussian Noise
1. Try bilateral filter first (edge-preserving)
2. Use non-local means for best quality
3. Gaussian blur for speed

### For Salt & Pepper Noise
1. Median filter is the best choice
2. Use 5x5 or 7x7 kernel
3. Avoid Gaussian (will blur noise)

### For Sharpening
1. Use unsharp masking
2. Adjust blend weight carefully
3. May amplify noise - denoise first

### For Edge Preservation
1. Bilateral filter: good balance
2. Non-local means: best quality
3. Anisotropic diffusion: iterative approach

## Common Issues

**Over-smoothing**: Reduce kernel size or filter strength
**Under-smoothing**: Increase kernel size or iterations
**Edge blurring**: Use edge-preserving filters
**Slow performance**: Use faster filters or reduce image size
**Artifacts**: Adjust parameters or try different filter

## Extensions

Can be extended with:
- Adaptive filtering based on local variance
- Cascade filters for better results
- Real-time video filtering
- GPU-accelerated filtering
- Machine learning denoising
