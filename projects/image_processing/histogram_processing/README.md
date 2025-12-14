# Histogram Processing and Color Analysis

Comprehensive histogram techniques for image enhancement, contrast adjustment, and color analysis.

## Overview

This module implements advanced histogram processing techniques for improving image quality and analyzing color distributions. It covers global and adaptive equalization, gamma correction, contrast stretching, and color histogram analysis.

## Features

### Histogram Equalization

#### Global Histogram Equalization
- Spreads out intensity values across full range
- Enhances overall contrast
- Works well for images with narrow intensity distribution
- Available for grayscale and color images

#### Color Image Equalization Methods
- **YUV Method**: Equalizes luminance channel only
- **HSV Method**: Equalizes value channel, preserves hue and saturation
- **BGR Method**: Equalizes each channel independently (can cause color shifts)

### CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Adaptive local contrast enhancement
- Prevents over-amplification of noise
- Adjustable tile grid size
- Clip limit to control contrast enhancement
- Best for images with varying local illumination

### Gamma Correction
- Non-linear intensity adjustment
- Gamma < 1: Brightens image
- Gamma > 1: Darkens image
- Gamma = 1: No change
- Useful for correcting camera response curves

### Contrast Stretching
- Linear mapping to full intensity range
- Automatic min/max detection
- Improves dynamic range utilization
- Simple and fast enhancement

### Histogram Matching (Specification)
- Match one image's histogram to reference
- Transfer color/tone characteristics
- Useful for image standardization
- Color grading and style transfer

### Color Histogram Analysis
- 2D color histograms (hue-saturation)
- 3D color distribution visualization
- Dominant color extraction
- Color space conversions (RGB, HSV, LAB)

## Usage

```bash
python histogram_demo.py
```

The script will:
1. Create test images with various lighting conditions
2. Apply all histogram techniques
3. Display before/after comparisons
4. Show histogram plots for analysis
5. Save comprehensive visualizations

## Algorithm Details

### Histogram Equalization Process
1. Compute histogram of intensity values
2. Calculate cumulative distribution function (CDF)
3. Normalize CDF to [0, 255] range
4. Map original intensities using CDF as lookup table

### CLAHE Process
1. Divide image into tiles (e.g., 8x8)
2. Compute histogram for each tile
3. Clip histogram to limit contrast amplification
4. Equalize each tile's histogram
5. Use bilinear interpolation at tile boundaries

### Gamma Correction
```python
# Apply gamma correction
corrected = 255 * (img / 255) ** gamma
```

### Contrast Stretching
```python
# Linear stretch to [0, 255]
stretched = 255 * (img - min_val) / (max_val - min_val)
```

## Applications

### Photography Enhancement
- Correct underexposed or overexposed photos
- Enhance details in shadow and highlight regions
- Balance lighting in backlit scenes
- Improve visibility of features

### Medical Imaging
- Enhance X-ray and MRI contrast
- Improve visibility of anatomical structures
- Standardize image appearance across modalities
- Highlight regions of interest

### Document Processing
- Improve readability of scanned documents
- Enhance faded text
- Standardize document appearance
- Prepare for OCR

### Video Processing
- Real-time contrast enhancement
- Consistent appearance across frames
- Low-light video enhancement
- HDR tone mapping

### Computer Vision Preprocessing
- Normalize lighting conditions
- Improve feature detection
- Enhance edge visibility
- Standardize inputs to ML models

## Output Files

- `histogram_comparison.png`: Original vs. equalized histograms
- `clahe_results.png`: CLAHE with different parameters
- `gamma_correction.png`: Various gamma values comparison
- `contrast_stretching.png`: Before/after contrast stretch
- `color_histogram_2d.png`: 2D hue-saturation histogram
- `histogram_matching.png`: Histogram specification results

## Parameters Guide

### Histogram Equalization
```python
# Grayscale
equalized = cv2.equalizeHist(gray_img)

# Color (YUV method)
yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
```

### CLAHE
```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# clipLimit: Threshold for contrast limiting (1.0-4.0 typical)
# tileGridSize: Size of grid for histogram equalization
```

### Gamma Correction
```python
# gamma < 1.0: Brighten
# gamma > 1.0: Darken
corrected = np.power(img / 255.0, gamma) * 255.0
```

### Contrast Stretching
```python
# Automatic
min_val = np.min(img)
max_val = np.max(img)
stretched = 255 * (img - min_val) / (max_val - min_val)

# Percentile-based (robust to outliers)
p2, p98 = np.percentile(img, (2, 98))
stretched = np.clip((img - p2) * 255 / (p98 - p2), 0, 255)
```

## Tips for Best Results

### When to Use Each Method

**Global Histogram Equalization**
- Good for: Images with narrow intensity distribution
- Avoid: Images with already good contrast
- Note: Can over-enhance noise

**CLAHE**
- Good for: Medical images, varying illumination
- Parameters: Start with clipLimit=2.0, tileGridSize=(8,8)
- Note: Best general-purpose method

**Gamma Correction**
- Good for: Systematic brightness adjustment
- Values: Try 0.5 (brighten) or 1.5 (darken) first
- Note: Preserves color relationships

**Contrast Stretching**
- Good for: Quick improvement, preprocessing
- Note: Simple but effective for many cases

### Color Image Enhancement
1. Convert to LAB or YUV color space
2. Apply enhancement to luminance channel only
3. Convert back to BGR
4. This preserves color information better

### Avoiding Common Pitfalls
- Don't equalize images that already have good contrast
- Use CLAHE for images with varying illumination
- Apply gamma correction before equalization
- Consider percentile-based stretching to handle outliers
- Equalize luminance only for color images

## Performance Comparison

| Method | Speed | Quality | Use Case |
|--------|-------|---------|----------|
| Global Equalization | Very Fast | Good | Uniform lighting |
| CLAHE | Fast | Excellent | Varying lighting |
| Gamma Correction | Very Fast | Good | Brightness adjust |
| Contrast Stretching | Very Fast | Fair | Quick preprocessing |
| Histogram Matching | Medium | Good | Style transfer |

## Mathematical Background

### Histogram Equalization
Transform intensity values using:
```
s = T(r) = (L-1) * CDF(r)
```
Where:
- r: input intensity
- s: output intensity
- L: number of intensity levels (256 for 8-bit)
- CDF: cumulative distribution function

### Gamma Correction
```
V_out = V_max * (V_in / V_max) ^ gamma
```

### Probability Distribution
```
p(r_k) = n_k / n
```
Where:
- n_k: number of pixels with intensity r_k
- n: total number of pixels

## Color Spaces

### RGB
- Intuitive but not perceptually uniform
- R, G, B channels correlated
- Not ideal for enhancement

### HSV
- Hue: Color type (0-180°)
- Saturation: Color intensity (0-100%)
- Value: Brightness (0-100%)
- Best for color-based operations

### LAB
- L: Lightness (0-100)
- a: Green-Red axis
- b: Blue-Yellow axis
- Perceptually uniform
- Best for image enhancement

### YUV
- Y: Luminance
- U, V: Chrominance
- Used in video compression
- Good for enhancement

## Common Issues

**Problem**: Color shifts after equalization
**Solution**: Equalize luminance channel only (use YUV or LAB)

**Problem**: Noise amplification
**Solution**: Use CLAHE with lower clip limit or denoise first

**Problem**: Washed out appearance
**Solution**: Reduce clip limit in CLAHE or use milder enhancement

**Problem**: Unnatural colors
**Solution**: Apply enhancement to V channel in HSV or L channel in LAB

## Extensions

The code can be extended with:
- Automatic parameter selection
- Multi-scale histogram processing
- Histogram backprojection for tracking
- Color transfer between images
- Deep learning-based enhancement
- HDR tone mapping
- Retinex algorithm

## Requirements

- opencv-python >= 4.5.0
- numpy >= 1.19.0
- matplotlib >= 3.3.0

## References

- Pizer, S. M., et al. (1987). Adaptive histogram equalization and its variations. Computer Vision, Graphics, and Image Processing, 39(3), 355-368
- Gonzalez, R. C., & Woods, R. E. (2018). Digital Image Processing (4th ed.). Pearson
- Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram Equalization. Graphics Gems IV, Academic Press, 474-485
- OpenCV Documentation: Histograms
