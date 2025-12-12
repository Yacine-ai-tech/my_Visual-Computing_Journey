# Edge Detection

Comparison of edge detection algorithms in OpenCV.

## Description

This script compares three edge detection methods:
- **Canny Edge Detection** - Multi-stage algorithm with noise reduction and hysteresis thresholding
- **Sobel Operator** - Gradient-based method computing edges in x and y directions
- **Laplacian** - Second derivative method for detecting rapid intensity changes

## Usage

```bash
python edge_detector.py
```

Update the image path in the script as needed.

## Methods

### Canny Edge Detection
Multi-stage process including noise reduction, gradient calculation, non-maximum suppression, and hysteresis. Produces clean edge results suitable for most applications.

### Sobel Operator
Calculates image gradients in x and y directions separately. Provides both gradient magnitude and direction information. More sensitive to noise than Canny.

### Laplacian
Uses second derivatives to find edges. Highly sensitive to noise and requires careful preprocessing with Gaussian blur.

## Requirements

```bash
opencv-python
numpy
matplotlib
```
