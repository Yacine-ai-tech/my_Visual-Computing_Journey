# Edge Detection Project

Comparing different edge detection algorithms in OpenCV.

## What This Does

This script compares three popular edge detection methods:
1. **Canny Edge Detection** - Multi-stage algorithm that produces clean edges
2. **Sobel Operator** - Gradient-based method that finds edges in x and y directions
3. **Laplacian** - Second derivative method, very sensitive to edges

## Why I Made This

After working with contours, I wanted to understand the different ways to detect edges. Each method has pros and cons, and I wanted to see them side-by-side.

## Usage

```bash
python edge_detector.py
```

Make sure the cameraman image is in the morphological_operations folder, or update the path.

## What I Learned

### Canny Edge Detection
- Gives the cleanest results
- Multiple stages: noise reduction, gradient calculation, non-maximum suppression, hysteresis
- Two thresholds: lower and upper
- Best for most general purposes

**Parameters I tried:**
- (30, 100) - Too many edges, noisy
- (100, 200) - Too few edges, missed some details
- (50, 150) - Goldilocks zone ✓

### Sobel Operator
- Calculates gradient in x and y directions separately
- Can see directional information
- Good for when you need gradient magnitude and direction
- More sensitive to noise than Canny

### Laplacian
- Uses second derivative to find edges
- Very sensitive - picks up noise easily
- Might need stronger preprocessing (more blur)
- Good when you need to detect rapid intensity changes

## Requirements

```bash
opencv-python
numpy
matplotlib
```

## Next Steps

Want to try:
- [ ] Scharr operator (better than Sobel for small kernels?)
- [ ] Combining edge detection with morphological operations
- [ ] Edge detection on color images (process each channel?)
- [ ] Real-time edge detection with webcam

## Notes

- Always preprocess! GaussianBlur before edge detection reduces noise
- Canny is usually the go-to for most applications
- Sobel is good when you need gradient information
- Laplacian needs careful tuning due to noise sensitivity
- Different images might need different threshold values

---

*Created: April 2024*
