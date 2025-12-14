# Geometric Transformations

Comprehensive implementation of 2D geometric transformations for image processing.

## Overview

This module implements a complete suite of geometric transformations including basic transformations (translation, rotation, scaling), advanced transformations (affine, perspective), and specialized transforms (polar coordinates).

## Features

### Basic Transformations

#### Translation
- Shift images by specified pixel offsets
- Independent control of horizontal and vertical translation
- Useful for image alignment and registration

#### Rotation
- Rotate images around arbitrary center points
- Specify rotation angle in degrees
- Optional scaling during rotation
- Proper handling of image borders

#### Scaling
- Resize images by scale factors
- Independent x and y scaling
- Multiple interpolation methods supported

### Advanced Transformations

#### Shearing
- Apply shear transformation in x or y direction
- Useful for correcting perspective distortions
- Automatic dimension adjustment

#### Affine Transformation
- General linear transformation
- Maps 3 source points to 3 destination points
- Preserves parallel lines
- Combines rotation, scaling, translation, and shearing

#### Perspective Transformation
- Full perspective warping using homography
- Maps 4 source points to 4 destination points
- Essential for document scanning and rectification
- Useful for AR applications

### Specialized Transforms

#### Polar Transformations
- **Linear Polar**: Convert Cartesian to polar coordinates
- **Logarithmic Polar**: Scale-invariant transformation
- Useful for rotation-invariant feature detection

### Composite Transformations
- Combine multiple transformations
- Apply sequences of operations efficiently
- Chain transformations with single operation

## Usage

```bash
python geometric_transforms.py
```

The script will:
1. Create a test image with grid and reference shapes
2. Apply all transformation types
3. Display results with before/after comparison
4. Save comprehensive visualizations

## Transformation Details

### Translation
```python
# Shift image by (tx, ty) pixels
M = [[1, 0, tx],
     [0, 1, ty]]
result = cv2.warpAffine(img, M, (width, height))
```

### Rotation
```python
# Rotate around center by angle degrees
center = (cx, cy)
M = cv2.getRotationMatrix2D(center, angle, scale)
result = cv2.warpAffine(img, M, (width, height))
```

### Affine Transform
```python
# Map 3 points to new positions
src_points = np.float32([[x1,y1], [x2,y2], [x3,y3]])
dst_points = np.float32([[x1',y1'], [x2',y2'], [x3',y3']])
M = cv2.getAffineTransform(src_points, dst_points)
result = cv2.warpAffine(img, M, (width, height))
```

### Perspective Transform
```python
# Map 4 corner points to new positions
src_points = np.float32([[x1,y1], [x2,y2], [x3,y3], [x4,y4]])
dst_points = np.float32([[x1',y1'], [x2',y2'], [x3',y3'], [x4',y4']])
M = cv2.getPerspectiveTransform(src_points, dst_points)
result = cv2.warpPerspective(img, M, (width, height))
```

## Interpolation Methods

The module supports multiple interpolation methods:
- **Nearest Neighbor** (`cv2.INTER_NEAREST`): Fastest, blocky results
- **Linear** (`cv2.INTER_LINEAR`): Good balance of speed and quality
- **Cubic** (`cv2.INTER_CUBIC`): Higher quality, slower
- **Lanczos** (`cv2.INTER_LANCZOS4`): Best quality, slowest

## Applications

### Document Processing
- Perspective correction for scanned documents
- Document alignment and registration
- Removing skew from images

### Computer Vision
- Image registration and alignment
- Camera calibration
- Stereo vision rectification
- Homography estimation

### Augmented Reality
- Overlay virtual objects on real scenes
- Marker tracking and alignment
- Perspective matching

### Medical Imaging
- Image registration between modalities
- Alignment of temporal sequences
- Atlas-based segmentation

### Photography
- Lens distortion correction
- Panorama stitching preparation
- Image stabilization

## Output Files

- `geometric_transforms_basic.png`: Translation, rotation, scaling results
- `geometric_transforms_advanced.png`: Affine and perspective results
- `geometric_transforms_polar.png`: Polar coordinate transformations
- `geometric_transforms_composite.png`: Combined transformations

## Parameters Guide

### Translation
- `tx`: Horizontal shift in pixels (positive = right)
- `ty`: Vertical shift in pixels (positive = down)

### Rotation
- `angle`: Rotation angle in degrees (positive = counter-clockwise)
- `center`: Center of rotation (cx, cy)
- `scale`: Scaling factor during rotation (default: 1.0)

### Scaling
- `sx`: Horizontal scale factor
- `sy`: Vertical scale factor
- Values > 1 enlarge, values < 1 shrink

### Shearing
- `shx`: Horizontal shear factor
- `shy`: Vertical shear factor
- Typical values: -0.5 to 0.5

### Polar Transform
- `center`: Center point for polar conversion
- `max_radius`: Maximum radius in polar space
- `flags`: Interpolation method

## Tips for Best Results

### Document Rectification
1. Detect document corners using edge detection
2. Order corners consistently (top-left, top-right, bottom-right, bottom-left)
3. Apply perspective transform to rectangle
4. Use appropriate output dimensions

### Image Alignment
1. Detect corresponding feature points
2. Compute transformation matrix
3. Verify with RANSAC for robustness
4. Apply transformation with proper interpolation

### Avoiding Artifacts
- Use appropriate interpolation method for task
- Consider border mode (constant, replicate, reflect)
- Apply anti-aliasing when downscaling
- Maintain aspect ratio when needed

## Common Issues

**Problem**: Clipped content after transformation
**Solution**: Calculate proper output dimensions or increase canvas size

**Problem**: Black borders around transformed image
**Solution**: Use `borderMode` parameter or set `borderValue` to appropriate color

**Problem**: Jagged edges after rotation
**Solution**: Use higher quality interpolation (cubic or Lanczos)

**Problem**: Distorted aspect ratio
**Solution**: Maintain aspect ratio by using same scale factor for x and y

## Mathematical Background

### Affine Transformation Matrix
```
[x']   [a  b  tx]   [x]
[y'] = [c  d  ty] × [y]
[1 ]   [0  0  1 ]   [1]
```

Where:
- (a, b, c, d) define rotation, scaling, and shearing
- (tx, ty) define translation

### Perspective Transformation (Homography)
```
[x']   [h11 h12 h13]   [x]
[y'] = [h21 h22 h23] × [y]
[w']   [h31 h32 h33]   [1]

Final coordinates: (x'/w', y'/w')
```

## Extensions

The code can be extended with:
- Elastic deformations
- Thin plate spline warping
- Mesh-based transformations
- Camera distortion models
- 3D transformations
- GPU-accelerated transforms

## Requirements

- opencv-python >= 4.5.0
- numpy >= 1.19.0
- matplotlib >= 3.3.0

## References

- Hartley, R., & Zisserman, A. (2003). Multiple View Geometry in Computer Vision
- Szeliski, R. (2010). Computer Vision: Algorithms and Applications
- OpenCV Documentation: Geometric Image Transformations
