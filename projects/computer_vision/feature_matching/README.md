# Feature Detection and Matching

Advanced feature detection algorithms and matching techniques for computer vision applications.

## Overview

This module implements and compares four state-of-the-art feature detection algorithms:
- **SIFT** (Scale-Invariant Feature Transform)
- **ORB** (Oriented FAST and Rotated BRIEF)
- **AKAZE** (Accelerated-KAZE)
- **BRISK** (Binary Robust Invariant Scalable Keypoints)

Plus classical corner detection methods:
- **Harris Corner Detection**
- **Shi-Tomasi Corner Detection**

## Features

### Feature Detectors
- Multiple algorithm implementations
- Keypoint visualization with scale and orientation
- Descriptor computation for matching
- Performance comparison

### Feature Matching
- Brute-force matching with cross-check
- Lowe's ratio test for robust matching
- Geometric verification using RANSAC
- Homography estimation

### Corner Detection
- Harris corner detector
- Shi-Tomasi (Good Features to Track)
- Parameter tuning examples

## Usage

```bash
python feature_detector.py
```

The script will:
1. Create test images with geometric transformations
2. Detect features using all algorithms
3. Match features between images
4. Compute homography matrices
5. Display and save comprehensive results

## Algorithm Comparison

| Algorithm | Type | Speed | Robustness | Patent |
|-----------|------|-------|------------|--------|
| SIFT | Float | Medium | Excellent | Yes |
| ORB | Binary | Fast | Good | Free |
| AKAZE | Float/Binary | Fast | Very Good | Free |
| BRISK | Binary | Very Fast | Good | Free |

## Key Concepts

### Keypoints
- Location (x, y coordinates)
- Scale (size of detected feature)
- Orientation (rotation angle)
- Response (strength of feature)

### Descriptors
- Feature vectors describing local appearance
- Used for matching between images
- Float descriptors (SIFT, AKAZE) use L2 norm
- Binary descriptors (ORB, BRISK) use Hamming distance

### Matching Strategies
1. **Brute-Force**: Compare all descriptor pairs
2. **Ratio Test**: Filter ambiguous matches
3. **Cross-Check**: Bidirectional consistency
4. **RANSAC**: Geometric verification

## Applications

- Image stitching and panoramas
- Object recognition and tracking
- 3D reconstruction
- Visual odometry
- Augmented reality
- Image registration
- Place recognition

## Output Files

- `feature_detection_comparison.png`: Keypoints from all detectors
- `feature_matching_comparison.png`: Matched features visualization
- `corner_detection.png`: Harris and Shi-Tomasi corners

## Parameters

### SIFT
- `nfeatures`: Maximum number of features (default: 500)
- `nOctaveLayers`: Layers per octave (default: 3)
- `contrastThreshold`: Filter weak features (default: 0.04)

### ORB
- `nfeatures`: Maximum number of features (default: 500)
- `scaleFactor`: Pyramid decimation ratio (default: 1.2)
- `nlevels`: Number of pyramid levels (default: 8)

### AKAZE
- `threshold`: Detector response threshold (default: 0.001)
- `nOctaves`: Number of octaves (default: 4)

### BRISK
- `threshold`: Detection threshold (default: 30)
- `octaves`: Number of octaves (default: 3)

## Performance Tips

- Use ORB for real-time applications
- Use SIFT for maximum accuracy
- Use AKAZE for balanced speed/accuracy
- Use BRISK for resource-constrained systems
