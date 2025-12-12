# Visual Computing Projects

A comprehensive collection of computer vision, image processing, and visual computing projects using OpenCV and Python. This repository covers all major areas of visual computing with functional, well-documented implementations.

## Overview

This repository contains extensive implementations of visual computing techniques, from basic image processing to advanced computer vision algorithms. Each project includes detailed documentation, working code, and visualizations.

## Projects

### 📸 Image Processing

#### **Segmentation**
Advanced image segmentation techniques
- Watershed segmentation with marker-based approach
- K-means clustering for color segmentation (k=2,3,5)
- GrabCut for foreground extraction
- Comprehensive visualization and analysis

#### **Filtering**
Comprehensive filtering and denoising methods
- Gaussian, Bilateral, Median filters
- Morphological operations (opening, closing, gradient, top-hat, black-hat)
- Non-local means denoising
- Anisotropic diffusion (Perona-Malik)
- Frequency domain filtering (FFT-based)
- Unsharp masking for sharpening
- Handles Gaussian and salt & pepper noise

#### **Edge Detection**
Multiple edge detection algorithms
- Canny edge detection
- Sobel operator (X and Y gradients)
- Laplacian edge detection
- Comparative analysis and visualization

#### **Morphological Operations**
Basic morphological transformations
- Erosion and dilation
- Opening and closing
- Morphological gradient
- Multiple iterations and kernel sizes

#### **Geometric Transformations**
Complete 2D transformation suite
- Translation, rotation, scaling
- Shearing and affine transforms
- Perspective transformation and homography
- Polar transformations (linear and logarithmic)
- Composite transformations
- Multiple interpolation methods (nearest, linear, cubic, Lanczos)
- Perspective rectification

#### **Histogram Processing**
Advanced histogram techniques
- Histogram equalization (global and color-aware)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Gamma correction
- Contrast stretching
- Histogram matching and specification
- 2D color histograms
- Color distribution analysis

### 🎯 Computer Vision

#### **Feature Matching**
State-of-the-art feature detection and matching
- SIFT (Scale-Invariant Feature Transform)
- ORB (Oriented FAST and Rotated BRIEF)
- AKAZE (Accelerated-KAZE)
- BRISK (Binary Robust Invariant Scalable Keypoints)
- Harris and Shi-Tomasi corner detection
- Feature matching with ratio test
- Homography estimation with RANSAC
- Comprehensive performance comparison

#### **Optical Flow**
Motion estimation and visualization
- Lucas-Kanade sparse optical flow
- Farneback dense optical flow
- Multiple visualization methods (HSV, arrows, trajectories)
- Flow statistics and analysis
- Synthetic video generation for testing

#### **Object Detection**
Multiple detection approaches
- Color-based detection in HSV space
- Background subtraction (MOG2)
- Template matching (multiple methods)
- Centroid-based tracking
- Bounding box visualization
- Multi-object tracking with IDs

#### **Face Recognition**
Face detection and analysis
- Haar Cascade classifiers
- Face, eye, and smile detection
- Multiple cascade configurations
- Parameter tuning demonstration
- Real-time capable implementations

#### **Contour Detection**
Contour analysis and boundary detection
- Adaptive thresholding
- Gaussian blur pre-processing
- Contour hierarchy analysis
- Bounding box extraction

### 🎨 Interactive Tools

#### **Mouse Drawing**
Interactive drawing with mouse events
- Circle drawing on click
- Real-time canvas updates
- Event handling demonstration

### 🧪 Experiments
Various experimental implementations and tests for rapid prototyping

## Getting Started

### Prerequisites
```bash
Python 3.8+
```

### Installation
```bash
git clone https://github.com/Yacine-ai-tech/my_Visual-Computing_Journey.git
cd my_Visual-Computing_Journey
pip install -r requirements.txt
```

### Quick Start
Navigate to any project directory and run the Python script:

```bash
# Image Segmentation
cd projects/image_processing/segmentation
python watershed_segmentation.py

# Feature Detection
cd projects/computer_vision/feature_matching
python feature_detector.py

# Optical Flow
cd projects/computer_vision/optical_flow
python optical_flow_demo.py

# Advanced Filtering
cd projects/image_processing/filtering
python advanced_filters.py

# Object Detection
cd projects/computer_vision/object_detection
python object_detector.py

# Face Detection
cd projects/computer_vision/face_recognition
python face_detector.py

# Histogram Processing
cd projects/image_processing/histogram_processing
python histogram_demo.py

# Geometric Transforms
cd projects/image_processing/geometric_transforms
python geometric_transforms.py
```

## Project Structure
```
projects/
├── image_processing/
│   ├── segmentation/              # Watershed, K-means, GrabCut
│   ├── filtering/                 # Advanced filters and denoising
│   ├── edge_detection/            # Canny, Sobel, Laplacian
│   ├── morphological_operations/  # Erosion, dilation, etc.
│   ├── geometric_transforms/      # Affine, perspective, polar
│   └── histogram_processing/      # Equalization, CLAHE, gamma
├── computer_vision/
│   ├── feature_matching/          # SIFT, ORB, AKAZE, BRISK
│   ├── optical_flow/              # Lucas-Kanade, Farneback
│   ├── object_detection/          # Color, background subtraction
│   ├── face_recognition/          # Haar cascades, detection
│   └── contour_detection/         # Boundary analysis
├── interactive/
│   └── mouse_draw_circle/         # Mouse event handling
└── experiments/                    # Experimental code
```

## Features

### Comprehensive Coverage
- ✅ Image preprocessing and enhancement
- ✅ Feature detection and matching
- ✅ Object detection and tracking
- ✅ Motion estimation (optical flow)
- ✅ Image segmentation
- ✅ Geometric transformations
- ✅ Histogram analysis and equalization
- ✅ Face detection
- ✅ Advanced filtering techniques
- ✅ Morphological operations

### Educational Value
- 📚 Detailed inline comments explaining algorithms
- 📊 Comprehensive visualizations
- 🎯 Multiple methods compared side-by-side
- 📈 Performance analysis and trade-offs
- 🔧 Parameter tuning demonstrations
- 💡 Real-world application examples

### Production-Ready Code
- ✨ Clean, well-structured implementations
- 🧪 Tested on synthetic data
- 📝 Extensive documentation (README in each project)
- 🎨 Publication-quality visualizations
- ⚡ Optimized for performance where possible

## Requirements
- opencv-python >= 4.5.0
- numpy >= 1.19.0
- matplotlib >= 3.3.0
- scipy >= 1.5.0

See `requirements.txt` for full dependencies.

## Algorithm Reference

### Image Processing
| Algorithm | Use Case | Speed | Quality |
|-----------|----------|-------|---------|
| Gaussian Blur | General smoothing | Very Fast | Good |
| Bilateral Filter | Edge-preserving smoothing | Medium | Excellent |
| Non-Local Means | Texture preservation | Slow | Excellent |
| Watershed | Region segmentation | Medium | Good |
| K-means | Color clustering | Fast | Good |
| CLAHE | Adaptive contrast | Fast | Excellent |

### Computer Vision
| Algorithm | Use Case | Speed | Accuracy |
|-----------|----------|-------|----------|
| SIFT | Feature matching | Medium | Excellent |
| ORB | Real-time features | Very Fast | Good |
| Lucas-Kanade | Sparse tracking | Fast | Good |
| Farneback | Dense flow | Medium | Good |
| Haar Cascade | Face detection | Very Fast | Good |
| Background Subtraction | Motion detection | Fast | Good |

## Applications

### Medical Imaging
- Image enhancement and denoising
- Segmentation of organs and tissues
- Feature detection for diagnosis

### Autonomous Vehicles
- Object detection and tracking
- Optical flow for motion estimation
- Lane detection using edge detection

### Security & Surveillance
- Face detection and recognition
- Motion detection (background subtraction)
- Object tracking

### Photography & Media
- Image enhancement (histogram equalization)
- Panorama creation (feature matching)
- Color correction and grading

### Augmented Reality
- Feature detection and matching
- Perspective transformation
- Object tracking

## Contributing
Contributions are welcome! Each project should:
- Include comprehensive documentation
- Have working example code
- Provide visualizations
- Follow existing code style

## License
MIT License - see LICENSE file for details.

## Acknowledgments
- OpenCV community for excellent libraries
- Classic computer vision papers and textbooks
- Computer vision research community

---

**Note**: All projects are self-contained with synthetic test data, so you can run them immediately without external datasets.
