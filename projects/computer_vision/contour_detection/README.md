# Contour Detection

Detects and visualizes contours (boundaries) around shapes in images.

## Description

Implements contour detection using adaptive thresholding and Gaussian blur preprocessing. Handles varying lighting conditions and displays results with highlighted contours.

## Usage

```bash
pip install -r requirements.txt
python contour_detection.py
```

## Pipeline

1. Load image and convert to RGB
2. Convert to grayscale
3. Apply Gaussian blur (5x5 kernel) to reduce noise
4. Apply adaptive threshold to create binary image
5. Detect contours using `cv2.findContours()`
6. Draw contours on original image

## Key Techniques

### Preprocessing
- Gaussian blur reduces noise before thresholding
- Adaptive thresholding handles varying lighting conditions better than global thresholding
- Binary image required for contour detection

### Contour Detection Parameters
- `RETR_TREE` - Retrieves all contours with hierarchy information
- `CHAIN_APPROX_SIMPLE` - Compresses horizontal, vertical, and diagonal segments

## Applications

- Object detection and counting
- Shape analysis and recognition
- Document scanning
- Boundary extraction
- Motion detection

## Requirements

See `requirements.txt`:
- opencv-python
- matplotlib
- numpy

