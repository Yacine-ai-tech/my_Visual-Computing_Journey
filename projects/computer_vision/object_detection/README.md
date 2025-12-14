# Object Detection and Tracking

Comprehensive object detection and tracking system using multiple computer vision techniques.

## Overview

This module implements various object detection methods including color-based detection, motion-based detection with background subtraction, template matching, and multi-object tracking with unique ID assignment.

## Features

### Detection Methods

#### Color-Based Detection
- HSV color space segmentation
- Multiple color detection (red, green, blue, yellow, etc.)
- Morphological operations for noise reduction
- Contour-based object localization

#### Motion-Based Detection
- Background subtraction (MOG2 algorithm)
- Adaptive background modeling
- Shadow detection and removal
- Foreground segmentation

#### Template Matching
- Multiple matching methods (TM_CCOEFF, TM_SQDIFF, etc.)
- Multi-scale template matching
- Rotation-invariant matching options
- Confidence scoring

### Tracking Features

#### Multi-Object Tracking
- Unique ID assignment for each object
- Centroid-based tracking algorithm
- Track history and trajectory visualization
- Lost track handling

#### Object Tracking Algorithms
- Centroid tracking (simple, fast)
- Kalman filter prediction (smooth tracking)
- Track association with Hungarian algorithm
- Confidence-based track management

### Visualization

- Bounding boxes with object IDs
- Trajectory trails
- Detection confidence scores
- Color-coded object classes
- Real-time statistics display

## Usage

```bash
python object_detector.py
```

The script will:
1. Create synthetic video with moving objects
2. Apply color-based detection
3. Apply motion-based detection
4. Track objects with unique IDs
5. Visualize trajectories and statistics
6. Save detection and tracking results

## Algorithm Details

### Color-Based Detection

```python
# Convert to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Define color range
lower_bound = np.array([H_min, S_min, V_min])
upper_bound = np.array([H_max, S_max, V_max])

# Create mask
mask = cv2.inRange(hsv, lower_bound, upper_bound)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                cv2.CHAIN_APPROX_SIMPLE)
```

### Background Subtraction (MOG2)

```python
# Create background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=True
)

# Apply to frame
fg_mask = bg_subtractor.apply(frame)

# Remove shadows (gray pixels)
_, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
```

### Centroid Tracking

1. Detect objects in current frame
2. Compute centroids of bounding boxes
3. Match with previous frame centroids (minimum distance)
4. Assign existing IDs or create new ones
5. Remove tracks lost for N consecutive frames

### Template Matching

```python
# Match template in image
result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

# Find best match location
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Threshold for multiple detections
locations = np.where(result >= threshold)
```

## Color Ranges for HSV

Common color ranges in HSV space:

| Color | H Range | S Range | V Range |
|-------|---------|---------|---------|
| Red | 0-10, 170-180 | 120-255 | 70-255 |
| Green | 40-80 | 40-255 | 40-255 |
| Blue | 100-130 | 50-255 | 50-255 |
| Yellow | 20-30 | 100-255 | 100-255 |
| Orange | 10-20 | 100-255 | 100-255 |
| Purple | 130-160 | 50-255 | 50-255 |
| White | 0-180 | 0-30 | 200-255 |
| Black | 0-180 | 0-255 | 0-30 |

Note: H in OpenCV is in range [0, 180], not [0, 360]

## Applications

### Surveillance
- Intruder detection
- Crowd monitoring
- Abandoned object detection
- Perimeter security

### Industrial Automation
- Product counting on conveyor belts
- Quality inspection
- Assembly line monitoring
- Defect detection

### Traffic Monitoring
- Vehicle counting and classification
- Speed estimation
- License plate detection
- Parking space detection

### Retail Analytics
- Customer tracking
- Queue management
- Heatmap generation
- Dwell time analysis

### Sports Analytics
- Player tracking
- Ball tracking
- Formation analysis
- Performance metrics

## Output Files

- `color_detection_results.png`: Color-based detection visualization
- `motion_detection_results.png`: Background subtraction results
- `template_matching_results.png`: Template matching visualization
- `object_tracking_results.png`: Multi-object tracking with IDs
- `tracking_statistics.txt`: Detection and tracking statistics

## Parameters Guide

### Color Detection
```python
# Morphological kernel for noise removal
kernel_size = (5, 5)

# Minimum contour area to filter small detections
min_area = 500

# Color range tuning
# Wider range: More detections, more false positives
# Narrower range: Fewer false positives, may miss objects
```

### Background Subtraction
```python
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,        # Number of frames for background model
    varThreshold=16,    # Threshold for pixel-model match (lower = more sensitive)
    detectShadows=True  # Detect and mark shadows
)
```

### Object Tracking
```python
max_disappeared = 30    # Frames before removing lost track
max_distance = 50       # Maximum distance for track association
min_confidence = 0.5    # Minimum detection confidence
```

## Tips for Best Results

### Color-Based Detection

**Lighting Conditions**
- Use HSV space (more robust than RGB)
- Calibrate color ranges for your lighting
- Consider automatic white balance
- Use histogram equalization for varying light

**Noise Reduction**
- Apply morphological operations (opening, closing)
- Use median blur before color detection
- Adjust minimum contour area threshold
- Filter by aspect ratio for specific shapes

### Motion Detection

**Background Model**
- Allow warmup period for background learning
- Adjust `history` based on scene dynamics
- Higher `varThreshold` for noisy environments
- Use shadow detection to avoid false positives

**Foreground Mask Cleaning**
- Apply morphological operations
- Use connected component analysis
- Filter by size and shape
- Consider temporal consistency

### Tracking

**Track Association**
- Use Euclidean distance for simple cases
- Use Mahalanobis distance for Kalman filter
- Consider appearance features for re-identification
- Handle occlusions with prediction

**Performance**
- Process at appropriate frame rate
- Use ROI to limit search area
- Downsample high-resolution video
- Consider GPU acceleration for real-time

## Detection Method Comparison

| Method | Speed | Accuracy | Lighting Sensitivity | Use Case |
|--------|-------|----------|---------------------|----------|
| Color | Fast | Medium | High | Colored objects, controlled lighting |
| Motion | Fast | Good | Low | Moving objects, static camera |
| Template | Medium | Good | Medium | Known object appearance |
| Cascade | Very Fast | Good | Low | Faces, pedestrians |
| DNN | Slow | Excellent | Low | General objects, complex scenes |

## Common Issues

**Problem**: False positives in color detection
**Solution**: Narrow HSV range, increase `minNeighbors`, use shape filtering

**Problem**: Objects not detected in motion detection
**Solution**: Lower `varThreshold`, ensure camera is static, check min area

**Problem**: Track ID switching between objects
**Solution**: Reduce `max_distance`, use appearance features, improve detection

**Problem**: Lost tracks during occlusion
**Solution**: Use Kalman filter prediction, increase `max_disappeared`

**Problem**: Slow performance
**Solution**: Reduce resolution, limit ROI, optimize detection frequency

## Advanced Techniques

### Kalman Filter for Prediction
```python
# Predict object position during occlusion
kalman = cv2.KalmanFilter(4, 2)  # 4 state params, 2 measurement params
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
```

### Non-Maximum Suppression
```python
# Remove overlapping detections
def non_max_suppression(boxes, overlap_thresh=0.3):
    # Implementation of NMS
    # Keep detection with highest confidence
    # Remove detections with high IOU overlap
    pass
```

### Multi-Scale Detection
```python
# Detect objects at different scales
for scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
    resized = cv2.resize(image, None, fx=scale, fy=scale)
    detections = detect_objects(resized)
    # Scale back coordinates
```

## Modern Alternatives

For production systems, consider:

**Deep Learning Based**
- **YOLO** (v5, v8): Real-time, excellent accuracy
- **SSD**: Fast, good for embedded systems
- **Faster R-CNN**: High accuracy, slower
- **RetinaNet**: Balanced speed/accuracy

**Tracking Algorithms**
- **SORT**: Simple Online and Realtime Tracking
- **DeepSORT**: SORT with appearance features
- **ByteTrack**: State-of-art multi-object tracking
- **FairMOT**: One-shot detection and tracking

## Extensions

The code can be extended with:
- Deep learning object detection (YOLO, SSD)
- Advanced tracking (DeepSORT, ByteTrack)
- Object re-identification
- Action recognition
- Anomaly detection
- 3D object tracking
- Sensor fusion (camera + LiDAR)

## Requirements

- opencv-python >= 4.5.0
- numpy >= 1.19.0
- matplotlib >= 3.3.0
- scipy >= 1.5.0 (for Hungarian algorithm)

## References

- Bradski, G., & Kaehler, A. (2008). Learning OpenCV
- Yilmaz, A., Javed, O., & Shah, M. (2006). Object tracking: A survey
- OpenCV Documentation: Object Detection and Tracking
- MOG2: Z. Zivkovic (2004). Improved adaptive Gaussian mixture model
