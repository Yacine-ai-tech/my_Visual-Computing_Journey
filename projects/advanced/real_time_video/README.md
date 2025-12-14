# Real-Time Video Detection

High-performance real-time object detection system optimized for video streams and live camera feeds.

## Overview

This module implements a comprehensive real-time detection system with multiple detection methods, object tracking, performance monitoring, and optimization for 30+ FPS processing on video streams.

## Features

### Detection Methods

#### Color-Based Detection
- Real-time HSV color segmentation
- Multiple simultaneous color detection
- Morphological noise filtering
- Adaptive thresholding

#### Motion-Based Detection
- Background subtraction (MOG2 algorithm)
- Adaptive background modeling
- Shadow detection and removal
- Foreground object segmentation

#### Cascade-Based Detection
- Haar Cascade face/body detection
- Multi-scale detection
- Fast processing for real-time use
- Pre-trained classifiers

### Tracking System

#### Multi-Object Tracking
- Unique ID assignment for each object
- Centroid-based association
- Track history and trajectories
- Lost track management

#### Advanced Tracking Features
- Track confidence scoring
- Occlusion handling
- Track lifecycle management
- Trajectory smoothing

### Performance Monitoring

#### Real-Time Metrics
- FPS (Frames Per Second) calculation
- Detection time measurement
- Processing time breakdown
- Frame drop detection

#### Statistics Display
- Live FPS counter
- Active track count
- Detection confidence
- Processing bottlenecks

### Optimization Features

- Frame skipping for performance
- ROI (Region of Interest) processing
- Multi-threading support
- GPU acceleration (when available)
- Adaptive quality settings

## Usage

```bash
cd projects/advanced/real_time_video
python realtime_detection.py
```

The script will:
1. Create synthetic video stream with moving objects
2. Apply selected detection method(s)
3. Track detected objects with unique IDs
4. Display real-time FPS and statistics
5. Visualize trajectories and bounding boxes
6. Save performance metrics

### Command-Line Arguments

```bash
# Color-based detection
python realtime_detection.py --method color

# Motion-based detection
python realtime_detection.py --method motion

# Cascade detection (faces)
python realtime_detection.py --method cascade

# Webcam input
python realtime_detection.py --source webcam

# Video file input
python realtime_detection.py --source video.mp4

# Target FPS
python realtime_detection.py --fps 30
```

## Algorithm Details

### RealTimeDetector Class

```python
class RealTimeDetector:
    def __init__(self, detection_method='color'):
        self.detection_method = detection_method
        self.trackers = []
        self.fps_history = deque(maxlen=30)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
```

### Detection Pipeline

1. **Frame Acquisition**: Capture frame from video source
2. **Preprocessing**: Resize, denoise, color conversion
3. **Detection**: Apply selected detection method
4. **Tracking**: Associate detections with existing tracks
5. **Visualization**: Draw bounding boxes and information
6. **Performance Monitoring**: Calculate and display FPS

### Color Detection Process

```python
def detect_objects_color(frame):
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Apply color masks for each target color
    for color_name, ranges in color_ranges.items():
        mask = cv2.inRange(hsv, lower, upper)
        
        # Clean mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours = cv2.findContours(mask, ...)
        
        # Create bounding boxes
        for contour in contours:
            if area > threshold:
                detections.append(bbox)
```

### Motion Detection Process

```python
def detect_objects_motion(frame):
    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)
    
    # Remove shadows
    fg_mask = cv2.threshold(fg_mask, 200, 255, THRESH_BINARY)
    
    # Clean foreground mask
    fg_mask = cv2.morphologyEx(fg_mask, MORPH_OPEN, kernel)
    
    # Find moving objects
    contours = cv2.findContours(fg_mask, ...)
```

### Centroid Tracking

```python
def update_tracks(detections):
    # Compute centroids of new detections
    new_centroids = [compute_centroid(bbox) for bbox in detections]
    
    # Match with existing tracks (minimum distance)
    for track in existing_tracks:
        distances = [distance(track.centroid, c) for c in new_centroids]
        best_match = min(distances)
        
        if best_match < max_distance:
            track.update(new_centroids[best_match_idx])
        else:
            track.mark_disappeared()
    
    # Create new tracks for unmatched detections
    for unmatched_centroid in unmatched:
        create_new_track(unmatched_centroid)
```

## Performance Optimization

### Speed Improvements

#### Frame Processing
- Reduce resolution (e.g., 640x480)
- Skip frames (process every Nth frame)
- Use ROI to limit search area
- Parallel processing of detections

#### Detection Optimization
- Use faster detection methods (color vs DNN)
- Limit detection frequency (every N frames)
- Use tracking to fill gaps between detections
- Optimize morphological operations

#### Code-Level Optimization
```python
# Process at lower resolution
frame_small = cv2.resize(frame, (640, 480))

# Skip frames
if frame_count % 2 == 0:  # Process every 2nd frame
    detections = detect_objects(frame_small)

# Use ROI
roi = frame[y1:y2, x1:x2]
detections = detect_in_roi(roi)
```

### GPU Acceleration

```python
# Use CUDA if available
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    # Upload to GPU
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)
    
    # Process on GPU
    result = process_gpu(gpu_frame)
    
    # Download result
    result_cpu = result.download()
```

## Real-Time Considerations

### Target Frame Rates

| Application | Target FPS | Notes |
|-------------|-----------|-------|
| Basic surveillance | 15-20 | Sufficient for motion detection |
| Standard monitoring | 25-30 | Smooth playback |
| Real-time tracking | 30+ | No visible lag |
| High-speed analysis | 60+ | Fast moving objects |

### Latency Budget (30 FPS target = 33ms per frame)

| Operation | Time Budget | Optimization |
|-----------|-------------|--------------|
| Frame capture | 1-2 ms | Hardware dependent |
| Preprocessing | 2-3 ms | Vectorize operations |
| Detection | 15-20 ms | Use efficient algorithms |
| Tracking | 3-5 ms | Optimize matching |
| Visualization | 5-8 ms | Limit drawing operations |
| **Total** | **~30 ms** | **~30 FPS** |

## Applications

### Surveillance Systems
- Live monitoring dashboards
- Motion detection alerts
- Person counting
- Intrusion detection

### Industrial Monitoring
- Production line tracking
- Quality control inspection
- Worker safety monitoring
- Inventory management

### Traffic Management
- Vehicle counting and classification
- Traffic flow analysis
- Incident detection
- Parking management

### Sports Analytics
- Player tracking in real-time
- Ball trajectory analysis
- Formation tracking
- Live statistics

### Interactive Installations
- Gesture control
- Interactive displays
- Museum installations
- Gaming applications

## Output and Metrics

### Saved Files
- `realtime_detection_results.png`: Sample frame with detections
- `tracking_trajectories.png`: Object trajectories visualization
- `performance_metrics.txt`: FPS and timing statistics
- `detection_log.csv`: Frame-by-frame detection log

### Performance Metrics
- Average FPS over session
- Detection rate (objects per second)
- Track duration statistics
- Frame processing time breakdown

## Parameters Guide

### Detection Parameters

```python
# Color detection
color_sensitivity = 'medium'  # 'low', 'medium', 'high'
min_object_area = 500         # pixels

# Motion detection
bg_history = 500              # frames for background model
bg_threshold = 16             # sensitivity
detect_shadows = True         # shadow detection

# Tracking
max_disappeared = 30          # frames before removing track
max_distance = 50             # pixels for track association
```

### Performance Tuning

```python
# Frame processing
target_fps = 30
frame_skip = 1                # Process every Nth frame
resize_factor = 1.0           # Scale input frames

# Detection frequency
detect_every_n_frames = 1     # Run detection every N frames
track_every_frame = True      # Track on non-detection frames
```

## Tips for Best Results

### High Performance

1. **Reduce Resolution**: Process at 640x480 or lower
2. **Skip Frames**: Detect every 2-3 frames, track in between
3. **Use Color Detection**: Fastest method for colored objects
4. **Limit ROI**: Process only relevant image regions
5. **Optimize Display**: Reduce visualization overhead

### High Accuracy

1. **Use Multiple Methods**: Combine color and motion
2. **Tune Thresholds**: Adjust for your environment
3. **Smooth Tracking**: Use Kalman filter for predictions
4. **Verify Detections**: Require multiple consecutive detections
5. **Handle Occlusions**: Predict positions during occlusion

### Balanced Approach

```python
# Alternate between detection and tracking
if frame_count % 3 == 0:
    # Full detection (expensive)
    detections = full_detection(frame)
    update_tracks(detections)
else:
    # Track only (cheap)
    predict_track_positions()
    update_visualizations()
```

## Common Issues

**Problem**: FPS too low (< 20)
**Solution**: Reduce resolution, skip frames, use faster detection method

**Problem**: Objects not detected
**Solution**: Tune detection thresholds, check lighting conditions

**Problem**: Track ID switching
**Solution**: Reduce max_distance, improve detection consistency

**Problem**: High CPU usage
**Solution**: Limit frame rate, reduce processing frequency, use GPU

**Problem**: Memory usage growing
**Solution**: Limit track history, clear old tracks, optimize buffers

## System Requirements

### Minimum
- CPU: Dual-core 2.0 GHz
- RAM: 4 GB
- Webcam: 720p @ 15 FPS

### Recommended
- CPU: Quad-core 3.0 GHz
- RAM: 8 GB
- Webcam: 1080p @ 30 FPS
- GPU: NVIDIA with CUDA support

### High Performance
- CPU: 8-core 3.5+ GHz
- RAM: 16 GB
- GPU: NVIDIA RTX series
- Camera: High-speed camera (60+ FPS)

## Comparison with Other Methods

| Method | FPS (CPU) | FPS (GPU) | Accuracy | Use Case |
|--------|-----------|-----------|----------|----------|
| Color | 60+ | N/A | Medium | Colored objects |
| Motion | 40+ | N/A | Good | Moving objects |
| Cascade | 30+ | N/A | Good | Faces, pedestrians |
| DNN (YOLO) | 10-15 | 60+ | Excellent | General objects |
| DNN (SSD) | 15-20 | 80+ | Very Good | Mobile devices |

## Extensions

The code can be extended with:
- Deep learning detection (YOLO, SSD)
- Advanced tracking (DeepSORT, ByteTrack)
- Multi-camera synchronization
- Cloud streaming integration
- Action recognition
- Alert system with notifications
- Database logging
- Web dashboard

## Requirements

- opencv-python >= 4.5.0
- numpy >= 1.19.0
- matplotlib >= 3.3.0
- scipy >= 1.5.0

Optional for enhanced performance:
- opencv-contrib-python (CUDA support)
- cupy (GPU acceleration)

## References

- Bradski, G., & Kaehler, A. (2008). Learning OpenCV
- OpenCV Documentation: Video Analysis
- Zivkovic, Z. (2004). Improved Adaptive Gaussian Mixture Model
- Real-Time Object Detection and Tracking: A Survey
