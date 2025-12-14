# Optical Flow Visualization

Motion estimation and visualization using optical flow algorithms.

## Overview

This module implements two major optical flow approaches:
- **Sparse Optical Flow** (Lucas-Kanade)
- **Dense Optical Flow** (Farneback)

Plus multiple visualization techniques for understanding motion in video sequences.

## Features

### Lucas-Kanade Method
- Tracks individual feature points
- Shows motion trajectories over time
- Fast and efficient for sparse tracking
- Uses pyramid approach for large motions

### Farneback Method
- Computes optical flow for every pixel
- Dense motion field estimation
- Polynomial expansion approach
- Color-coded visualization

### Visualizations
- Trajectory tracking (sparse)
- HSV color mapping (dense)
- Arrow field visualization
- Motion statistics and analysis

## Usage

```bash
python optical_flow_demo.py
```

The script will:
1. Generate synthetic video with moving shapes
2. Compute sparse optical flow (Lucas-Kanade)
3. Compute dense optical flow (Farneback)
4. Visualize flow fields with arrows
5. Analyze motion statistics
6. Save results

## Algorithm Details

### Lucas-Kanade (Sparse)
1. Detect good features to track (Shi-Tomasi)
2. Compute optical flow for each point
3. Track points across frames
4. Draw trajectories

**Parameters:**
- `winSize`: Search window size (15x15)
- `maxLevel`: Pyramid levels (2)
- `maxCorners`: Maximum features to track (100)

### Farneback (Dense)
1. Build image pyramids
2. Estimate polynomial expansion
3. Compute dense flow field
4. Visualize with HSV color mapping

**Parameters:**
- `pyr_scale`: Pyramid scale factor (0.5)
- `levels`: Number of pyramid levels (3)
- `winsize`: Averaging window size (15)
- `iterations`: Iterations per level (3)

## Visualization Encoding

### HSV Color Map (Dense Flow)
- **Hue**: Direction of motion (angle)
- **Saturation**: Always maximum (255)
- **Value**: Magnitude of motion (speed)

### Arrow Visualization
- Arrow direction: Motion direction
- Arrow length: Motion magnitude
- Only significant motion shown (threshold: 1.0 pixel)

## Applications

### Video Analysis
- Motion detection and tracking
- Action recognition
- Gesture recognition
- Gait analysis

### Computer Vision
- Visual odometry
- Structure from motion
- Video stabilization
- Background subtraction

### Video Processing
- Frame interpolation
- Video compression (motion vectors)
- Temporal filtering
- Super-resolution

### Robotics
- Ego-motion estimation
- Obstacle detection
- Navigation
- Visual servoing

## Output Files

- `optical_flow_results.png`: Complete comparison of all methods
- `optical_flow_comparison.png`: Side-by-side frame comparison

## Performance Comparison

| Method | Speed | Coverage | Use Case |
|--------|-------|----------|----------|
| Lucas-Kanade | Fast | Sparse | Point tracking |
| Farneback | Medium | Dense | Full motion field |
| Arrows | Medium | Dense | Motion visualization |

## Tips for Best Results

### For Sparse Flow
- Increase `maxCorners` for more tracking points
- Adjust `qualityLevel` to filter weak corners
- Use larger `winSize` for faster motion

### For Dense Flow
- Increase `levels` for larger motions
- Decrease `winsize` for finer details
- Adjust `poly_n` for smoothness

## Common Issues

**Problem**: Points disappear during tracking
**Solution**: Reduce `qualityLevel` or increase `maxCorners`

**Problem**: Flow looks noisy
**Solution**: Increase `winsize` or `poly_sigma`

**Problem**: Missing fast motion
**Solution**: Increase pyramid `levels` or `maxLevel`

**Problem**: Slow performance
**Solution**: Reduce image resolution or use sparse flow

## Extensions

The code can be extended to:
- Real-time webcam processing
- Video file analysis
- Multi-scale flow estimation
- Flow-based segmentation
- Temporal filtering
- Motion detection zones
