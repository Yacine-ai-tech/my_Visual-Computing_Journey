# Image Segmentation

Advanced image segmentation techniques for separating regions and objects in images.

## Overview

This module implements three powerful segmentation algorithms:
- **Watershed Segmentation**: Marker-based region growing
- **K-means Clustering**: Color-based segmentation
- **GrabCut**: Interactive foreground/background separation

## Features

### Watershed Segmentation
- Marker-based segmentation using distance transform
- Automatic region detection
- Boundary visualization
- Colored region mapping

### K-means Segmentation
- Cluster-based color segmentation
- Adjustable number of clusters (k)
- Demonstrates k=2, 3, and 5 clustering

### GrabCut Segmentation
- Foreground extraction
- Iterative refinement
- Mask-based visualization

## Usage

```bash
python watershed_segmentation.py
```

The script will:
1. Create a test image with multiple shapes
2. Apply all three segmentation techniques
3. Display and save results

## Algorithm Details

### Watershed
1. Apply morphological opening for noise removal
2. Compute distance transform
3. Find sure foreground and background regions
4. Apply watershed algorithm
5. Mark boundaries between regions

### K-means
1. Reshape image to pixel array
2. Apply k-means clustering
3. Replace pixels with cluster centers
4. Reshape back to image

### GrabCut
1. Define rectangular ROI
2. Initialize foreground/background models
3. Iterate to refine segmentation
4. Extract foreground mask

## Output

All results are saved as PNG files:
- `watershed_result.png`: Watershed segmentation stages
- `kmeans_result.png`: K-means with different k values
- `grabcut_result.png`: GrabCut foreground extraction

## Parameters

- **K-means k**: Number of clusters (default: 2, 3, 5)
- **Watershed iterations**: Morphological operations (default: 2-3)
- **GrabCut iterations**: Refinement cycles (default: 5)

## Applications

- Object separation
- Medical image analysis
- Background removal
- Region-based image analysis
- Pre-processing for object detection
