# YOLO Object Detection

YOLO-style object detection implementation with both educational demo and production-ready versions.

## Overview

This module provides two implementations of YOLO (You Only Look Once) object detection:
- **Educational Demo** (`yolo_object_detection.py`): Self-contained implementation showing YOLO concepts
- **Production** (`production_yolo.py`): Industry-standard YOLOv8 with pre-trained models

## Features

### Educational Demo Features

#### Grid-Based Detection
- 7x7 grid cell division (configurable)
- Each cell predicts bounding boxes
- Multiple boxes per cell (typically 2)
- Class probability predictions per cell

#### Detection Components
- Color-based object detection (simplified)
- Confidence scoring system
- Non-Maximum Suppression (NMS)
- Intersection over Union (IOU) calculation

#### Visualization
- Grid overlay display
- Bounding box predictions
- Confidence scores
- Class labels
- Performance comparison across grid sizes

### Production Features (YOLOv8)

#### Pre-trained Models
- COCO dataset (80 object classes)
- Multiple model sizes (nano to extra-large)
- Transfer learning support
- Custom training capabilities

#### Performance
- Real-time inference (100+ FPS on GPU)
- Batch processing support
- Video stream processing
- Image and video input

#### Export Capabilities
- ONNX format
- TensorRT optimization
- CoreML for iOS
- OpenVINO for Intel hardware

## Usage

### Educational Demo

```bash
cd projects/advanced/yolo_detection
python yolo_object_detection.py
```

The demo will:
1. Create synthetic test images with colored objects
2. Apply grid-based detection
3. Perform NMS to remove duplicates
4. Visualize results with confidence scores
5. Compare different grid sizes (5x5, 7x7, 9x9)

### Production (Ultralytics YOLOv8)

```bash
# Install dependencies
pip install ultralytics

# Run detection
cd projects/advanced/yolo_detection
python production_yolo.py
```

For detailed production setup, see `projects/advanced/README_PRODUCTION.md`.

## YOLO Algorithm Explained

### Core Concept
YOLO treats object detection as a regression problem:
1. Divide image into SxS grid (e.g., 7x7)
2. Each grid cell predicts B bounding boxes
3. Each bounding box predicts: (x, y, w, h, confidence)
4. Each cell predicts class probabilities
5. Final output: Grid × Boxes × (5 + Classes)

### Grid Cell Predictions

Each cell outputs:
```
- Bounding boxes: B × (x, y, w, h, confidence)
- Class probabilities: C classes
- Total: (B × 5) + C values per cell
```

For 7×7 grid, 2 boxes, 20 classes:
- Output tensor: 7 × 7 × 30

### Confidence Score
```
Confidence = P(Object) × IOU(pred, truth)
```
- P(Object): Probability cell contains object
- IOU: Intersection over Union with ground truth

### Class Probability
```
Class Score = Confidence × P(Class|Object)
```

### Non-Maximum Suppression (NMS)

1. Sort all detections by confidence
2. Select detection with highest confidence
3. Remove detections with IOU > threshold (e.g., 0.5)
4. Repeat for remaining detections

## Educational Implementation Details

### Grid-Based Detection
```python
class YOLODetector:
    def __init__(self, grid_size=7, num_boxes=2, num_classes=3):
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
```

### Simplified Detection Process
1. Divide image into grid cells
2. For each cell, detect objects by color
3. Calculate confidence based on coverage
4. Assign class based on dominant color
5. Generate bounding boxes
6. Apply NMS to remove duplicates

### IOU Calculation
```python
def calculate_iou(box1, box2):
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0
```

## Production YOLOv8 Models

### Model Variants

| Model | Size (MB) | Params (M) | FPS (GPU) | mAP@0.5 |
|-------|-----------|------------|-----------|---------|
| YOLOv8n | 6 | 3.2 | 140+ | 37.3 |
| YOLOv8s | 22 | 11.2 | 110+ | 44.9 |
| YOLOv8m | 52 | 25.9 | 80+ | 50.2 |
| YOLOv8l | 87 | 43.7 | 60+ | 52.9 |
| YOLOv8x | 136 | 68.2 | 45+ | 53.9 |

### COCO Classes (80 total)

Common classes include:
- Person, bicycle, car, motorcycle, airplane, bus, train, truck
- Traffic light, fire hydrant, stop sign, parking meter, bench
- Cat, dog, horse, sheep, cow, elephant, bear, zebra
- Backpack, umbrella, handbag, tie, suitcase
- Frisbee, skis, snowboard, sports ball, kite
- Bottle, wine glass, cup, fork, knife, spoon, bowl
- And 50+ more...

## Applications

### Security & Surveillance
- Intrusion detection
- Crowd monitoring
- Abandoned object detection
- Person counting

### Autonomous Vehicles
- Pedestrian detection
- Vehicle detection
- Traffic sign recognition
- Lane detection preparation

### Retail Analytics
- Shelf monitoring
- Customer counting
- Product recognition
- Queue management

### Industrial Automation
- Defect detection
- Part recognition
- Assembly verification
- Quality control

### Agriculture
- Crop disease detection
- Livestock monitoring
- Weed detection
- Harvest optimization

## Parameters Guide

### Educational Demo
```python
detector = YOLODetector(
    grid_size=7,           # Grid dimension (7x7)
    num_boxes=2,           # Boxes per cell
    num_classes=3          # Number of object classes
)

detector.confidence_threshold = 0.5  # Minimum confidence
detector.nms_threshold = 0.4         # IOU threshold for NMS
```

### Production YOLOv8
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load model

results = model.predict(
    source='image.jpg',
    conf=0.25,              # Confidence threshold
    iou=0.7,                # NMS IOU threshold
    imgsz=640,              # Input image size
    device='cuda',          # GPU device
    half=True,              # FP16 inference
    max_det=300             # Maximum detections
)
```

## Tips for Best Results

### Educational Demo
- Adjust grid size based on object density
- Tune confidence threshold to balance precision/recall
- Modify NMS threshold based on object overlap
- Use appropriate color ranges for your objects

### Production YOLOv8

**Model Selection**
- Use nano (n) for edge devices, real-time requirements
- Use small (s) or medium (m) for balanced performance
- Use large (l) or extra-large (x) for maximum accuracy

**Inference Optimization**
- Use GPU for real-time processing
- Enable half-precision (FP16) on compatible GPUs
- Export to TensorRT for 2-3x speedup
- Batch process multiple images
- Reduce input size for faster inference

**Accuracy Improvements**
- Use larger model variant
- Increase input image size (640 → 1280)
- Lower confidence threshold
- Apply test-time augmentation

## Performance Comparison

### Educational vs Production

| Aspect | Educational Demo | Production YOLOv8 |
|--------|-----------------|-------------------|
| Purpose | Learning, visualization | Real-world deployment |
| Speed | Fast (simplified) | Very fast (optimized) |
| Accuracy | Basic (color-based) | Excellent (deep learning) |
| Classes | 3 (demo colors) | 80 (COCO dataset) |
| Setup | Zero dependencies | Requires model download |
| Size | < 1 KB | 6-136 MB |

## Common Issues

**Educational Demo:**

**Problem**: Grid too coarse, missing small objects
**Solution**: Increase grid size (e.g., 9x9 or 11x11)

**Problem**: Too many duplicate detections
**Solution**: Increase NMS threshold or confidence threshold

**Production:**

**Problem**: Out of memory on GPU
**Solution**: Use smaller model (nano), reduce batch size, or image size

**Problem**: Slow inference on CPU
**Solution**: Export to ONNX, use smaller model, reduce image size

**Problem**: Poor detection on custom data
**Solution**: Fine-tune model on your dataset

## Training Custom Models (Production)

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

# Train on custom dataset
results = model.train(
    data='dataset.yaml',    # Dataset configuration
    epochs=100,             # Training epochs
    imgsz=640,              # Image size
    batch=16,               # Batch size
    device='cuda',          # GPU device
    workers=8,              # Data loader workers
    project='runs/train',   # Save directory
    name='custom_model'     # Experiment name
)
```

### Dataset Format (YOLO)
```
dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── val/
│       ├── img3.jpg
│       └── img4.jpg
├── labels/
│   ├── train/
│   │   ├── img1.txt  # class x_center y_center width height
│   │   └── img2.txt
│   └── val/
│       ├── img3.txt
│       └── img4.txt
└── dataset.yaml
```

## Extensions

The code can be extended with:
- Object tracking (combining with tracking algorithms)
- Action recognition (temporal YOLO)
- Instance segmentation (YOLOv8-seg)
- Pose estimation (YOLOv8-pose)
- Oriented bounding boxes (OBB)
- Multi-camera fusion
- Edge deployment (TensorRT, CoreML)

## Requirements

### Educational Demo
- opencv-python >= 4.5.0
- numpy >= 1.19.0
- matplotlib >= 3.3.0

### Production
- ultralytics >= 8.0.0
- torch >= 1.8.0
- torchvision >= 0.9.0

See `requirements_production.txt` for full list.

## References

- Redmon, J., et al. (2016). You Only Look Once: Unified, Real-Time Object Detection
- Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement
- Bochkovskiy, A., et al. (2020). YOLOv4: Optimal Speed and Accuracy
- Ultralytics YOLOv8 Documentation: https://docs.ultralytics.com/
- COCO Dataset: https://cocodataset.org/
