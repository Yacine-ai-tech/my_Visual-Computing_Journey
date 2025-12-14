# Production-Ready Advanced Projects

This directory contains production-ready implementations using industry-standard models and real datasets.

## 🚀 Quick Start

### Installation

```bash
# Install production dependencies
pip install -r requirements_production.txt

# For GPU support (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Requirements

- **Python**: 3.8+
- **GPU**: Recommended for YOLO and real-time processing (CUDA 11.8+)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 5GB for models

## 📦 Projects

### 1. YOLO Object Detection (Production)

**File**: `yolo_detection/production_yolo.py`

Uses Ultralytics YOLOv8 with pre-trained COCO models.

```python
from production_yolo import ProductionYOLODetector

# Initialize detector
detector = ProductionYOLODetector('yolov8n.pt', device='cuda')

# Detect objects in image
results, annotated = detector.detect('image.jpg')

# Process video
detector.detect_video('video.mp4', 'output.mp4')
```

**Features**:
- Pre-trained on COCO (80 classes)
- Real-time inference (100+ FPS on GPU)
- Multiple model sizes (nano to extra-large)
- Export to ONNX, TensorRT, CoreML

**Models Available**:
- `yolov8n.pt` - Nano (fastest, 3.2M params)
- `yolov8s.pt` - Small (11.2M params)
- `yolov8m.pt` - Medium (25.9M params)
- `yolov8l.pt` - Large (43.7M params)
- `yolov8x.pt` - Extra Large (most accurate, 68.2M params)

### 2. Document Processing with OCR

**File**: `document_processing/production_ocr.py`

Production OCR using Tesseract and EasyOCR.

```python
from production_ocr import ProductionDocumentProcessor

# Initialize processor
processor = ProductionDocumentProcessor(ocr_engine='easyocr')

# Process document
results = processor.process_document('invoice.pdf')
print(results['text'])
```

**Features**:
- Real text extraction (Tesseract/EasyOCR)
- Automatic document detection
- Perspective correction
- 100+ language support
- JSON export

**Supported Documents**:
- Invoices and receipts
- Forms and applications
- ID cards and passports
- Business cards
- PDFs and scanned images

### 3. Real-Time Video Detection (Coming Soon)

High-performance video processing with:
- Multi-stream support
- GPU optimization
- Frame buffering
- Adaptive quality

### 4. Visual Language Models (Coming Soon)

Production VLM using:
- CLIP for vision-language
- BLIP-2 for captioning
- LLaVA for VQA

### 5. Surveillance System (Coming Soon)

Production surveillance with:
- Multi-camera support
- Cloud storage
- Real-time alerts
- Face recognition

## 🔧 Configuration

### GPU Setup

For NVIDIA GPUs:
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Model Downloads

Models are automatically downloaded on first use:
- YOLOv8n: ~6 MB
- YOLOv8s: ~22 MB
- YOLOv8m: ~52 MB
- YOLOv8l: ~87 MB
- YOLOv8x: ~136 MB

Downloaded models are cached in `~/.cache/ultralytics/`

## 📊 Performance

### YOLO Detection (YOLOv8n)

| Device | Resolution | FPS | Latency |
|--------|-----------|-----|---------|
| RTX 3090 | 640x640 | 140 | 7ms |
| RTX 3060 | 640x640 | 90 | 11ms |
| CPU (i7) | 640x640 | 15 | 67ms |

### Document OCR

| Engine | Speed | Accuracy | Languages |
|--------|-------|----------|-----------|
| Tesseract | Fast | Good | 100+ |
| EasyOCR | Medium | Excellent | 80+ |

## 🎓 Training Custom Models

### YOLO Custom Training

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Train on custom dataset
results = model.train(
    data='custom_dataset.yaml',
    epochs=100,
    imgsz=640,
    device='cuda'
)
```

Dataset format (YOLO):
```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

### Document OCR Fine-tuning

For specialized documents:
1. Collect domain-specific dataset
2. Fine-tune Tesseract with custom training data
3. Or use EasyOCR with custom character recognition

## 🚢 Deployment

### Docker Deployment

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY requirements_production.txt .
RUN pip install -r requirements_production.txt

COPY projects/advanced /app/

EXPOSE 8000

CMD ["python", "api_server.py"]
```

### REST API

```python
from fastapi import FastAPI, File, UploadFile
from production_yolo import ProductionYOLODetector

app = FastAPI()
detector = ProductionYOLODetector('yolov8n.pt', device='cuda')

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    image = await file.read()
    results, _ = detector.detect(image)
    return {"detections": results.json()}
```

### Cloud Deployment

**AWS Lambda** (with Docker):
- Use lightweight models (yolov8n)
- Cold start: ~2-5 seconds
- Inference: 100-200ms on CPU

**Google Cloud Run**:
- Auto-scaling
- GPU support
- Pay per use

**Azure Container Instances**:
- GPU support (K80, P100, V100)
- Managed service

## 📈 Optimization

### Model Optimization

```python
# Export to ONNX for faster inference
model = YOLO('yolov8n.pt')
model.export(format='onnx')

# Use ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession('yolov8n.onnx')
```

### TensorRT Optimization (NVIDIA)

```python
# Export to TensorRT
model.export(format='engine', device='0')

# 2-3x faster inference on NVIDIA GPUs
```

### Batch Processing

```python
# Process multiple images efficiently
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = detector.model.predict(images, batch=True)
```

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**:
```python
# Use smaller model or reduce batch size
detector = ProductionYOLODetector('yolov8n.pt')  # Smallest model
```

**Slow CPU Inference**:
```python
# Export to ONNX for better CPU performance
model.export(format='onnx')
```

**Tesseract Not Found**:
```bash
# Install Tesseract binary
# Ubuntu: sudo apt-get install tesseract-ocr
# macOS: brew install tesseract
# Windows: Download from GitHub releases
```

## 📚 Resources

### Documentation
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [PyTorch](https://pytorch.org/docs/)

### Datasets
- [COCO](https://cocodataset.org/) - Object detection
- [Open Images](https://storage.googleapis.com/openimages/web/index.html) - Large-scale
- [Roboflow Universe](https://universe.roboflow.com/) - Custom datasets

### Model Hubs
- [Ultralytics Hub](https://hub.ultralytics.com/) - YOLO models
- [Hugging Face](https://huggingface.co/models) - VLMs and transformers
- [ONNX Model Zoo](https://github.com/onnx/models) - Optimized models

## 📄 License

See main repository LICENSE file.

## 🤝 Contributing

For production improvements:
1. Test on real-world data
2. Benchmark performance
3. Document edge cases
4. Submit pull request

## ⚠️ Production Checklist

Before deploying:
- [ ] Test on representative dataset
- [ ] Benchmark performance
- [ ] Set up monitoring
- [ ] Configure error handling
- [ ] Add logging
- [ ] Implement rate limiting
- [ ] Set up CI/CD
- [ ] Document API
- [ ] Security audit
- [ ] Load testing

---

**Note**: These are production-ready implementations designed for real-world use. They require external dependencies and model downloads. For educational/demo versions, see the original project files.
