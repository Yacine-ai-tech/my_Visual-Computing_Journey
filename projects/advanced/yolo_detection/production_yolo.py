"""
Production YOLO Object Detection using Ultralytics YOLOv8
Real-time object detection with pre-trained models on COCO dataset
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("WARNING: ultralytics not installed. Install with: pip install ultralytics")

class ProductionYOLODetector:
    """
    Production-ready YOLO detector using Ultralytics YOLOv8
    Supports YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x
    """
    
    def __init__(self, model_name='yolov8n.pt', conf_threshold=0.25, iou_threshold=0.45, device='cpu'):
        """
        Initialize YOLO detector with pre-trained model
        
        Args:
            model_name: Model size (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
                       n=nano, s=small, m=medium, l=large, x=extra large
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            device: 'cpu', 'cuda', or 'mps'
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Please install ultralytics: pip install ultralytics")
        
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        print(f"Loading {model_name} model...")
        self.model = YOLO(model_name)
        
        # Move to specified device
        if device != 'cpu':
            self.model.to(device)
        
        print(f"✓ Model loaded on {device}")
        print(f"✓ Model classes: {len(self.model.names)} (COCO dataset)")
        
    def detect(self, image, visualize=True):
        """
        Run object detection on image
        
        Args:
            image: Input image (numpy array or path)
            visualize: Whether to draw bounding boxes
            
        Returns:
            results: Detection results from YOLO
            annotated_image: Image with bounding boxes (if visualize=True)
        """
        # Run inference
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        annotated_image = None
        if visualize:
            # Get annotated image
            annotated_image = results[0].plot()
        
        return results[0], annotated_image
    
    def detect_video(self, video_path, output_path=None, show_fps=True):
        """
        Run detection on video file
        
        Args:
            video_path: Path to video file or 0 for webcam
            output_path: Path to save output video (optional)
            show_fps: Whether to display FPS counter
        """
        cap = cv2.VideoCapture(video_path if video_path != 'webcam' else 0)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        fps_history = []
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height} @ {fps} FPS")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # Run detection
            results, annotated = self.detect(frame, visualize=True)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            current_fps = 1.0 / elapsed if elapsed > 0 else 0
            fps_history.append(current_fps)
            
            # Draw FPS
            if show_fps:
                avg_fps = np.mean(fps_history[-30:])  # Last 30 frames
                cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Write frame
            if writer:
                writer.write(annotated)
            
            frame_count += 1
            if frame_count % 30 == 0:
                avg_fps = np.mean(fps_history[-30:])
                print(f"  Processed {frame_count} frames (avg FPS: {avg_fps:.1f})")
        
        cap.release()
        if writer:
            writer.release()
        
        avg_fps = np.mean(fps_history)
        print(f"\n✓ Video processing complete")
        print(f"  Total frames: {frame_count}")
        print(f"  Average FPS: {avg_fps:.1f}")
        
        return frame_count, avg_fps
    
    def get_detection_summary(self, results):
        """
        Get summary of detections
        
        Args:
            results: YOLO results object
            
        Returns:
            Dictionary with detection statistics
        """
        boxes = results.boxes
        
        summary = {
            'total_detections': len(boxes),
            'classes': {},
            'confidences': [],
            'boxes': []
        }
        
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = self.model.names[cls_id]
            
            summary['classes'][class_name] = summary['classes'].get(class_name, 0) + 1
            summary['confidences'].append(conf)
            summary['boxes'].append({
                'class': class_name,
                'confidence': conf,
                'bbox': box.xyxy[0].tolist()
            })
        
        return summary
    
    def export_model(self, format='onnx'):
        """
        Export model to different formats for deployment
        
        Args:
            format: Export format (onnx, torchscript, tflite, etc.)
        """
        print(f"Exporting model to {format}...")
        self.model.export(format=format)
        print(f"✓ Model exported")


def download_sample_images():
    """Download sample images for testing"""
    import urllib.request
    
    sample_urls = [
        ('https://ultralytics.com/images/bus.jpg', 'sample_bus.jpg'),
        ('https://ultralytics.com/images/zidane.jpg', 'sample_people.jpg'),
    ]
    
    print("Downloading sample images...")
    for url, filename in sample_urls:
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"  ✓ Downloaded {filename}")
        except Exception as e:
            print(f"  ✗ Failed to download {filename}: {e}")
    
    return [f for _, f in sample_urls]


def compare_model_sizes():
    """Compare different YOLO model sizes"""
    if not ULTRALYTICS_AVAILABLE:
        print("Ultralytics not available. Skipping comparison.")
        return
    
    models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
    
    # Download sample image
    sample_image = 'sample_bus.jpg'
    if not Path(sample_image).exists():
        print("Downloading sample image...")
        import urllib.request
        urllib.request.urlretrieve('https://ultralytics.com/images/bus.jpg', sample_image)
    
    img = cv2.imread(sample_image)
    
    results_comparison = []
    
    print("\n" + "=" * 70)
    print("Model Size Comparison")
    print("=" * 70)
    
    for model_name in models:
        print(f"\nTesting {model_name}...")
        
        try:
            detector = ProductionYOLODetector(model_name, device='cpu')
            
            # Warmup
            _ = detector.detect(img, visualize=False)
            
            # Benchmark
            times = []
            for _ in range(10):
                start = time.time()
                results, _ = detector.detect(img, visualize=False)
                elapsed = time.time() - start
                times.append(elapsed)
            
            avg_time = np.mean(times)
            fps = 1.0 / avg_time
            
            summary = detector.get_detection_summary(results)
            
            results_comparison.append({
                'model': model_name,
                'avg_time': avg_time * 1000,  # ms
                'fps': fps,
                'detections': summary['total_detections'],
                'classes': len(summary['classes'])
            })
            
            print(f"  Average inference time: {avg_time*1000:.1f} ms")
            print(f"  FPS: {fps:.1f}")
            print(f"  Detections: {summary['total_detections']}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    return results_comparison


def main():
    """Main demonstration of production YOLO detection"""
    
    print("=" * 70)
    print("Production YOLO Object Detection (Ultralytics YOLOv8)")
    print("=" * 70)
    
    if not ULTRALYTICS_AVAILABLE:
        print("\nERROR: ultralytics not installed")
        print("Install with: pip install ultralytics")
        print("\nThis is a production-ready implementation that requires:")
        print("  • ultralytics library (YOLOv8)")
        print("  • Pre-trained models (auto-downloaded)")
        print("  • GPU recommended for best performance")
        return
    
    # Initialize detector with YOLOv8 nano (fastest)
    print("\n1. Initializing YOLOv8 nano detector...")
    detector = ProductionYOLODetector(
        model_name='yolov8n.pt',
        conf_threshold=0.25,
        iou_threshold=0.45,
        device='cpu'  # Change to 'cuda' if GPU available
    )
    
    # Download sample images if needed
    print("\n2. Preparing sample images...")
    sample_images = []
    sample_url = 'https://ultralytics.com/images/bus.jpg'
    sample_file = 'sample_bus.jpg'
    
    if not Path(sample_file).exists():
        print("  Downloading sample image...")
        import urllib.request
        try:
            urllib.request.urlretrieve(sample_url, sample_file)
            print(f"  ✓ Downloaded {sample_file}")
        except Exception as e:
            print(f"  ✗ Failed to download: {e}")
            print("  Creating synthetic test image instead...")
            # Create synthetic image
            img = np.ones((480, 640, 3), dtype=np.uint8) * 128
            cv2.rectangle(img, (100, 100), (300, 400), (0, 255, 0), -1)
            cv2.circle(img, (500, 300), 80, (0, 0, 255), -1)
            cv2.imwrite(sample_file, img)
    
    # Run detection
    print("\n3. Running object detection...")
    img = cv2.imread(sample_file)
    results, annotated = detector.detect(img, visualize=True)
    
    # Get detection summary
    summary = detector.get_detection_summary(results)
    
    print(f"\n  Detection Results:")
    print(f"  ✓ Total detections: {summary['total_detections']}")
    print(f"  ✓ Detected classes: {list(summary['classes'].keys())}")
    
    for class_name, count in summary['classes'].items():
        print(f"    - {class_name}: {count}")
    
    # Save annotated image
    output_file = 'yolo_detection_result.jpg'
    cv2.imwrite(output_file, annotated)
    print(f"\n  ✓ Saved result to {output_file}")
    
    # Visualize
    print("\n4. Creating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'YOLOv8 Detection ({summary["total_detections"]} objects)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('production_yolo_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved visualization to production_yolo_comparison.png")
    
    # Performance benchmark
    print("\n5. Performance Benchmark...")
    times = []
    for i in range(20):
        start = time.time()
        _ = detector.detect(img, visualize=False)
        elapsed = time.time() - start
        times.append(elapsed)
        if (i + 1) % 5 == 0:
            print(f"  Iteration {i+1}/20...")
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print(f"\n  Performance Results:")
    print(f"  ✓ Average inference time: {avg_time*1000:.1f} ms")
    print(f"  ✓ Throughput: {fps:.1f} FPS")
    print(f"  ✓ Min/Max time: {min(times)*1000:.1f} / {max(times)*1000:.1f} ms")
    
    # Summary
    print("\n" + "=" * 70)
    print("Production YOLO Summary")
    print("=" * 70)
    
    print("\nModel Information:")
    print(f"  • Model: YOLOv8 nano (fastest)")
    print(f"  • Dataset: COCO (80 classes)")
    print(f"  • Input size: 640x640")
    print(f"  • Parameters: ~3M")
    
    print("\nAvailable Models:")
    print("  • yolov8n.pt - Nano (3.2M params, fastest)")
    print("  • yolov8s.pt - Small (11.2M params)")
    print("  • yolov8m.pt - Medium (25.9M params)")
    print("  • yolov8l.pt - Large (43.7M params)")
    print("  • yolov8x.pt - Extra Large (68.2M params, most accurate)")
    
    print("\nProduction Features:")
    print("  ✓ Pre-trained on COCO dataset")
    print("  ✓ Real-time inference (GPU: 100+ FPS, CPU: 10-30 FPS)")
    print("  ✓ 80 object classes supported")
    print("  ✓ Export to ONNX, TensorRT, CoreML")
    print("  ✓ Supports image, video, and webcam input")
    
    print("\nDeployment Options:")
    print("  • Local inference (CPU/GPU)")
    print("  • ONNX Runtime for cross-platform")
    print("  • TensorRT for NVIDIA optimization")
    print("  • CoreML for Apple devices")
    print("  • TFLite for mobile/edge devices")
    
    print("\nUsage Examples:")
    print("  # Image detection")
    print("  detector = ProductionYOLODetector('yolov8n.pt')")
    print("  results, annotated = detector.detect('image.jpg')")
    print()
    print("  # Video detection")
    print("  detector.detect_video('video.mp4', 'output.mp4')")
    print()
    print("  # Webcam (real-time)")
    print("  detector.detect_video('webcam', show_fps=True)")
    
    print("\n" + "=" * 70)
    print("For custom training, see Ultralytics documentation:")
    print("https://docs.ultralytics.com/modes/train/")
    print("=" * 70)


if __name__ == "__main__":
    main()
