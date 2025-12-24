"""
YOLO Object Detection - Built from Scratch to Understand How It Works

I implemented this to truly understand YOLO's revolutionary approach to object detection.
Instead of using sliding windows or region proposals, YOLO looks at the entire image once
and predicts bounding boxes and class probabilities directly. This makes it incredibly fast!

This is an educational implementation that demonstrates the core concepts:
- Grid-based detection (dividing images into cells)
- Multiple bounding box predictions per cell
- Non-maximum suppression to filter duplicates
- Confidence scoring that combines objectness and localization

For production use, check out production_yolo.py which uses Ultralytics YOLOv8.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

class YOLODetector:
    """
    A simplified YOLO-style object detector built from scratch
    
    YOLO's key insight: treat object detection as a regression problem!
    Instead of sliding windows over the image, we divide it into a grid and
    let each grid cell predict bounding boxes for objects whose center falls in that cell.
    
    This approach is what makes YOLO so fast - single forward pass through the network.
    """
    
    def __init__(self, grid_size=7, num_boxes=2, num_classes=3):
        """
        Initialize the YOLO detector with grid configuration
        
        The grid_size is crucial - YOLO divides the image into SxS cells.
        Original YOLO used 7x7, but you can experiment! Larger grids can detect
        smaller objects but are slower. Each cell predicts multiple bounding boxes
        (num_boxes) to handle overlapping objects.
        
        Args:
            grid_size: How many cells to divide the image into (e.g., 7 means 7x7 = 49 cells)
            num_boxes: How many bounding boxes each cell should predict (usually 2)
            num_classes: Number of object categories we want to detect
        """
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        
    def create_grid_representation(self, img_shape):
        """
        Visualize the grid overlay on the image
        
        This helps understand how YOLO divides up the image. Each cell is responsible
        for detecting objects whose center falls within that cell. This is why YOLO
        struggles with small, clustered objects - multiple objects might fall in the
        same cell, but each cell only predicts a fixed number of boxes.
        """
        h, w = img_shape[:2]
        cell_h = h // self.grid_size
        cell_w = w // self.grid_size
        
        grid_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw grid lines
        for i in range(self.grid_size + 1):
            y = i * cell_h
            cv2.line(grid_img, (0, y), (w, y), (100, 100, 100), 1)
        
        for j in range(self.grid_size + 1):
            x = j * cell_w
            cv2.line(grid_img, (x, 0), (x, h), (100, 100, 100), 1)
        
        return grid_img, cell_h, cell_w
    
    def detect_objects_simple(self, img):
        """
        Simplified object detection using color and contours
        
        In a real YOLO implementation, this would be a deep neural network!
        But for educational purposes, I'm using color detection to simulate
        how the network would identify objects. The important part is understanding
        the grid-based approach and how predictions are made per cell.
        
        The real magic of YOLO is in the neural network architecture and training,
        but the detection pipeline and NMS logic shown here are exactly what's used
        in production YOLO models.
        """
        h, w = img.shape[:2]
        cell_h = h // self.grid_size
        cell_w = w // self.grid_size
        
        detections = []
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for different "classes"
        color_ranges = {
            'red': ([0, 120, 70], [10, 255, 255]),
            'green': ([40, 40, 40], [80, 255, 255]),
            'blue': ([100, 50, 50], [130, 255, 255])
        }
        
        class_names = ['red', 'green', 'blue']
        
        # Process each grid cell
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell_y1 = i * cell_h
                cell_y2 = (i + 1) * cell_h
                cell_x1 = j * cell_w
                cell_x2 = (j + 1) * cell_w
                
                cell_hsv = hsv[cell_y1:cell_y2, cell_x1:cell_x2]
                
                # Check each color class
                for class_id, class_name in enumerate(class_names):
                    lower, upper = color_ranges[class_name]
                    lower = np.array(lower)
                    upper = np.array(upper)
                    
                    # Special handling for red (wraps around in HSV)
                    if class_name == 'red':
                        lower2 = np.array([170, 120, 70])
                        upper2 = np.array([180, 255, 255])
                        mask1 = cv2.inRange(cell_hsv, lower, upper)
                        mask2 = cv2.inRange(cell_hsv, lower2, upper2)
                        mask = cv2.bitwise_or(mask1, mask2)
                    else:
                        mask = cv2.inRange(cell_hsv, lower, upper)
                    
                    # Calculate confidence based on mask coverage
                    coverage = np.sum(mask > 0) / (cell_h * cell_w)
                    
                    if coverage > 0.1:  # Minimum coverage threshold
                        # Find contours in mask
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                                       cv2.CHAIN_APPROX_SIMPLE)
                        
                        for contour in contours:
                            if cv2.contourArea(contour) > 100:
                                x, y, bw, bh = cv2.boundingRect(contour)
                                
                                # Convert to image coordinates
                                abs_x = cell_x1 + x
                                abs_y = cell_y1 + y
                                
                                # Calculate center and normalize
                                center_x = (abs_x + bw / 2) / w
                                center_y = (abs_y + bh / 2) / h
                                norm_w = bw / w
                                norm_h = bh / h
                                
                                confidence = min(coverage * 2, 1.0)
                                
                                detection = {
                                    'grid_i': i,
                                    'grid_j': j,
                                    'class': class_id,
                                    'class_name': class_name,
                                    'confidence': confidence,
                                    'bbox': [center_x, center_y, norm_w, norm_h],
                                    'abs_bbox': [abs_x, abs_y, bw, bh]
                                }
                                detections.append(detection)
        
        return detections
    
    def non_max_suppression(self, detections):
        """
        Apply Non-Maximum Suppression - crucial for any object detector!
        
        When we detect objects, we often get multiple overlapping bounding boxes
        for the same object. NMS solves this by keeping only the highest-confidence
        detection and suppressing (removing) nearby boxes that likely represent
        the same object.
        
        The algorithm:
        1. Sort all detections by confidence
        2. Take the highest confidence detection and keep it
        3. Remove all other detections that overlap significantly (high IOU)
        4. Repeat with remaining detections
        
        This is used in YOLO, Faster R-CNN, and pretty much every modern detector!
        """
        if len(detections) == 0:
            return []
        
        # Group by class
        by_class = {}
        for det in detections:
            class_id = det['class']
            if class_id not in by_class:
                by_class[class_id] = []
            by_class[class_id].append(det)
        
        final_detections = []
        
        # Apply NMS per class
        for class_id, dets in by_class.items():
            # Sort by confidence
            dets = sorted(dets, key=lambda x: x['confidence'], reverse=True)
            
            keep = []
            while len(dets) > 0:
                # Keep highest confidence detection
                best = dets.pop(0)
                keep.append(best)
                
                # Remove overlapping detections
                remaining = []
                for det in dets:
                    iou = self.calculate_iou(best['abs_bbox'], det['abs_bbox'])
                    if iou < self.nms_threshold:
                        remaining.append(det)
                dets = remaining
            
            final_detections.extend(keep)
        
        return final_detections
    
    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IOU) - fundamental metric in object detection
        
        IOU measures how much two bounding boxes overlap. It's the ratio of:
        - Intersection area (where boxes overlap)
        - Union area (total area covered by both boxes)
        
        IOU = 1.0 means perfect overlap (same box)
        IOU = 0.0 means no overlap at all
        IOU > 0.5 typically means the boxes represent the same object
        
        This metric is used everywhere: NMS, evaluation (mAP), anchor matching in training...
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0
        
        return inter_area / union_area
    
    def draw_detections(self, img, detections):
        """Draw bounding boxes and labels"""
        result = img.copy()
        
        colors = {
            0: (0, 0, 255),    # red
            1: (0, 255, 0),    # green
            2: (255, 0, 0)     # blue
        }
        
        for det in detections:
            if det['confidence'] < self.confidence_threshold:
                continue
            
            x, y, w, h = det['abs_bbox']
            class_id = det['class']
            class_name = det['class_name']
            confidence = det['confidence']
            
            color = colors.get(class_id, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw label background
            cv2.rectangle(result, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(result, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result


def create_test_image_with_objects():
    """
    Create a synthetic test image with colored objects
    
    I'm generating synthetic data here so anyone can run this code immediately
    without needing to download datasets. In real applications, you'd use
    actual images, but this approach is perfect for learning and demonstrating
    the algorithm's behavior.
    """
    img = np.ones((448, 448, 3), dtype=np.uint8) * 220
    
    # Add gradient background
    for i in range(448):
        img[i, :] = [220 + i * 30 // 448, 230 + i * 20 // 448, 240]
    
    # Red objects
    cv2.circle(img, (100, 100), 40, (0, 0, 255), -1)
    cv2.rectangle(img, (300, 50), (380, 130), (0, 0, 255), -1)
    
    # Green objects
    cv2.circle(img, (350, 300), 35, (0, 255, 0), -1)
    cv2.rectangle(img, (80, 250), (150, 320), (0, 255, 0), -1)
    
    # Blue objects
    cv2.ellipse(img, (250, 350), (50, 30), 0, 0, 360, (255, 0, 0), -1)
    cv2.rectangle(img, (150, 150), (220, 220), (255, 0, 0), -1)
    
    return img


def visualize_yolo_pipeline(img, detector):
    """Visualize the YOLO detection pipeline"""
    
    # 1. Show grid overlay
    grid_img, _, _ = detector.create_grid_representation(img.shape)
    img_with_grid = cv2.addWeighted(img, 0.7, grid_img, 0.3, 0)
    
    # 2. Detect objects
    detections = detector.detect_objects_simple(img)
    print(f"   - Initial detections: {len(detections)}")
    
    # 3. Apply NMS
    filtered_detections = detector.non_max_suppression(detections)
    print(f"   - After NMS: {len(filtered_detections)}")
    
    # 4. Draw results
    result = detector.draw_detections(img, filtered_detections)
    
    return img_with_grid, result, filtered_detections


def demonstrate_yolo_concepts():
    """Demonstrate key YOLO concepts"""
    
    print("\n" + "=" * 70)
    print("YOLO Object Detection Concepts")
    print("=" * 70)
    
    # 1. Grid-based detection
    print("\n1. Grid-Based Detection:")
    print("   - Image divided into SxS grid (e.g., 7x7)")
    print("   - Each cell predicts B bounding boxes")
    print("   - Each box has: x, y, w, h, confidence")
    print("   - Each cell predicts class probabilities")
    
    # 2. Bounding box format
    print("\n2. Bounding Box Encoding:")
    print("   - (x, y): center relative to grid cell")
    print("   - (w, h): width and height relative to image")
    print("   - All values normalized to [0, 1]")
    
    # 3. Confidence score
    print("\n3. Confidence Score:")
    print("   - Pr(Object) * IOU(pred, truth)")
    print("   - Measures objectness and localization accuracy")
    
    # 4. Class prediction
    print("\n4. Class Prediction:")
    print("   - Conditional probability Pr(Class|Object)")
    print("   - Final score: confidence * class probability")
    
    # 5. Non-Maximum Suppression
    print("\n5. Non-Maximum Suppression (NMS):")
    print("   - Removes duplicate detections")
    print("   - Keeps boxes with highest confidence")
    print("   - Suppresses overlapping boxes (IOU threshold)")


def compare_detection_parameters():
    """Compare different YOLO parameters"""
    
    img = create_test_image_with_objects()
    
    results = {}
    
    # Test different grid sizes
    grid_sizes = [3, 5, 7]
    
    for gs in grid_sizes:
        detector = YOLODetector(grid_size=gs)
        detections = detector.detect_objects_simple(img)
        filtered = detector.non_max_suppression(detections)
        result = detector.draw_detections(img, filtered)
        
        results[f'Grid {gs}x{gs}'] = (result, len(filtered))
    
    return results


def main():
    """Main function demonstrating YOLO object detection"""
    
    print("=" * 70)
    print("YOLO-Style Object Detection")
    print("=" * 70)
    
    # Create test image
    print("\nCreating test image with multiple objects...")
    img = create_test_image_with_objects()
    print("   ✓ Test image created")
    
    # Initialize detector
    print("\nInitializing YOLO detector (7x7 grid)...")
    detector = YOLODetector(grid_size=7, num_boxes=2, num_classes=3)
    print("   ✓ Detector initialized")
    
    # Demonstrate YOLO concepts
    demonstrate_yolo_concepts()
    
    # Run detection pipeline
    print("\n" + "=" * 70)
    print("Running Detection Pipeline")
    print("=" * 70)
    print("\n1. Grid-based detection...")
    grid_img, result, detections = visualize_yolo_pipeline(img, detector)
    print("   ✓ Detection complete")
    
    # Visualize results
    print("\n2. Visualizing results...")
    
    fig = plt.figure(figsize=(16, 5))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB))
    plt.title(f'7x7 Detection Grid')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(f'Detections ({len(detections)} objects)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('yolo_detection_pipeline.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("   ✓ Visualization saved")
    
    # Compare different grid sizes
    print("\n3. Comparing grid sizes...")
    comparison_results = compare_detection_parameters()
    
    fig = plt.figure(figsize=(15, 5))
    
    for idx, (name, (result_img, count)) in enumerate(comparison_results.items(), 1):
        plt.subplot(1, 3, idx)
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title(f'{name} ({count} detections)')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('yolo_grid_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("   ✓ Grid size comparison complete")
    
    # Print detection summary
    print("\n" + "=" * 70)
    print("Detection Summary")
    print("=" * 70)
    
    for det in detections:
        print(f"\n{det['class_name'].upper()}:")
        print(f"  Grid cell: ({det['grid_i']}, {det['grid_j']})")
        print(f"  Confidence: {det['confidence']:.3f}")
        print(f"  Bounding box: ({det['bbox'][0]:.3f}, {det['bbox'][1]:.3f}, "
              f"{det['bbox'][2]:.3f}, {det['bbox'][3]:.3f})")
    
    # Summary
    print("\n" + "=" * 70)
    print("YOLO Detection Complete")
    print("=" * 70)
    print("\nKey Advantages of YOLO:")
    print("  ✓ Real-time detection (45+ FPS)")
    print("  ✓ Single forward pass through network")
    print("  ✓ Sees entire image (better context)")
    print("  ✓ Fewer false positives in background")
    print("  ✓ Generalizes well to new domains")
    
    print("\nApplications:")
    print("  • Autonomous vehicles")
    print("  • Surveillance systems")
    print("  • Retail analytics")
    print("  • Sports analysis")
    print("  • Medical imaging")
    
    print("\nFor production use:")
    print("  • Use pre-trained YOLOv5/YOLOv8 models")
    print("  • Consider Ultralytics YOLO library")
    print("  • Fine-tune on custom datasets")
    print("  • Optimize with TensorRT for edge devices")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
