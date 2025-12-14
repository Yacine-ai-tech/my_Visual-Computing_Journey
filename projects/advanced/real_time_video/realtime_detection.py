"""
Real-Time Video Object Detection
Demonstrates real-time detection and tracking on video streams
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

class RealTimeDetector:
    """
    Real-time object detector with tracking and performance monitoring
    """
    
    def __init__(self, detection_method='color'):
        """
        Initialize real-time detector
        
        Args:
            detection_method: 'color', 'motion', or 'cascade'
        """
        self.detection_method = detection_method
        self.tracker_type = 'CSRT'  # Can be CSRT, KCF, MOSSE
        self.trackers = []
        self.track_ids = []
        self.next_id = 0
        
        # Performance metrics
        self.fps_history = deque(maxlen=30)
        self.detection_times = deque(maxlen=30)
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        
        # Initialize cascade classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect_objects_color(self, frame):
        """Detect objects by color"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color ranges
        color_ranges = {
            'red': [([0, 120, 70], [10, 255, 255]), 
                   ([170, 120, 70], [180, 255, 255])],
            'green': [([40, 40, 40], [80, 255, 255])],
            'blue': [([100, 50, 50], [130, 255, 255])]
        }
        
        detections = []
        
        for color_name, ranges in color_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            for lower, upper in ranges:
                lower = np.array(lower)
                upper = np.array(upper)
                color_mask = cv2.inRange(hsv, lower, upper)
                mask = cv2.bitwise_or(mask, color_mask)
            
            # Clean up mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    detections.append({
                        'bbox': (x, y, w, h),
                        'class': color_name,
                        'confidence': min(area / 5000, 1.0)
                    })
        
        return detections
    
    def detect_objects_motion(self, frame):
        """Detect moving objects using background subtraction"""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                
                detections.append({
                    'bbox': (x, y, w, h),
                    'class': 'moving_object',
                    'confidence': min(area / 10000, 1.0)
                })
        
        return detections
    
    def detect_objects_cascade(self, frame):
        """Detect faces using Haar cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        detections = []
        
        for (x, y, w, h) in faces:
            detections.append({
                'bbox': (x, y, w, h),
                'class': 'face',
                'confidence': 1.0
            })
        
        return detections
    
    def detect(self, frame):
        """Run detection based on selected method"""
        start_time = time.time()
        
        if self.detection_method == 'color':
            detections = self.detect_objects_color(frame)
        elif self.detection_method == 'motion':
            detections = self.detect_objects_motion(frame)
        elif self.detection_method == 'cascade':
            detections = self.detect_objects_cascade(frame)
        else:
            detections = []
        
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)
        
        return detections
    
    def draw_detections(self, frame, detections, show_fps=True):
        """Draw bounding boxes and labels"""
        result = frame.copy()
        
        for det in detections:
            x, y, w, h = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            
            # Color based on class
            if class_name == 'red':
                color = (0, 0, 255)
            elif class_name == 'green':
                color = (0, 255, 0)
            elif class_name == 'blue':
                color = (255, 0, 0)
            elif class_name == 'face':
                color = (255, 255, 0)
            else:
                color = (0, 255, 255)
            
            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(result, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw FPS and performance metrics
        if show_fps and len(self.fps_history) > 0:
            avg_fps = np.mean(self.fps_history)
            avg_det_time = np.mean(self.detection_times) * 1000  # ms
            
            cv2.putText(result, f"FPS: {avg_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result, f"Detection: {avg_det_time:.1f}ms", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result, f"Objects: {len(detections)}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result


def create_synthetic_video(num_frames=100, fps=30):
    """Create synthetic video with moving objects"""
    width, height = 640, 480
    frames = []
    
    for i in range(num_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 240
        
        # Add gradient background
        for y in range(height):
            frame[y, :] = [200 + y * 40 // height, 220 + y * 30 // height, 240]
        
        # Moving red ball
        ball_x = int(50 + (i * 5) % (width - 100))
        ball_y = int(100 + 50 * np.sin(i * 0.1))
        cv2.circle(frame, (ball_x, ball_y), 25, (0, 0, 255), -1)
        
        # Moving green car
        car_x = int(width - 100 - (i * 4) % (width - 200))
        car_y = 250
        cv2.rectangle(frame, (car_x, car_y), (car_x + 80, car_y + 40), 
                     (0, 255, 0), -1)
        
        # Stationary blue box
        cv2.rectangle(frame, (450, 350), (550, 430), (255, 0, 0), -1)
        
        # Add frame number
        cv2.putText(frame, f"Frame {i}", (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        frames.append(frame)
    
    return frames, fps


def process_video_realtime(frames, detector, fps):
    """Process video frames in real-time simulation"""
    results = []
    frame_times = []
    
    for i, frame in enumerate(frames):
        start_time = time.time()
        
        # Detect objects
        detections = detector.detect(frame)
        
        # Draw results
        result = detector.draw_detections(frame, detections, show_fps=True)
        
        # Calculate FPS
        elapsed = time.time() - start_time
        current_fps = 1.0 / elapsed if elapsed > 0 else 0
        detector.fps_history.append(current_fps)
        
        frame_times.append(elapsed)
        results.append(result)
        
        # Print progress
        if (i + 1) % 20 == 0:
            avg_fps = np.mean(list(detector.fps_history)[-20:])
            print(f"   Processed {i + 1}/{len(frames)} frames (avg FPS: {avg_fps:.1f})")
    
    return results, frame_times


def analyze_performance(frame_times, fps):
    """Analyze processing performance"""
    frame_times = np.array(frame_times)
    
    stats = {
        'mean_time': np.mean(frame_times) * 1000,  # ms
        'std_time': np.std(frame_times) * 1000,
        'min_time': np.min(frame_times) * 1000,
        'max_time': np.max(frame_times) * 1000,
        'mean_fps': 1.0 / np.mean(frame_times),
        'realtime_capable': np.mean(frame_times) < (1.0 / fps)
    }
    
    return stats


def main():
    """Main function for real-time video detection"""
    
    print("=" * 70)
    print("Real-Time Video Object Detection")
    print("=" * 70)
    
    # Create synthetic video
    print("\nGenerating synthetic video...")
    frames, fps = create_synthetic_video(num_frames=60, fps=30)
    print(f"   ✓ Created {len(frames)} frames at {fps} FPS")
    
    # Test different detection methods
    methods = {
        'color': 'Color-Based Detection',
        'motion': 'Motion-Based Detection (Background Subtraction)'
    }
    
    all_results = {}
    all_stats = {}
    
    for method, description in methods.items():
        print(f"\n{description}:")
        print("-" * 70)
        
        detector = RealTimeDetector(detection_method=method)
        print(f"   Initializing {method} detector...")
        
        # Process video
        print("   Processing video frames...")
        results, frame_times = process_video_realtime(frames, detector, fps)
        
        # Analyze performance
        stats = analyze_performance(frame_times, fps)
        all_stats[method] = stats
        all_results[method] = results
        
        print(f"\n   Performance Statistics:")
        print(f"   - Mean processing time: {stats['mean_time']:.2f} ms")
        print(f"   - Std processing time: {stats['std_time']:.2f} ms")
        print(f"   - Min/Max time: {stats['min_time']:.2f} / {stats['max_time']:.2f} ms")
        print(f"   - Average FPS: {stats['mean_fps']:.1f}")
        print(f"   - Real-time capable ({fps} FPS): {'✓ Yes' if stats['realtime_capable'] else '✗ No'}")
    
    # Visualize results
    print("\n" + "=" * 70)
    print("Visualization")
    print("=" * 70)
    
    # Show sample frames
    print("\n1. Comparing detection methods...")
    
    sample_indices = [10, 25, 40, 55]
    
    fig = plt.figure(figsize=(18, 8))
    
    for row, (method, description) in enumerate(methods.items()):
        for col, idx in enumerate(sample_indices):
            plt.subplot(2, 4, row * 4 + col + 1)
            plt.imshow(cv2.cvtColor(all_results[method][idx], cv2.COLOR_BGR2RGB))
            if col == 0:
                plt.title(f'{description}\nFrame {idx}')
            else:
                plt.title(f'Frame {idx}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('realtime_detection_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("   ✓ Method comparison saved")
    
    # Performance comparison
    print("\n2. Performance analysis...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Processing time comparison
    methods_list = list(methods.keys())
    mean_times = [all_stats[m]['mean_time'] for m in methods_list]
    std_times = [all_stats[m]['std_time'] for m in methods_list]
    
    axes[0].bar(range(len(methods_list)), mean_times, yerr=std_times, 
               capsize=5, color=['blue', 'green'])
    axes[0].axhline(y=1000/fps, color='r', linestyle='--', label=f'Real-time threshold ({fps} FPS)')
    axes[0].set_xticks(range(len(methods_list)))
    axes[0].set_xticklabels([methods[m] for m in methods_list], rotation=15, ha='right')
    axes[0].set_ylabel('Processing Time (ms)')
    axes[0].set_title('Detection Speed Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # FPS comparison
    fps_values = [all_stats[m]['mean_fps'] for m in methods_list]
    
    axes[1].bar(range(len(methods_list)), fps_values, color=['blue', 'green'])
    axes[1].axhline(y=fps, color='r', linestyle='--', label=f'Target FPS ({fps})')
    axes[1].set_xticks(range(len(methods_list)))
    axes[1].set_xticklabels([methods[m] for m in methods_list], rotation=15, ha='right')
    axes[1].set_ylabel('Frames Per Second')
    axes[1].set_title('Throughput Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('realtime_performance_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("   ✓ Performance analysis saved")
    
    # Summary
    print("\n" + "=" * 70)
    print("Real-Time Detection Summary")
    print("=" * 70)
    
    print("\nKey Performance Factors:")
    print("  • Detection algorithm complexity")
    print("  • Image resolution")
    print("  • Number of objects in scene")
    print("  • Hardware capabilities (CPU/GPU)")
    
    print("\nOptimization Strategies:")
    print("  ✓ Use GPU acceleration (CUDA, OpenCL)")
    print("  ✓ Reduce input resolution")
    print("  ✓ Skip frames (process every Nth frame)")
    print("  ✓ Use faster detection algorithms")
    print("  ✓ Implement multi-threading")
    print("  ✓ Use hardware acceleration (Intel OpenVINO, NVIDIA TensorRT)")
    
    print("\nApplications:")
    print("  • Live surveillance monitoring")
    print("  • Autonomous vehicle perception")
    print("  • Sports analytics")
    print("  • Retail customer tracking")
    print("  • Industrial quality control")
    print("  • Augmented reality")
    
    print("\nFor Production Deployment:")
    print("  • Use pre-trained models (YOLO, SSD, EfficientDet)")
    print("  • Implement video streaming protocols (RTSP, WebRTC)")
    print("  • Add frame buffering and queue management")
    print("  • Monitor system resources (CPU, memory, GPU)")
    print("  • Implement automatic quality degradation under load")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
