"""
Real-Time Surveillance System
Comprehensive surveillance system with motion detection, person tracking, and event logging
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime
import time

class SurveillanceSystem:
    """
    Complete surveillance system with multiple detection and tracking capabilities
    """
    
    def __init__(self, sensitivity='medium'):
        """
        Initialize surveillance system
        
        Args:
            sensitivity: 'low', 'medium', or 'high' detection sensitivity
        """
        self.sensitivity = sensitivity
        self.setup_parameters()
        
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=self.bg_threshold, 
            detectShadows=True
        )
        
        # Cascade classifiers
        self.person_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_fullbody.xml'
        )
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Event tracking
        self.events = []
        self.active_tracks = {}
        self.next_track_id = 0
        
        # Performance monitoring
        self.fps_history = deque(maxlen=30)
        self.detection_count = 0
        
        # Motion history
        self.motion_history = deque(maxlen=100)
        
    def setup_parameters(self):
        """Setup detection parameters based on sensitivity"""
        sensitivity_params = {
            'low': {'bg_threshold': 50, 'min_area': 2000, 'motion_threshold': 0.05},
            'medium': {'bg_threshold': 30, 'min_area': 1000, 'motion_threshold': 0.03},
            'high': {'bg_threshold': 16, 'min_area': 500, 'motion_threshold': 0.01}
        }
        
        params = sensitivity_params.get(self.sensitivity, sensitivity_params['medium'])
        self.bg_threshold = params['bg_threshold']
        self.min_area = params['min_area']
        self.motion_threshold = params['motion_threshold']
    
    def detect_motion(self, frame):
        """
        Detect motion using background subtraction
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows (value 127)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Calculate motion percentage
        motion_pixels = np.sum(fg_mask > 0)
        total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
        motion_percentage = motion_pixels / total_pixels
        
        self.motion_history.append(motion_percentage)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    'bbox': (x, y, w, h),
                    'type': 'motion',
                    'confidence': min(area / (self.min_area * 10), 1.0),
                    'area': area
                })
        
        return detections, fg_mask, motion_percentage
    
    def detect_persons(self, frame):
        """
        Detect persons using Haar cascades
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect full body
        persons = self.person_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 90)
        )
        
        detections = []
        for (x, y, w, h) in persons:
            detections.append({
                'bbox': (x, y, w, h),
                'type': 'person',
                'confidence': 1.0
            })
        
        return detections
    
    def detect_faces(self, frame):
        """
        Detect faces for person identification
        """
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
                'type': 'face',
                'confidence': 1.0
            })
        
        return detections
    
    def track_objects(self, current_detections, timestamp):
        """
        Track detected objects across frames
        """
        max_distance = 100  # Maximum distance for matching
        
        # Match current detections to existing tracks
        matched_tracks = set()
        new_detections = []
        
        for detection in current_detections:
            current_bbox = detection['bbox']
            current_center = (
                current_bbox[0] + current_bbox[2] // 2,
                current_bbox[1] + current_bbox[3] // 2
            )
            
            best_match = None
            best_distance = max_distance
            
            # Find closest existing track
            for track_id, track_info in self.active_tracks.items():
                if track_id in matched_tracks:
                    continue
                
                last_center = track_info['center']
                distance = np.sqrt(
                    (current_center[0] - last_center[0]) ** 2 +
                    (current_center[1] - last_center[1]) ** 2
                )
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = track_id
            
            if best_match is not None:
                # Update existing track
                self.active_tracks[best_match]['center'] = current_center
                self.active_tracks[best_match]['bbox'] = current_bbox
                self.active_tracks[best_match]['last_seen'] = timestamp
                self.active_tracks[best_match]['frames_tracked'] += 1
                matched_tracks.add(best_match)
                detection['track_id'] = best_match
            else:
                # Create new track
                track_id = self.next_track_id
                self.next_track_id += 1
                
                self.active_tracks[track_id] = {
                    'center': current_center,
                    'bbox': current_bbox,
                    'first_seen': timestamp,
                    'last_seen': timestamp,
                    'frames_tracked': 1,
                    'type': detection['type']
                }
                
                detection['track_id'] = track_id
                
                # Log event
                self.log_event('new_detection', track_id, detection['type'], timestamp)
        
        # Remove stale tracks (not seen for 30 frames)
        stale_tracks = [
            tid for tid, info in self.active_tracks.items()
            if (timestamp - info['last_seen']) > 30
        ]
        
        for tid in stale_tracks:
            self.log_event('track_lost', tid, 
                          self.active_tracks[tid]['type'], timestamp)
            del self.active_tracks[tid]
        
        return current_detections
    
    def log_event(self, event_type, track_id, object_type, timestamp):
        """
        Log surveillance events
        """
        event = {
            'type': event_type,
            'track_id': track_id,
            'object_type': object_type,
            'timestamp': timestamp,
            'time_str': datetime.now().strftime('%H:%M:%S')
        }
        self.events.append(event)
        
        if event_type == 'new_detection':
            self.detection_count += 1
    
    def check_intrusion_zones(self, detections, zones):
        """
        Check if detections are in restricted zones
        """
        intrusions = []
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            center = (x + w // 2, y + h // 2)
            
            for zone_name, zone_polygon in zones.items():
                if cv2.pointPolygonTest(zone_polygon, center, False) >= 0:
                    intrusions.append({
                        'zone': zone_name,
                        'detection': detection,
                        'severity': 'high'
                    })
        
        return intrusions
    
    def draw_surveillance_overlay(self, frame, detections, motion_mask, zones=None):
        """
        Draw surveillance information overlay
        """
        result = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw motion heatmap
        motion_colored = cv2.applyColorMap(motion_mask, cv2.COLORMAP_JET)
        result = cv2.addWeighted(result, 0.7, motion_colored, 0.3, 0)
        
        # Draw zones
        if zones:
            for zone_name, zone_polygon in zones.items():
                cv2.polylines(result, [zone_polygon], True, (255, 255, 0), 2)
                
                # Label zone
                M = cv2.moments(zone_polygon)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.putText(result, zone_name, (cx - 30, cy), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw detections with tracks
        for detection in detections:
            x, y, w, h = detection['bbox']
            track_id = detection.get('track_id', -1)
            obj_type = detection['type']
            
            # Color by type
            if obj_type == 'person':
                color = (0, 255, 0)
            elif obj_type == 'face':
                color = (255, 255, 0)
            else:
                color = (0, 255, 255)
            
            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            if track_id >= 0:
                label = f"ID:{track_id} {obj_type}"
            else:
                label = obj_type
            
            cv2.putText(result, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw status panel
        panel_h = 150
        panel = np.zeros((panel_h, w, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)
        
        # System status
        cv2.putText(panel, f"SURVEILLANCE SYSTEM - {datetime.now().strftime('%H:%M:%S')}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Stats
        if len(self.fps_history) > 0:
            fps = np.mean(self.fps_history)
            cv2.putText(panel, f"FPS: {fps:.1f}", (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.putText(panel, f"Active Tracks: {len(self.active_tracks)}", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.putText(panel, f"Total Detections: {self.detection_count}", (10, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Motion indicator
        if len(self.motion_history) > 0:
            motion_level = np.mean(list(self.motion_history)[-10:])
            motion_text = "HIGH" if motion_level > 0.05 else "LOW"
            motion_color = (0, 0, 255) if motion_level > 0.05 else (0, 255, 0)
            cv2.putText(panel, f"Motion: {motion_text}", (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, motion_color, 1)
        
        # Recent events (right side)
        recent_events = self.events[-5:]
        for i, event in enumerate(reversed(recent_events)):
            event_text = f"{event['time_str']} - {event['type']} (ID:{event['track_id']})"
            cv2.putText(panel, event_text, (w - 400, 25 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Combine frame with panel
        result = np.vstack([result, panel])
        
        return result


def create_surveillance_video(num_frames=100):
    """Create synthetic surveillance video"""
    width, height = 640, 480
    frames = []
    
    for i in range(num_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # Add textured background
        for y in range(height):
            for x in range(width):
                noise = np.random.randint(-20, 20)
                frame[y, x] = np.clip(frame[y, x] + noise, 0, 255)
        
        # Moving person (rectangle simulation)
        person_x = int(50 + (i * 3) % (width - 150))
        person_y = 200
        cv2.rectangle(frame, (person_x, person_y), 
                     (person_x + 40, person_y + 100), (100, 100, 150), -1)
        
        # Another person (different path)
        if i > 30:
            person2_x = int(width - 100 - ((i - 30) * 2) % (width - 200))
            person2_y = 300
            cv2.rectangle(frame, (person2_x, person2_y), 
                         (person2_x + 35, person2_y + 90), (120, 120, 160), -1)
        
        # Static object (should be learned as background)
        cv2.rectangle(frame, (500, 50), (600, 150), (80, 80, 80), -1)
        
        frames.append(frame)
    
    return frames


def main():
    """Main surveillance system demonstration"""
    
    print("=" * 70)
    print("Real-Time Surveillance System")
    print("=" * 70)
    
    # Create synthetic surveillance video
    print("\nGenerating surveillance footage...")
    frames = create_surveillance_video(num_frames=80)
    print(f"   ✓ Created {len(frames)} frames")
    
    # Define restricted zones
    zones = {
        'Zone_A': np.array([[100, 100], [250, 100], [250, 250], [100, 250]], np.int32)
    }
    
    # Test different sensitivity levels
    sensitivity_levels = ['low', 'medium', 'high']
    all_results = {}
    
    for sensitivity in sensitivity_levels:
        print(f"\n{'='*70}")
        print(f"Testing {sensitivity.upper()} sensitivity")
        print(f"{'='*70}")
        
        system = SurveillanceSystem(sensitivity=sensitivity)
        results = []
        
        print("   Processing frames...")
        for frame_idx, frame in enumerate(frames):
            start_time = time.time()
            
            # Detect motion
            motion_detections, motion_mask, motion_pct = system.detect_motion(frame)
            
            # Track objects
            tracked_detections = system.track_objects(motion_detections, frame_idx)
            
            # Check intrusion zones
            intrusions = system.check_intrusion_zones(tracked_detections, zones)
            
            # Draw overlay
            result = system.draw_surveillance_overlay(
                frame, tracked_detections, motion_mask, zones
            )
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0
            system.fps_history.append(fps)
            
            results.append(result)
            
            if (frame_idx + 1) % 20 == 0:
                print(f"      Frame {frame_idx + 1}/{len(frames)} "
                      f"(FPS: {np.mean(list(system.fps_history)[-10:]):.1f})")
        
        print(f"\n   Summary:")
        print(f"   - Total detections: {system.detection_count}")
        print(f"   - Total events: {len(system.events)}")
        print(f"   - Average FPS: {np.mean(system.fps_history):.1f}")
        
        all_results[sensitivity] = {
            'frames': results,
            'system': system
        }
    
    # Visualize comparison
    print("\n" + "=" * 70)
    print("Visualization")
    print("=" * 70)
    
    sample_frame = 40
    
    fig = plt.figure(figsize=(18, 6))
    
    for idx, sensitivity in enumerate(sensitivity_levels):
        plt.subplot(1, 3, idx + 1)
        result_frame = all_results[sensitivity]['frames'][sample_frame]
        plt.imshow(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
        plt.title(f'{sensitivity.upper()} Sensitivity\n'
                 f'Detections: {all_results[sensitivity]["system"].detection_count}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('surveillance_sensitivity_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("   ✓ Comparison visualization saved")
    
    # Event timeline
    print("\n" + "=" * 70)
    print("Event Analysis")
    print("=" * 70)
    
    for sensitivity in sensitivity_levels:
        system = all_results[sensitivity]['system']
        print(f"\n{sensitivity.upper()} Sensitivity Events:")
        for event in system.events[:10]:  # Show first 10 events
            print(f"   [{event['time_str']}] {event['type']}: "
                  f"ID {event['track_id']} ({event['object_type']})")
    
    # Summary
    print("\n" + "=" * 70)
    print("Surveillance System Summary")
    print("=" * 70)
    
    print("\nKey Features:")
    print("  ✓ Real-time motion detection")
    print("  ✓ Object tracking with unique IDs")
    print("  ✓ Restricted zone monitoring")
    print("  ✓ Event logging and alerting")
    print("  ✓ Performance monitoring (FPS)")
    print("  ✓ Configurable sensitivity levels")
    
    print("\nApplications:")
    print("  • Perimeter security")
    print("  • Retail store monitoring")
    print("  • Parking lot surveillance")
    print("  • Traffic monitoring")
    print("  • Wildlife observation")
    print("  • Industrial safety monitoring")
    
    print("\nAdvanced Features (Production):")
    print("  • Multi-camera synchronization")
    print("  • Cloud storage integration")
    print("  • Mobile alerts and notifications")
    print("  • AI-powered anomaly detection")
    print("  • Facial recognition integration")
    print("  • License plate recognition")
    print("  • Heat mapping and analytics")
    print("  • Video compression and archival")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
