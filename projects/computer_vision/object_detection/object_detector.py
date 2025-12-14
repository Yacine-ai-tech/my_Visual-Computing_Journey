"""
Object Detection and Tracking
Demonstrates multiple object detection techniques and tracking algorithms
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_test_video_with_objects(num_frames=50):
    """Create synthetic video with moving objects for detection and tracking"""
    frames = []
    width, height = 640, 480
    
    for i in range(num_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Background gradient
        for y in range(height):
            frame[y, :] = [200 + y * 50 // height, 220 + y * 30 // height, 240]
        
        # Moving red ball
        ball_x = int(50 + i * 10)
        ball_y = int(100 + 30 * np.sin(i * 0.3))
        cv2.circle(frame, (ball_x % width, ball_y), 25, (0, 0, 255), -1)
        
        # Moving green car (rectangle)
        car_x = int(width - 50 - i * 8)
        car_y = 250
        cv2.rectangle(frame, (car_x % width, car_y), 
                     ((car_x + 80) % width, car_y + 40), (0, 255, 0), -1)
        cv2.rectangle(frame, (car_x % width, car_y + 40), 
                     ((car_x + 80) % width, car_y + 50), (50, 50, 50), -1)
        
        # Stationary blue box
        cv2.rectangle(frame, (450, 350), (550, 430), (255, 0, 0), -1)
        
        frames.append(frame)
    
    return frames

def color_based_detection(frame):
    """
    Detect objects based on color in HSV space
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color ranges
    # Red range
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # Green range
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Blue range
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    return mask_red, mask_green, mask_blue

def detect_and_draw_contours(frame, mask, color, label):
    """
    Find and draw bounding boxes around detected objects
    """
    result = frame.copy()
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter small contours
        if area > 500:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            cv2.putText(result, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Calculate center
            center = (x + w // 2, y + h // 2)
            cv2.circle(result, center, 5, color, -1)
            
            detections.append((x, y, w, h, center))
    
    return result, detections

def background_subtraction_detection(frames):
    """
    Detect moving objects using background subtraction
    """
    # Create background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=100, varThreshold=50, detectShadows=True
    )
    
    results = []
    
    for frame in frames:
        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)
        
        # Remove shadows (value 127)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw bounding boxes
        result = frame.copy()
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(result, "Moving", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        results.append(result)
    
    return results

def centroid_tracking(frames):
    """
    Simple centroid-based object tracking
    """
    # First frame - detect objects
    mask_red, mask_green, mask_blue = color_based_detection(frames[0])
    _, initial_detections = detect_and_draw_contours(
        frames[0], cv2.bitwise_or(mask_red, mask_green), (0, 255, 0), "Object"
    )
    
    # Initialize tracks
    tracks = {}
    next_id = 0
    
    for det in initial_detections:
        x, y, w, h, center = det
        tracks[next_id] = [center]
        next_id += 1
    
    results = []
    max_distance = 50  # Maximum distance for matching
    
    for frame in frames:
        result = frame.copy()
        
        # Detect objects in current frame
        mask_red, mask_green, mask_blue = color_based_detection(frame)
        combined_mask = cv2.bitwise_or(mask_red, mask_green)
        _, detections = detect_and_draw_contours(frame, combined_mask, 
                                                  (0, 255, 0), "")
        
        current_centers = [det[4] for det in detections]
        
        # Match detections to existing tracks
        matched_tracks = set()
        
        for center in current_centers:
            best_match = None
            best_distance = max_distance
            
            for track_id, track_history in tracks.items():
                if track_id in matched_tracks:
                    continue
                
                last_position = track_history[-1]
                distance = np.sqrt((center[0] - last_position[0])**2 + 
                                 (center[1] - last_position[1])**2)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = track_id
            
            if best_match is not None:
                tracks[best_match].append(center)
                matched_tracks.add(best_match)
            else:
                # New track
                tracks[next_id] = [center]
                next_id += 1
        
        # Draw tracks
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                 (255, 0, 255), (0, 255, 255)]
        
        for track_id, track_history in tracks.items():
            color = colors[track_id % len(colors)]
            
            # Draw trajectory
            for i in range(1, len(track_history)):
                if track_history[i - 1] is None or track_history[i] is None:
                    continue
                cv2.line(result, track_history[i - 1], track_history[i], 
                        color, 2)
            
            # Draw current position
            if len(track_history) > 0:
                cv2.circle(result, track_history[-1], 8, color, -1)
                cv2.putText(result, f"ID: {track_id}", 
                           (track_history[-1][0] + 10, track_history[-1][1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        results.append(result)
    
    return results

def template_matching_detection(frame, template):
    """
    Detect objects using template matching
    """
    result = frame.copy()
    
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Template matching
    methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF_NORMED]
    method_names = ["Correlation", "Cross-Correlation", "Square Difference"]
    
    results_list = []
    
    for method, name in zip(methods, method_names):
        result_img = frame.copy()
        match_result = cv2.matchTemplate(gray_frame, gray_template, method)
        
        # Find best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)
        
        # For SQDIFF, minimum is best; for others, maximum is best
        if method == cv2.TM_SQDIFF_NORMED:
            top_left = min_loc
            confidence = 1 - min_val
        else:
            top_left = max_loc
            confidence = max_val
        
        # Get template dimensions
        h, w = gray_template.shape
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        # Draw rectangle
        cv2.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(result_img, f"{name}: {confidence:.2f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        results_list.append(result_img)
    
    return results_list

def main():
    """Main function demonstrating object detection and tracking"""
    print("=" * 70)
    print("Object Detection and Tracking")
    print("=" * 70)
    
    # Create test video
    print("\nGenerating test video with moving objects...")
    frames = create_test_video_with_objects(num_frames=40)
    print(f"   ✓ Created {len(frames)} frames")
    
    # 1. Color-based Detection
    print("\n1. Color-Based Object Detection...")
    
    test_frame = frames[15]
    mask_red, mask_green, mask_blue = color_based_detection(test_frame)
    
    result_red, _ = detect_and_draw_contours(test_frame, mask_red, 
                                              (0, 0, 255), "Red Object")
    result_green, _ = detect_and_draw_contours(test_frame, mask_green, 
                                                (0, 255, 0), "Green Object")
    result_blue, _ = detect_and_draw_contours(test_frame, mask_blue, 
                                               (255, 0, 0), "Blue Object")
    
    # Combine all detections
    combined_mask = cv2.bitwise_or(mask_red, cv2.bitwise_or(mask_green, mask_blue))
    result_all, detections = detect_and_draw_contours(test_frame, combined_mask, 
                                                       (255, 255, 0), "Object")
    
    fig = plt.figure(figsize=(16, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB))
    plt.title('Original Frame')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(mask_red, cmap='gray')
    plt.title('Red Mask')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(mask_green, cmap='gray')
    plt.title('Green Mask')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(mask_blue, cmap='gray')
    plt.title('Blue Mask')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(combined_mask, cmap='gray')
    plt.title('Combined Mask')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(result_all, cv2.COLOR_BGR2RGB))
    plt.title(f'Detections ({len(detections)} objects)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('color_based_detection.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"   ✓ Detected {len(detections)} objects using color")
    
    # 2. Background Subtraction
    print("\n2. Background Subtraction (Motion Detection)...")
    
    bg_sub_results = background_subtraction_detection(frames)
    
    fig = plt.figure(figsize=(16, 4))
    
    for i, idx in enumerate([5, 15, 25, 35]):
        plt.subplot(1, 4, i + 1)
        plt.imshow(cv2.cvtColor(bg_sub_results[idx], cv2.COLOR_BGR2RGB))
        plt.title(f'Frame {idx}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('background_subtraction.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("   ✓ Motion detection complete")
    
    # 3. Object Tracking
    print("\n3. Centroid-Based Object Tracking...")
    
    tracking_results = centroid_tracking(frames)
    
    fig = plt.figure(figsize=(16, 4))
    
    for i, idx in enumerate([10, 20, 30, 39]):
        plt.subplot(1, 4, i + 1)
        plt.imshow(cv2.cvtColor(tracking_results[idx], cv2.COLOR_BGR2RGB))
        plt.title(f'Frame {idx} - Tracking')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('object_tracking.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("   ✓ Object tracking complete")
    
    # 4. Template Matching
    print("\n4. Template Matching...")
    
    # Create a template (crop from frame)
    template = frames[15][90:140, 140:190].copy()
    
    template_results = template_matching_detection(frames[20], template)
    
    fig = plt.figure(figsize=(16, 4))
    
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
    plt.title('Template')
    plt.axis('off')
    
    for i, result in enumerate(template_results):
        plt.subplot(1, 4, i + 2)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(f'Method {i + 1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('template_matching.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("   ✓ Template matching complete")
    
    # Summary
    print("\n" + "=" * 70)
    print("Detection & Tracking Methods Summary")
    print("=" * 70)
    print("\nColor-Based Detection:")
    print("  • Fast and simple")
    print("  • Works well in controlled lighting")
    print("  • Sensitive to illumination changes")
    print("  • Use case: Colored marker tracking")
    
    print("\nBackground Subtraction:")
    print("  • Detects moving objects")
    print("  • Requires static camera")
    print("  • Adapts to slow background changes")
    print("  • Use case: Surveillance, motion detection")
    
    print("\nCentroid Tracking:")
    print("  • Simple and fast")
    print("  • Good for well-separated objects")
    print("  • May fail with occlusions")
    print("  • Use case: Multiple object tracking")
    
    print("\nTemplate Matching:")
    print("  • Detects specific patterns")
    print("  • Not scale or rotation invariant")
    print("  • Fast for small templates")
    print("  • Use case: Logo detection, QR codes")
    
    print("\n" + "=" * 70)
    print("All object detection demonstrations completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
