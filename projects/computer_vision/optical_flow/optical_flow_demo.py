"""
Optical Flow Visualization
Implements dense and sparse optical flow algorithms for motion estimation
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_moving_shapes_video(num_frames=50):
    """
    Create synthetic video with moving shapes for optical flow demonstration
    
    Returns:
        List of frames
    """
    frames = []
    width, height = 640, 480
    
    for i in range(num_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Moving circle (left to right)
        circle_x = int(50 + i * 10)
        circle_y = 100
        cv2.circle(frame, (circle_x, circle_y), 30, (255, 0, 0), -1)
        
        # Moving rectangle (right to left)
        rect_x = int(550 - i * 8)
        rect_y = 250
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + 60, rect_y + 40), 
                     (0, 255, 0), -1)
        
        # Rotating triangle
        angle = i * 7
        center = (320, 400)
        size = 40
        pts = np.array([
            [center[0], center[1] - size],
            [center[0] - size, center[1] + size],
            [center[0] + size, center[1] + size]
        ], np.int32)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        pts = cv2.transform(np.array([pts]), M)[0]
        cv2.fillPoly(frame, [pts], (0, 0, 255))
        
        frames.append(frame)
    
    return frames

def lucas_kanade_optical_flow(frames):
    """
    Sparse optical flow using Lucas-Kanade method
    Tracks individual feature points across frames
    
    Args:
        frames: List of video frames
        
    Returns:
        List of frames with optical flow visualization
    """
    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )
    
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    # Create random colors for trajectories
    color = np.random.randint(0, 255, (100, 3))
    
    # Take first frame and find corners
    old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    
    # Create mask for drawing
    mask = np.zeros_like(frames[0])
    
    result_frames = []
    
    for frame in frames[1:]:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, 
                                                **lk_params)
        
        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
        
        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            a, b, c, d = int(a), int(b), int(c), int(d)
            
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        
        img = cv2.add(frame, mask)
        result_frames.append(img.copy())
        
        # Update previous frame and points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    
    return result_frames

def dense_optical_flow_farneback(frames):
    """
    Dense optical flow using Farneback method
    Computes flow for every pixel in the image
    
    Args:
        frames: List of video frames
        
    Returns:
        List of frames with dense optical flow visualization
    """
    result_frames = []
    
    old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    
    # Create HSV mask for visualization
    hsv = np.zeros_like(frames[0])
    hsv[..., 1] = 255
    
    for frame in frames[1:]:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            old_gray, frame_gray, None,
            pyr_scale=0.5,      # Pyramid scale
            levels=3,           # Number of pyramid layers
            winsize=15,         # Window size
            iterations=3,       # Iterations at each pyramid level
            poly_n=5,           # Neighborhood size for polynomial expansion
            poly_sigma=1.2,     # Gaussian standard deviation
            flags=0
        )
        
        # Convert flow to polar coordinates (magnitude and angle)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Encode angle as hue, magnitude as value
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert HSV to RGB for display
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        result_frames.append(rgb.copy())
        old_gray = frame_gray
    
    return result_frames

def dense_optical_flow_visualization(frames):
    """
    Visualize dense optical flow with arrows and magnitude
    """
    result_frames = []
    
    old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    
    for frame in frames[1:]:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            old_gray, frame_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Draw flow field with arrows
        step = 20
        img_with_arrows = frame.copy()
        
        for y in range(0, flow.shape[0], step):
            for x in range(0, flow.shape[1], step):
                fx, fy = flow[y, x]
                
                # Only draw significant motion
                magnitude = np.sqrt(fx**2 + fy**2)
                if magnitude > 1.0:
                    # Draw arrow
                    end_point = (int(x + fx), int(y + fy))
                    cv2.arrowedLine(img_with_arrows, (x, y), end_point,
                                   (0, 255, 0), 1, tipLength=0.3)
        
        result_frames.append(img_with_arrows)
        old_gray = frame_gray
    
    return result_frames

def compute_flow_statistics(frames):
    """Compute and display optical flow statistics"""
    old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    
    magnitudes = []
    directions = []
    
    for frame in frames[1:]:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            old_gray, frame_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        magnitudes.append(np.mean(mag))
        directions.append(np.mean(ang))
        
        old_gray = frame_gray
    
    return magnitudes, directions

def main():
    """Main function demonstrating optical flow techniques"""
    print("=" * 70)
    print("Optical Flow Visualization")
    print("=" * 70)
    
    # Create synthetic video
    print("\nGenerating synthetic video with moving objects...")
    frames = create_moving_shapes_video(num_frames=30)
    print(f"   ✓ Created {len(frames)} frames")
    
    # 1. Lucas-Kanade Sparse Optical Flow
    print("\n1. Computing Lucas-Kanade Sparse Optical Flow...")
    lk_frames = lucas_kanade_optical_flow(frames)
    print(f"   ✓ Processed {len(lk_frames)} frames")
    print("   - Tracks individual feature points")
    print("   - Shows motion trajectories")
    
    # 2. Farneback Dense Optical Flow
    print("\n2. Computing Farneback Dense Optical Flow...")
    dense_frames = dense_optical_flow_farneback(frames)
    print(f"   ✓ Processed {len(dense_frames)} frames")
    print("   - Computes flow for every pixel")
    print("   - Color indicates direction, brightness indicates magnitude")
    
    # 3. Arrow Visualization
    print("\n3. Computing Flow Field with Arrows...")
    arrow_frames = dense_optical_flow_visualization(frames)
    print(f"   ✓ Processed {len(arrow_frames)} frames")
    print("   - Arrows show direction and magnitude of motion")
    
    # 4. Flow Statistics
    print("\n4. Computing Flow Statistics...")
    magnitudes, directions = compute_flow_statistics(frames)
    print(f"   ✓ Average flow magnitude: {np.mean(magnitudes):.2f} pixels")
    
    # Visualize results
    print("\n5. Visualizing Results...")
    
    # Show comparison at frame 15
    frame_idx = 15
    
    fig = plt.figure(figsize=(16, 10))
    
    # Original frame
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB))
    plt.title('Original Frame')
    plt.axis('off')
    
    # Lucas-Kanade
    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(lk_frames[frame_idx-1], cv2.COLOR_BGR2RGB))
    plt.title('Lucas-Kanade (Sparse)')
    plt.axis('off')
    
    # Dense flow
    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(dense_frames[frame_idx-1], cv2.COLOR_BGR2RGB))
    plt.title('Farneback (Dense)')
    plt.axis('off')
    
    # Arrows
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(arrow_frames[frame_idx-1], cv2.COLOR_BGR2RGB))
    plt.title('Flow Vectors')
    plt.axis('off')
    
    # Flow magnitude over time
    plt.subplot(2, 3, 5)
    plt.plot(magnitudes, 'b-', linewidth=2)
    plt.xlabel('Frame Number')
    plt.ylabel('Average Flow Magnitude')
    plt.title('Motion Over Time')
    plt.grid(True, alpha=0.3)
    
    # Flow direction histogram
    plt.subplot(2, 3, 6)
    plt.hist(directions, bins=30, color='green', alpha=0.7, edgecolor='black')
    plt.xlabel('Flow Direction (radians)')
    plt.ylabel('Frequency')
    plt.title('Direction Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optical_flow_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("   ✓ Visualization complete")
    
    # Create comparison video frames
    print("\n6. Saving Comparison Frames...")
    comparison_frames = []
    for i in range(min(len(lk_frames), len(dense_frames), len(arrow_frames))):
        # Create side-by-side comparison
        row1 = np.hstack([frames[i+1], lk_frames[i]])
        row2 = np.hstack([dense_frames[i], arrow_frames[i]])
        comparison = np.vstack([row1, row2])
        comparison_frames.append(comparison)
    
    # Save middle frame as example
    cv2.imwrite('optical_flow_comparison.png', comparison_frames[15])
    print("   ✓ Comparison frame saved")
    
    print("\n" + "=" * 70)
    print("Optical Flow Analysis Summary")
    print("=" * 70)
    print(f"\nLucas-Kanade (Sparse):")
    print("  + Fast and efficient")
    print("  + Good for tracking specific points")
    print("  - Limited to sparse features")
    
    print(f"\nFarneback (Dense):")
    print("  + Computes flow for entire image")
    print("  + Good for motion analysis")
    print("  - Computationally expensive")
    
    print(f"\nApplications:")
    print("  • Video stabilization")
    print("  • Motion detection")
    print("  • Action recognition")
    print("  • Object tracking")
    print("  • Visual odometry")
    print("  • Video compression")
    
    print("\n" + "=" * 70)
    print("Optical flow demonstration completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
