"""
Feature Detection and Matching - The Foundation of Computer Vision

This project was fascinating to build! Feature detection is how computers "see" and 
understand images. Instead of looking at every pixel, we identify interesting points
(corners, edges, blobs) that are distinctive and can be reliably found even when the
image is rotated, scaled, or viewed from a different angle.

I implemented multiple algorithms (SIFT, ORB, AKAZE, BRISK) to understand their
trade-offs. SIFT is the gold standard for accuracy, ORB is lightning-fast, AKAZE
balances both, and BRISK uses binary descriptors for speed.

This is used everywhere: panorama stitching, AR apps, 3D reconstruction, object tracking...
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_test_images():
    """
    Create two test images to demonstrate feature matching
    
    The second image is a rotated and scaled version of the first. This is perfect
    for testing because if our feature detector is truly scale and rotation invariant,
    it should find matching features between these images despite the transformation.
    
    In real applications, these might be two photos of the same scene from different
    angles, or consecutive frames in a video.
    """
    # Image 1 - Original
    img1 = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Add various shapes with different features
    cv2.circle(img1, (100, 100), 50, (0, 0, 0), 2)
    cv2.circle(img1, (100, 100), 30, (0, 0, 0), 2)
    cv2.rectangle(img1, (200, 50), (350, 150), (0, 0, 0), 2)
    cv2.line(img1, (400, 50), (550, 150), (0, 0, 0), 2)
    cv2.ellipse(img1, (450, 250), (80, 40), 30, 0, 360, (0, 0, 0), 2)
    
    # Add text
    cv2.putText(img1, "FEATURES", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Image 2 - Rotated and slightly scaled version
    M = cv2.getRotationMatrix2D((300, 200), 20, 1.1)
    img2 = cv2.warpAffine(img1, M, (600, 400), borderValue=(255, 255, 255))
    
    return img1, img2

def detect_and_compute_sift(img):
    """
    SIFT (Scale-Invariant Feature Transform) - The Classic
    
    SIFT was revolutionary when it came out (1999)! It finds features that remain
    stable across different scales, rotations, and lighting conditions. The key
    innovation was building a scale-space pyramid and finding local extrema.
    
    Pros: Excellent accuracy and invariance
    Cons: Patented (though now expired), slower than modern alternatives
    
    Use when: Accuracy matters more than speed, or when dealing with significant
    scale/rotation changes between images.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create SIFT detector
    sift = cv2.SIFT_create(nfeatures=500)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    return keypoints, descriptors, "SIFT"

def detect_and_compute_orb(img):
    """
    ORB (Oriented FAST and Rotated BRIEF) - The Speed Demon
    
    ORB was designed as a free, fast alternative to SIFT and SURF. It combines
    FAST keypoint detection with BRIEF descriptors, adding orientation computation
    to make it rotation-invariant.
    
    Pros: Very fast, no patents, good enough for real-time applications
    Cons: Less accurate than SIFT, binary descriptors are less distinctive
    
    Use when: Speed matters (mobile apps, real-time tracking), or when you want
    a patent-free solution that's good enough for most tasks.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=500)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    return keypoints, descriptors, "ORB"

def detect_and_compute_akaze(img):
    """
    AKAZE (Accelerated-KAZE) - The Balanced Approach
    
    AKAZE uses nonlinear diffusion filtering to build scale space, which better
    preserves boundaries compared to Gaussian pyramids. It's faster than KAZE
    while maintaining good quality.
    
    Pros: Good balance of speed and accuracy, better edge localization
    Cons: Slower than ORB, not as widely adopted as SIFT
    
    Use when: You need better accuracy than ORB but faster than SIFT, especially
    when precise edge localization matters.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create AKAZE detector
    akaze = cv2.AKAZE_create()
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = akaze.detectAndCompute(gray, None)
    
    return keypoints, descriptors, "AKAZE"

def detect_and_compute_brisk(img):
    """
    BRISK (Binary Robust Invariant Scalable Keypoints) detection
    - Binary descriptor for fast matching
    - Scale and rotation invariant
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create BRISK detector
    brisk = cv2.BRISK_create()
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = brisk.detectAndCompute(gray, None)
    
    return keypoints, descriptors, "BRISK"

def match_features(desc1, desc2, detector_type, ratio_thresh=0.75):
    """
    Match features between two images using different matching strategies
    
    Args:
        desc1, desc2: Descriptors from two images
        detector_type: Type of detector used
        ratio_thresh: Ratio test threshold for Lowe's ratio test
        
    Returns:
        List of good matches
    """
    if detector_type in ["SIFT", "AKAZE"]:
        # Use BFMatcher with L2 norm for SIFT
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
    else:
        # Use Hamming distance for binary descriptors (ORB, BRISK)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        
        # Sort by distance (lower is better)
        good_matches = sorted(matches, key=lambda x: x.distance)[:50]
    
    return good_matches

def draw_keypoints_comparison(img1, img2, detectors):
    """Draw keypoints detected by different algorithms"""
    fig, axes = plt.subplots(len(detectors), 2, figsize=(14, 4 * len(detectors)))
    
    for idx, (detector_func, name) in enumerate(detectors):
        # Image 1
        kp1, _, _ = detector_func(img1)
        img1_kp = cv2.drawKeypoints(img1, kp1, None, 
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                     color=(0, 255, 0))
        
        # Image 2
        kp2, _, _ = detector_func(img2)
        img2_kp = cv2.drawKeypoints(img2, kp2, None,
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                     color=(0, 255, 0))
        
        # Plot
        axes[idx, 0].imshow(cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB))
        axes[idx, 0].set_title(f'{name} - Image 1 ({len(kp1)} keypoints)')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(cv2.cvtColor(img2_kp, cv2.COLOR_BGR2RGB))
        axes[idx, 1].set_title(f'{name} - Image 2 ({len(kp2)} keypoints)')
        axes[idx, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('feature_detection_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def visualize_matches(img1, img2, detector_func):
    """Visualize feature matches between two images"""
    # Detect features
    kp1, desc1, name = detector_func(img1)
    kp2, desc2, name = detector_func(img2)
    
    # Match features
    matches = match_features(desc1, desc2, name)
    
    # Draw matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return img_matches, len(matches), name

def compute_homography(img1, img2, detector_func):
    """
    Compute homography matrix between two images
    Demonstrates geometric verification of matches
    """
    # Detect and match features
    kp1, desc1, name = detector_func(img1)
    kp2, desc2, name = detector_func(img2)
    matches = match_features(desc1, desc2, name)
    
    if len(matches) < 4:
        return None, None, name
    
    # Extract location of good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Find homography using RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # Count inliers
    inliers = mask.ravel().sum()
    
    return H, inliers, name

def main():
    """Main function demonstrating all feature detection and matching techniques"""
    print("=" * 70)
    print("Feature Detection and Matching Demonstration")
    print("=" * 70)
    
    # Create test images
    print("\nCreating test images...")
    img1, img2 = create_test_images()
    print("   ✓ Test images created (original and transformed)")
    
    # Define detectors
    detectors = [
        (detect_and_compute_sift, "SIFT"),
        (detect_and_compute_orb, "ORB"),
        (detect_and_compute_akaze, "AKAZE"),
        (detect_and_compute_brisk, "BRISK")
    ]
    
    # 1. Compare keypoint detection
    print("\n1. Comparing Feature Detectors...")
    draw_keypoints_comparison(img1, img2, detectors)
    print("   ✓ Keypoint comparison complete")
    
    # 2. Feature matching visualization
    print("\n2. Feature Matching Visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.ravel()
    
    for idx, (detector_func, _) in enumerate(detectors):
        img_matches, num_matches, name = visualize_matches(img1, img2, detector_func)
        
        axes[idx].imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(f'{name} Matching ({num_matches} matches)')
        axes[idx].axis('off')
        
        print(f"   - {name}: {num_matches} matches found")
    
    plt.tight_layout()
    plt.savefig('feature_matching_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("   ✓ Feature matching complete")
    
    # 3. Homography estimation
    print("\n3. Homography Estimation (Geometric Verification)...")
    for detector_func, _ in detectors:
        H, inliers, name = compute_homography(img1, img2, detector_func)
        if H is not None:
            print(f"   - {name}: {inliers} inliers (geometrically consistent matches)")
        else:
            print(f"   - {name}: Not enough matches for homography")
    
    # 4. Performance comparison
    print("\n4. Performance Analysis...")
    print("\n   Algorithm Characteristics:")
    print("   " + "-" * 60)
    print("   SIFT:  Scale-invariant, robust, patented")
    print("   ORB:   Fast, free, binary descriptor")
    print("   AKAZE: Good speed/accuracy balance, nonlinear scale space")
    print("   BRISK: Binary, fast matching, scale and rotation invariant")
    print("   " + "-" * 60)
    
    print("\n" + "=" * 70)
    print("All feature detection demonstrations completed successfully!")
    print("Results saved as PNG files.")
    print("=" * 70)
    
    # Additional: Corner detection (Harris and Shi-Tomasi)
    print("\nBonus: Corner Detection...")
    demonstrate_corner_detection(img1)

def demonstrate_corner_detection(img):
    """Demonstrate Harris and Shi-Tomasi corner detection"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    # Harris corner detection
    harris = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)
    
    img_harris = img.copy()
    img_harris[harris > 0.01 * harris.max()] = [0, 0, 255]
    
    # Shi-Tomasi corner detection
    corners = cv2.goodFeaturesToTrack(np.uint8(gray), maxCorners=100, 
                                       qualityLevel=0.01, minDistance=10)
    
    img_shitomasi = img.copy()
    if corners is not None:
        corners = corners.astype(np.int32)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(img_shitomasi, (x, y), 5, (0, 0, 255), -1)
    
    # Visualize
    plt.figure(figsize=(12, 5))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(img_harris, cv2.COLOR_BGR2RGB))
    plt.title('Harris Corners')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(img_shitomasi, cv2.COLOR_BGR2RGB))
    plt.title('Shi-Tomasi Corners')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('corner_detection.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("   ✓ Corner detection complete")

if __name__ == "__main__":
    main()
