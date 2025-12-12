"""
Geometric Transformations
Comprehensive implementation of 2D and perspective transformations
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_test_image():
    """Create a test image with grid and shapes"""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Draw grid
    for i in range(0, 600, 50):
        cv2.line(img, (i, 0), (i, 400), (200, 200, 200), 1)
    for i in range(0, 400, 50):
        cv2.line(img, (0, i), (600, i), (200, 200, 200), 1)
    
    # Draw shapes for reference
    cv2.circle(img, (150, 150), 50, (255, 0, 0), -1)
    cv2.rectangle(img, (350, 100), (500, 200), (0, 255, 0), -1)
    cv2.putText(img, "TRANSFORM", (200, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Draw corner markers
    for x, y in [(50, 50), (550, 50), (550, 350), (50, 350)]:
        cv2.circle(img, (x, y), 10, (0, 0, 0), -1)
    
    return img

def translation_transform(img, tx, ty):
    """
    Translation: shifts image by (tx, ty) pixels
    """
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(img, M, (cols, rows), borderValue=(255, 255, 255))
    return translated, M

def rotation_transform(img, angle, center=None, scale=1.0):
    """
    Rotation: rotates image around center by given angle
    """
    rows, cols = img.shape[:2]
    if center is None:
        center = (cols // 2, rows // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, M, (cols, rows), borderValue=(255, 255, 255))
    return rotated, M

def scaling_transform(img, sx, sy):
    """
    Scaling: scales image by factors (sx, sy)
    """
    rows, cols = img.shape[:2]
    M = np.float32([[sx, 0, 0], [0, sy, 0]])
    scaled = cv2.warpAffine(img, M, (int(cols * sx), int(rows * sy)), 
                            borderValue=(255, 255, 255))
    return scaled, M

def shearing_transform(img, shx, shy):
    """
    Shearing: applies shear transformation
    """
    rows, cols = img.shape[:2]
    M = np.float32([[1, shx, 0], [shy, 1, 0]])
    
    # Calculate new dimensions
    pts = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]]).reshape(-1, 1, 2)
    pts_transformed = cv2.transform(pts, M)
    
    x_coords = pts_transformed[:, :, 0].ravel()
    y_coords = pts_transformed[:, :, 1].ravel()
    
    new_width = int(np.ceil(x_coords.max() - x_coords.min()))
    new_height = int(np.ceil(y_coords.max() - y_coords.min()))
    
    # Adjust translation
    tx = -x_coords.min() if x_coords.min() < 0 else 0
    ty = -y_coords.min() if y_coords.min() < 0 else 0
    M_adjusted = np.float32([[1, shx, tx], [shy, 1, ty]])
    
    sheared = cv2.warpAffine(img, M_adjusted, (new_width, new_height), 
                             borderValue=(255, 255, 255))
    return sheared, M_adjusted

def affine_transform(img, src_points, dst_points):
    """
    Affine transformation: maps 3 source points to 3 destination points
    Preserves parallel lines
    """
    rows, cols = img.shape[:2]
    
    M = cv2.getAffineTransform(src_points, dst_points)
    affine = cv2.warpAffine(img, M, (cols, rows), borderValue=(255, 255, 255))
    return affine, M

def perspective_transform(img, src_points, dst_points):
    """
    Perspective transformation: maps 4 source points to 4 destination points
    Simulates 3D viewing perspective
    """
    rows, cols = img.shape[:2]
    
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    perspective = cv2.warpPerspective(img, M, (cols, rows), 
                                      borderValue=(255, 255, 255))
    return perspective, M

def polar_transform(img):
    """
    Polar transformation: converts Cartesian to polar coordinates
    """
    rows, cols = img.shape[:2]
    center = (cols // 2, rows // 2)
    max_radius = np.sqrt(center[0]**2 + center[1]**2)
    
    # Linear polar transform
    polar = cv2.linearPolar(img, center, max_radius, 
                           cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    
    # Log polar transform
    log_polar = cv2.logPolar(img, center, 40, 
                             cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    
    return polar, log_polar

def composite_transform(img):
    """
    Composite transformation: combination of multiple transforms
    """
    rows, cols = img.shape[:2]
    center = (cols // 2, rows // 2)
    
    # Rotate 30 degrees and scale by 1.2
    M1 = cv2.getRotationMatrix2D(center, 30, 1.2)
    
    # Translate by (50, 30)
    M2 = np.float32([[1, 0, 50], [0, 1, 30]])
    
    # Combine transformations
    M_combined = np.vstack([M1, [0, 0, 1]]) @ np.vstack([M2, [0, 0, 1]])
    M_combined = M_combined[:2]
    
    result = cv2.warpAffine(img, M_combined, (cols, rows), 
                            borderValue=(255, 255, 255))
    return result, M_combined

def demonstrate_interpolation(img):
    """
    Demonstrate different interpolation methods during transformation
    """
    rows, cols = img.shape[:2]
    center = (cols // 2, rows // 2)
    M = cv2.getRotationMatrix2D(center, 45, 1.5)
    
    results = {}
    
    # Nearest neighbor (fastest, lowest quality)
    results['Nearest'] = cv2.warpAffine(img, M, (cols, rows), 
                                        flags=cv2.INTER_NEAREST,
                                        borderValue=(255, 255, 255))
    
    # Linear (good balance)
    results['Linear'] = cv2.warpAffine(img, M, (cols, rows), 
                                       flags=cv2.INTER_LINEAR,
                                       borderValue=(255, 255, 255))
    
    # Cubic (high quality, slower)
    results['Cubic'] = cv2.warpAffine(img, M, (cols, rows), 
                                      flags=cv2.INTER_CUBIC,
                                      borderValue=(255, 255, 255))
    
    # Lanczos (highest quality, slowest)
    results['Lanczos'] = cv2.warpAffine(img, M, (cols, rows), 
                                        flags=cv2.INTER_LANCZOS4,
                                        borderValue=(255, 255, 255))
    
    return results

def homography_rectification():
    """
    Demonstrate homography for perspective correction
    """
    # Create a trapezoid-shaped image (simulating perspective)
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Draw a perspective quadrilateral
    pts = np.array([[150, 100], [450, 80], [500, 350], [100, 320]], np.int32)
    cv2.fillPoly(img, [pts], (100, 150, 200))
    cv2.polylines(img, [pts], True, (0, 0, 0), 3)
    
    # Add text in perspective
    cv2.putText(img, "PERSPECTIVE", (200, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Define source points (in perspective)
    src_pts = np.float32([[150, 100], [450, 80], [500, 350], [100, 320]])
    
    # Define destination points (rectangular)
    dst_pts = np.float32([[150, 100], [450, 100], [450, 350], [150, 350]])
    
    # Compute homography
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Apply rectification
    rectified = cv2.warpPerspective(img, H, (600, 400), 
                                     borderValue=(255, 255, 255))
    
    return img, rectified, src_pts, dst_pts

def main():
    """Main function demonstrating all geometric transformations"""
    print("=" * 70)
    print("Geometric Transformations")
    print("=" * 70)
    
    # Create test image
    print("\nCreating test image...")
    img = create_test_image()
    print("   ✓ Test image created with grid and reference shapes")
    
    # 1. Basic Transformations
    print("\n1. Basic Transformations...")
    
    fig = plt.figure(figsize=(18, 12))
    
    # Original
    plt.subplot(3, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')
    
    # Translation
    translated, _ = translation_transform(img, 100, 50)
    plt.subplot(3, 3, 2)
    plt.imshow(cv2.cvtColor(translated, cv2.COLOR_BGR2RGB))
    plt.title('Translation (100, 50)')
    plt.axis('off')
    
    # Rotation
    rotated, _ = rotation_transform(img, 30)
    plt.subplot(3, 3, 3)
    plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    plt.title('Rotation (30°)')
    plt.axis('off')
    
    # Scaling
    scaled, _ = scaling_transform(img, 1.5, 0.8)
    plt.subplot(3, 3, 4)
    plt.imshow(cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB))
    plt.title('Scaling (1.5x, 0.8y)')
    plt.axis('off')
    
    # Shearing X
    sheared_x, _ = shearing_transform(img, 0.3, 0)
    plt.subplot(3, 3, 5)
    plt.imshow(cv2.cvtColor(sheared_x, cv2.COLOR_BGR2RGB))
    plt.title('Shear X (0.3)')
    plt.axis('off')
    
    # Shearing Y
    sheared_y, _ = shearing_transform(img, 0, 0.3)
    plt.subplot(3, 3, 6)
    plt.imshow(cv2.cvtColor(sheared_y, cv2.COLOR_BGR2RGB))
    plt.title('Shear Y (0.3)')
    plt.axis('off')
    
    # Affine
    src_pts = np.float32([[50, 50], [550, 50], [50, 350]])
    dst_pts = np.float32([[100, 80], [520, 100], [80, 320]])
    affine, _ = affine_transform(img, src_pts, dst_pts)
    plt.subplot(3, 3, 7)
    plt.imshow(cv2.cvtColor(affine, cv2.COLOR_BGR2RGB))
    plt.title('Affine Transform')
    plt.axis('off')
    
    # Perspective
    src_pts_p = np.float32([[50, 50], [550, 50], [550, 350], [50, 350]])
    dst_pts_p = np.float32([[100, 100], [500, 80], [520, 320], [80, 340]])
    perspective, _ = perspective_transform(img, src_pts_p, dst_pts_p)
    plt.subplot(3, 3, 8)
    plt.imshow(cv2.cvtColor(perspective, cv2.COLOR_BGR2RGB))
    plt.title('Perspective Transform')
    plt.axis('off')
    
    # Composite
    composite, _ = composite_transform(img)
    plt.subplot(3, 3, 9)
    plt.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    plt.title('Composite (Rotate + Translate)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('geometric_transformations.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("   ✓ Basic transformations complete")
    
    # 2. Polar Transformations
    print("\n2. Polar Transformations...")
    
    polar, log_polar = polar_transform(img)
    
    fig = plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(polar, cv2.COLOR_BGR2RGB))
    plt.title('Linear Polar')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(log_polar, cv2.COLOR_BGR2RGB))
    plt.title('Log Polar')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('polar_transforms.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("   ✓ Polar transformations complete")
    
    # 3. Interpolation Methods
    print("\n3. Interpolation Methods...")
    
    interp_results = demonstrate_interpolation(img)
    
    fig = plt.figure(figsize=(16, 4))
    
    for idx, (name, result) in enumerate(interp_results.items(), 1):
        plt.subplot(1, 4, idx)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(f'{name} Interpolation')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('interpolation_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("   ✓ Interpolation comparison complete")
    
    # 4. Perspective Rectification
    print("\n4. Perspective Rectification...")
    
    perspective_img, rectified, src_pts, dst_pts = homography_rectification()
    
    # Draw points on images
    perspective_viz = perspective_img.copy()
    rectified_viz = rectified.copy()
    
    for pt in src_pts:
        cv2.circle(perspective_viz, tuple(pt.astype(int)), 8, (255, 0, 0), -1)
    
    for pt in dst_pts:
        cv2.circle(rectified_viz, tuple(pt.astype(int)), 8, (0, 255, 0), -1)
    
    fig = plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(perspective_viz, cv2.COLOR_BGR2RGB))
    plt.title('Perspective View (Distorted)')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(rectified_viz, cv2.COLOR_BGR2RGB))
    plt.title('Rectified (Corrected)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('perspective_rectification.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("   ✓ Perspective rectification complete")
    
    # Summary
    print("\n" + "=" * 70)
    print("Transformation Summary")
    print("=" * 70)
    print("\nAffine Transformations (preserve parallel lines):")
    print("  • Translation: shift image position")
    print("  • Rotation: rotate around center")
    print("  • Scaling: change size")
    print("  • Shearing: skew the image")
    print("  • General Affine: combination using 3 points")
    
    print("\nPerspective Transformation:")
    print("  • Simulates 3D viewing angle")
    print("  • Requires 4 point correspondences")
    print("  • Used for rectification and augmented reality")
    
    print("\nPolar Transformations:")
    print("  • Linear Polar: rotation → horizontal shift")
    print("  • Log Polar: rotation and scale → shifts")
    print("  • Useful for rotation-invariant processing")
    
    print("\nInterpolation Methods:")
    print("  • Nearest: Fastest, blocky artifacts")
    print("  • Linear: Good balance")
    print("  • Cubic: Smooth, higher quality")
    print("  • Lanczos: Best quality, slowest")
    
    print("\n" + "=" * 70)
    print("All geometric transformations completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
