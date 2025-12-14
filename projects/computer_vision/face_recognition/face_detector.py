"""
Face Detection and Recognition
Implements face detection using Haar Cascades and HOG, plus facial landmark detection
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_face_test_image():
    """Create a synthetic face-like pattern for testing"""
    img = np.ones((400, 300, 3), dtype=np.uint8) * 220
    
    # Draw face oval
    cv2.ellipse(img, (150, 200), (100, 130), 0, 0, 360, (255, 220, 190), -1)
    cv2.ellipse(img, (150, 200), (100, 130), 0, 0, 360, (150, 100, 80), 2)
    
    # Draw eyes
    cv2.ellipse(img, (120, 170), (20, 15), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (180, 170), (20, 15), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img, (120, 170), 8, (50, 50, 50), -1)
    cv2.circle(img, (180, 170), 8, (50, 50, 50), -1)
    
    # Draw eyebrows
    cv2.ellipse(img, (120, 150), (25, 10), 0, 0, 180, (100, 70, 50), 3)
    cv2.ellipse(img, (180, 150), (25, 10), 0, 0, 180, (100, 70, 50), 3)
    
    # Draw nose
    pts = np.array([[150, 185], [145, 210], [155, 210]], np.int32)
    cv2.polylines(img, [pts], True, (150, 100, 80), 2)
    
    # Draw mouth
    cv2.ellipse(img, (150, 250), (35, 20), 0, 0, 180, (150, 80, 80), 3)
    
    # Add hair
    cv2.ellipse(img, (150, 130), (105, 80), 0, 180, 360, (80, 60, 40), -1)
    
    return img

def haar_cascade_face_detection(img):
    """
    Face detection using Haar Cascade classifiers
    Fast and classic method
    """
    # Load pre-trained Haar Cascade classifiers
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml'
    )
    smile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_smile.xml'
    )
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    result = img.copy()
    
    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(result, 'Face', (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Detect eyes within face region
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = result[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            cv2.putText(roi_color, 'Eye', (ex, ey - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Detect smile within face region
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
            cv2.putText(roi_color, 'Smile', (sx, sy - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    return result, len(faces)

def create_multiple_faces_image():
    """Create image with multiple face-like patterns"""
    img = np.ones((600, 800, 3), dtype=np.uint8) * 240
    
    # Function to draw a simple face
    def draw_simple_face(img, center, size, color):
        x, y = center
        s = size
        
        # Face circle
        cv2.circle(img, (x, y), s, color, -1)
        cv2.circle(img, (x, y), s, (0, 0, 0), 2)
        
        # Eyes
        eye_offset = s // 3
        eye_size = s // 5
        cv2.circle(img, (x - eye_offset, y - s//4), eye_size, (255, 255, 255), -1)
        cv2.circle(img, (x + eye_offset, y - s//4), eye_size, (255, 255, 255), -1)
        cv2.circle(img, (x - eye_offset, y - s//4), eye_size//2, (50, 50, 50), -1)
        cv2.circle(img, (x + eye_offset, y - s//4), eye_size//2, (50, 50, 50), -1)
        
        # Smile
        cv2.ellipse(img, (x, y + s//4), (s//2, s//3), 0, 0, 180, (150, 80, 80), 2)
    
    # Draw multiple faces
    faces_positions = [
        ((150, 150), 60, (255, 220, 190)),
        ((400, 150), 70, (255, 210, 180)),
        ((650, 150), 55, (255, 225, 195)),
        ((150, 400), 65, (255, 215, 185)),
        ((400, 400), 60, (255, 220, 190)),
        ((650, 400), 75, (255, 210, 180))
    ]
    
    for pos, size, color in faces_positions:
        draw_simple_face(img, pos, size, color)
    
    return img

def hog_face_detection(img):
    """
    Face detection using HOG (Histogram of Oriented Gradients)
    More accurate than Haar Cascades for frontal faces
    """
    # Note: OpenCV's HOG face detector requires dlib or specific models
    # Here we'll use Haar as fallback with different parameters for demonstration
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
    )
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30)
    )
    
    result = img.copy()
    
    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(result, f'Face {w}x{h}', (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    return result, len(faces)

def face_recognition_simple(img, known_faces_features):
    """
    Simple face recognition using template matching
    In production, use face_recognition library or deep learning models
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    result = img.copy()
    
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = gray[y:y+h, x:x+w]
        face_roi_resized = cv2.resize(face_roi, (100, 100))
        
        # Simple recognition using correlation (placeholder)
        # In real application, use deep learning embeddings
        best_match = "Unknown"
        best_score = 0
        
        for name, template in known_faces_features.items():
            template_resized = cv2.resize(template, (100, 100))
            
            # Calculate similarity (normalized correlation)
            result_match = cv2.matchTemplate(face_roi_resized, template_resized, 
                                            cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result_match)
            
            if max_val > best_score:
                best_score = max_val
                best_match = name
        
        # Draw detection
        color = (0, 255, 0) if best_score > 0.7 else (0, 0, 255)
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        cv2.putText(result, f'{best_match} ({best_score:.2f})', 
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return result

def eye_detection(img):
    """Dedicated eye detection"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml'
    )
    
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
    
    result = img.copy()
    
    for (x, y, w, h) in eyes:
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.circle(result, (x + w//2, y + h//2), 3, (255, 0, 255), -1)
    
    return result, len(eyes)

def demonstrate_cascade_parameters(img):
    """Show effect of different detection parameters"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    results = {}
    
    # Vary parameters
    params = [
        (1.1, 3, "Sensitive"),
        (1.1, 5, "Balanced"),
        (1.2, 7, "Conservative"),
        (1.3, 10, "Very Conservative")
    ]
    
    for scale_factor, min_neighbors, label in params:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(30, 30)
        )
        
        result = img.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.putText(result, f'{label}: {len(faces)} faces', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        results[label] = result
    
    return results

def main():
    """Main function demonstrating face detection techniques"""
    print("=" * 70)
    print("Face Detection and Recognition")
    print("=" * 70)
    
    # Create test images
    print("\nCreating test images...")
    single_face = create_face_test_image()
    multiple_faces = create_multiple_faces_image()
    print("   ✓ Test images created")
    
    # 1. Haar Cascade Face Detection
    print("\n1. Haar Cascade Face Detection...")
    
    result_single, num_faces_single = haar_cascade_face_detection(single_face)
    result_multiple, num_faces_multiple = haar_cascade_face_detection(multiple_faces)
    
    fig = plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(single_face, cv2.COLOR_BGR2RGB))
    plt.title('Original Single Face')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(result_single, cv2.COLOR_BGR2RGB))
    plt.title(f'Detected: {num_faces_single} face(s)')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(multiple_faces, cv2.COLOR_BGR2RGB))
    plt.title('Original Multiple Faces')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(result_multiple, cv2.COLOR_BGR2RGB))
    plt.title(f'Detected: {num_faces_multiple} face(s)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('face_detection_haar.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"   ✓ Single face: {num_faces_single} detected")
    print(f"   ✓ Multiple faces: {num_faces_multiple} detected")
    
    # 2. HOG Face Detection (alternative approach)
    print("\n2. Alternative Detection Method...")
    
    hog_result, num_hog = hog_face_detection(multiple_faces)
    
    fig = plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(multiple_faces, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(hog_result, cv2.COLOR_BGR2RGB))
    plt.title(f'Alternative Method: {num_hog} faces')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('face_detection_alternative.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"   ✓ Detected {num_hog} faces")
    
    # 3. Eye Detection
    print("\n3. Eye Detection...")
    
    eyes_result, num_eyes = eye_detection(single_face)
    
    fig = plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(single_face, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(eyes_result, cv2.COLOR_BGR2RGB))
    plt.title(f'Eye Detection: {num_eyes} eyes')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('eye_detection.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"   ✓ Detected {num_eyes} eyes")
    
    # 4. Parameter Effects
    print("\n4. Detection Parameter Effects...")
    
    param_results = demonstrate_cascade_parameters(multiple_faces)
    
    fig = plt.figure(figsize=(16, 4))
    
    for idx, (label, result) in enumerate(param_results.items(), 1):
        plt.subplot(1, 4, idx)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(label)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('detection_parameters.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("   ✓ Parameter comparison complete")
    
    # Summary
    print("\n" + "=" * 70)
    print("Face Detection Summary")
    print("=" * 70)
    print("\nHaar Cascade Classifiers:")
    print("  • Fast and lightweight")
    print("  • Works in real-time")
    print("  • Can detect faces, eyes, smile, etc.")
    print("  • Sensitive to face angle and lighting")
    
    print("\nKey Parameters:")
    print("  • scaleFactor: How much image size is reduced at each scale")
    print("    - Lower (1.1): More sensitive, slower")
    print("    - Higher (1.3): Less sensitive, faster")
    print("  • minNeighbors: How many neighbors each rectangle should have")
    print("    - Lower (3): More detections, more false positives")
    print("    - Higher (10): Fewer false positives, may miss faces")
    print("  • minSize: Minimum object size to detect")
    
    print("\nApplications:")
    print("  • Face detection in photos")
    print("  • Real-time video face tracking")
    print("  • Facial recognition systems")
    print("  • Emotion detection")
    print("  • Face filters and effects")
    print("  • Attendance systems")
    print("  • Security and surveillance")
    
    print("\nFor Production Face Recognition:")
    print("  • Use face_recognition library (dlib)")
    print("  • Use deep learning models (FaceNet, ArcFace)")
    print("  • Use cloud APIs (AWS Rekognition, Azure Face)")
    
    print("\n" + "=" * 70)
    print("All face detection demonstrations completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
