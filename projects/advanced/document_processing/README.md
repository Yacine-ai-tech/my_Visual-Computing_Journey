# Document Processing and Analysis

Comprehensive document processing system for detection, perspective correction, text region extraction, and OCR preparation.

## Overview

This module provides both educational and production-ready implementations for document processing:
- **Educational Demo** (`document_analyzer.py`): Classical CV techniques for document processing
- **Production** (`production_ocr.py`): Real OCR using Tesseract and EasyOCR

## Features

### Document Detection

#### Edge-Based Detection
- Canny edge detection
- Contour-based document finding
- Quadrilateral approximation
- Multi-document detection

#### Corner Detection
- Automatic corner point identification
- Corner ordering (top-left, top-right, bottom-right, bottom-left)
- Robust to rotation and skew

### Perspective Correction

#### Homography Transformation
- Four-point perspective transform
- Document rectification
- Aspect ratio preservation
- Bird's eye view generation

#### Deskewing
- Angle detection using Hough transform
- Rotation correction
- Horizon alignment

### Text Region Detection

#### Region Identification
- Connected component analysis
- Text line detection
- Word segmentation
- Character-level regions

#### Layout Analysis
- Header/footer detection
- Column detection
- Paragraph segmentation
- Table detection

### OCR Preparation

#### Image Enhancement
- Binarization (Otsu's method)
- Noise removal
- Contrast enhancement
- Border removal

#### Text Optimization
- Skew correction
- Size normalization
- Background removal
- Character isolation

## Usage

### Educational Demo

```bash
cd projects/advanced/document_processing
python document_analyzer.py
```

The demo will:
1. Create synthetic document images
2. Detect document boundaries
3. Apply perspective correction
4. Detect text regions
5. Analyze layout structure
6. Save annotated results

### Production OCR

```bash
# Install OCR engines
# Tesseract: apt-get install tesseract-ocr
# EasyOCR: pip install easyocr

cd projects/advanced/document_processing
python production_ocr.py
```

For detailed production setup, see `projects/advanced/README_PRODUCTION.md`.

## Algorithm Details

### Document Detection Process

```python
class DocumentProcessor:
    def detect_document(self, img):
        # 1. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # 4. Dilate edges
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # 5. Find contours
        contours = cv2.findContours(dilated, ...)
        
        # 6. Find largest quadrilateral
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4:  # Quadrilateral
                return approx
```

### Corner Ordering

```python
def order_points(pts):
    # Order: [top-left, top-right, bottom-right, bottom-left]
    rect = np.zeros((4, 2), dtype=np.float32)
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # Top-left: smallest sum
    rect[2] = pts[np.argmax(s)]      # Bottom-right: largest sum
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # Top-right: smallest diff
    rect[3] = pts[np.argmax(diff)]   # Bottom-left: largest diff
    
    return rect
```

### Perspective Transform

```python
def perspective_transform(img, corners):
    # Order corners
    rect = order_points(corners)
    (tl, tr, br, bl) = rect
    
    # Compute width and height
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    
    # Destination points
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)
    
    # Compute homography
    M = cv2.getPerspectiveTransform(rect, dst)
    
    # Apply transformation
    warped = cv2.warpPerspective(img, M, (max_width, max_height))
    
    return warped
```

### Text Region Detection

```python
def detect_text_regions(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply morphological gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    morph = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    
    # Threshold
    _, thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Connect text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours (text regions)
    contours = cv2.findContours(connected, cv2.RETR_EXTERNAL, ...)
    
    return contours
```

## Document Types

### Supported Document Types

#### Business Documents
- Invoices and receipts
- Forms and applications
- Contracts and agreements
- Letters and memos

#### Identity Documents
- Passports
- ID cards
- Driver's licenses
- Visas

#### Financial Documents
- Bank statements
- Tax forms
- Insurance documents
- Checks

#### Educational Documents
- Certificates
- Transcripts
- Diplomas
- Report cards

## Applications

### Document Digitization
- Scan-to-digital conversion
- Archive digitization
- Document management systems
- Cloud storage preparation

### Automated Data Entry
- Invoice processing
- Form filling
- Receipt scanning
- ID verification

### Document Analysis
- Text extraction
- Table extraction
- Signature detection
- Stamp recognition

### Mobile Scanning
- Camera-based scanning
- Real-time document capture
- On-device processing
- Cloud upload

## Production OCR Engines

### Tesseract OCR

**Advantages:**
- Free and open-source
- 100+ language support
- Mature and stable
- Good accuracy on printed text

**Usage:**
```python
import pytesseract
from PIL import Image

# Extract text
text = pytesseract.image_to_string(image, lang='eng')

# Get bounding boxes
data = pytesseract.image_to_data(image, output_type=Output.DICT)
```

### EasyOCR

**Advantages:**
- Deep learning-based
- 80+ language support
- Better on low-quality images
- Handles handwriting better

**Usage:**
```python
import easyocr

reader = easyocr.Reader(['en'])
results = reader.readtext(image)

for bbox, text, confidence in results:
    print(f"Text: {text}, Confidence: {confidence}")
```

## Performance Comparison

| Engine | Speed | Accuracy | Languages | Handwriting |
|--------|-------|----------|-----------|-------------|
| Tesseract | Fast | Good | 100+ | Fair |
| EasyOCR | Medium | Excellent | 80+ | Good |
| Cloud OCR | Slow | Excellent | 100+ | Excellent |

## Output Files

### Educational Demo
- `document_detection.png`: Detected document boundaries
- `perspective_correction.png`: Rectified documents
- `text_regions.png`: Detected text areas
- `layout_analysis.png`: Document structure

### Production OCR
- `ocr_results.txt`: Extracted text
- `ocr_annotated.png`: Visualization with bounding boxes
- `ocr_data.json`: Structured OCR output
- `confidence_scores.csv`: Per-word confidence

## Parameters Guide

### Document Detection

```python
# Edge detection
canny_low = 50      # Lower threshold
canny_high = 150    # Upper threshold

# Contour approximation
epsilon = 0.02      # Approximation accuracy (0.01-0.05)

# Size filtering
min_area = 10000    # Minimum document area in pixels
```

### Perspective Transform

```python
# Border handling
border_mode = cv2.BORDER_CONSTANT
border_value = (255, 255, 255)  # White background

# Interpolation
interpolation = cv2.INTER_LINEAR  # or INTER_CUBIC for better quality
```

### OCR Preparation

```python
# Binarization
threshold_method = cv2.THRESH_BINARY | cv2.THRESH_OTSU

# Denoising
denoise_strength = 10  # 5-15 typical range

# Dilation/Erosion
kernel_size = (3, 3)
iterations = 1
```

## Tips for Best Results

### Document Capture

**Lighting:**
- Use even, diffuse lighting
- Avoid shadows and glare
- Ensure sufficient brightness
- Consider flash for dark environments

**Camera Position:**
- Hold camera parallel to document
- Center document in frame
- Maintain consistent distance
- Avoid rotation and skew

**Document Condition:**
- Flatten wrinkled documents
- Clean stains and marks
- Ensure text is clear and legible
- Use higher resolution for small text

### Preprocessing

**Image Enhancement:**
1. Deskew document
2. Correct perspective distortion
3. Enhance contrast
4. Remove noise
5. Binarize image

**For Better OCR:**
- Use 300 DPI or higher
- Ensure text is at least 12pt equivalent
- Convert to grayscale
- Apply adaptive thresholding
- Remove backgrounds

### Layout Analysis

**Text Detection:**
- Use morphological operations to connect text
- Filter by aspect ratio for text lines
- Sort regions top-to-bottom, left-to-right
- Handle multi-column layouts

**Table Detection:**
- Detect horizontal and vertical lines
- Find line intersections
- Create cell grid
- Extract cell contents

## Common Issues

**Problem**: Document not detected
**Solution**: Increase contrast, use better lighting, adjust edge detection thresholds

**Problem**: Incorrect corner detection
**Solution**: Improve preprocessing, manually specify corners, use template matching

**Problem**: Distorted text after perspective correction
**Solution**: Verify corner ordering, check aspect ratio, manual corner adjustment

**Problem**: Poor OCR accuracy
**Solution**: Improve image quality, use appropriate language model, try different OCR engine

**Problem**: Missed text regions
**Solution**: Adjust morphological kernel sizes, lower threshold, use connected components

## Advanced Techniques

### Adaptive Thresholding

```python
# Better than global thresholding for varying illumination
thresh = cv2.adaptiveThreshold(
    gray, 255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 
    11, 2
)
```

### Skew Detection and Correction

```python
def correct_skew(image):
    # Detect skew angle using Hough transform
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    
    # Calculate average angle
    angles = [line[0][1] for line in lines]
    median_angle = np.median(angles)
    
    # Rotate image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), 
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    
    return rotated
```

### Table Extraction

```python
def extract_table(image):
    # Detect horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
    
    # Combine lines
    table_structure = cv2.add(horizontal_lines, vertical_lines)
    
    # Find intersections (cell corners)
    intersections = cv2.bitwise_and(horizontal_lines, vertical_lines)
    
    return table_structure, intersections
```

## Extensions

The code can be extended with:
- Deep learning document detection (DocUNet, DewarpNet)
- Table structure recognition
- Form field detection
- Signature verification
- Handwriting recognition (HTR)
- Multi-page document processing
- Language detection
- Layout understanding with transformers

## Requirements

### Educational Demo
- opencv-python >= 4.5.0
- numpy >= 1.19.0
- matplotlib >= 3.3.0

### Production OCR
- pytesseract >= 0.3.0
- tesseract-ocr (system package)
- easyocr >= 1.6.0
- pillow >= 9.0.0

## References

### Papers
- He, M., et al. (2017). Deep Residual Text Detection Network
- Liao, M., et al. (2019). Scene Text Detection with Polygon Offsetting
- Li, H., et al. (2021). TrOCR: Transformer-based Optical Character Recognition

### Tools & Libraries
- Tesseract OCR: https://github.com/tesseract-ocr/tesseract
- EasyOCR: https://github.com/JaidedAI/EasyOCR
- DocTR: https://github.com/mindee/doctr
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR

### Datasets
- ICDAR Robust Reading Competition
- FUNSD (Form Understanding in Noisy Scanned Documents)
- IAM Handwriting Database
- CORD (Consolidated Receipt Dataset)
