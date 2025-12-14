"""
Document Processing and Analysis
Implements document detection, text region extraction, and layout analysis
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

class DocumentProcessor:
    """
    Document processing system for detection, deskewing, and text extraction
    """
    
    def __init__(self):
        """Initialize document processor"""
        self.min_area = 10000  # Minimum area for document detection
        
    def detect_document(self, img):
        """
        Detect document in image using edge detection and contours
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to connect broken lines
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest quadrilateral contour
        document_contour = None
        max_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > self.min_area:
                # Approximate contour to polygon
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # Check if it's a quadrilateral and larger than current max
                if len(approx) == 4 and area > max_area:
                    document_contour = approx
                    max_area = area
        
        return document_contour
    
    def order_points(self, pts):
        """
        Order points in: top-left, top-right, bottom-right, bottom-left
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum and difference to find corners
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        rect[0] = pts[np.argmin(s)]      # top-left has smallest sum
        rect[2] = pts[np.argmax(s)]      # bottom-right has largest sum
        rect[1] = pts[np.argmin(diff)]   # top-right has smallest difference
        rect[3] = pts[np.argmax(diff)]   # bottom-left has largest difference
        
        return rect
    
    def perspective_transform(self, img, corners):
        """
        Apply perspective transformation to get bird's eye view
        """
        # Order the corners
        rect = self.order_points(corners.reshape(4, 2))
        (tl, tr, br, bl) = rect
        
        # Calculate width of new image
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        
        # Calculate height of new image
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        
        # Destination points for perspective transform
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(rect, dst)
        
        # Apply transformation
        warped = cv2.warpPerspective(img, M, (max_width, max_height))
        
        return warped, M
    
    def deskew_document(self, img):
        """
        Detect and correct skew in document
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        _, thresh = cv2.threshold(gray, 0, 255, 
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find all coordinates of text pixels
        coords = np.column_stack(np.where(thresh > 0))
        
        # Calculate minimum area rectangle
        angle = cv2.minAreaRect(coords)[-1]
        
        # Correct angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Rotate image
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), 
                                flags=cv2.INTER_CUBIC, 
                                borderMode=cv2.BORDER_REPLICATE)
        
        return rotated, angle
    
    def detect_text_regions(self, img):
        """
        Detect text regions using morphological operations
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Horizontal kernel to detect text lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, 
                                         iterations=2)
        
        # Vertical kernel for text blocks
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        detected_blocks = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, 
                                          iterations=2)
        
        # Combine horizontal and vertical
        combined = cv2.addWeighted(detected_lines, 0.5, detected_blocks, 0.5, 0.0)
        
        # Dilate to connect text regions
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(combined, kernel, iterations=3)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and sort text regions
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter based on size
            if area > 500 and w > 20 and h > 10:
                text_regions.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': w / h
                })
        
        # Sort by y-coordinate (top to bottom)
        text_regions = sorted(text_regions, key=lambda r: r['bbox'][1])
        
        return text_regions
    
    def extract_tables(self, img):
        """
        Detect and extract table regions
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, 
                                           iterations=2)
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, 
                                         iterations=2)
        
        # Combine lines
        table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        
        # Find contours
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        tables = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            if area > 5000:  # Minimum table size
                tables.append((x, y, w, h))
        
        return tables, horizontal_lines, vertical_lines
    
    def analyze_layout(self, img, text_regions):
        """
        Analyze document layout and classify regions
        """
        layout = {
            'header': [],
            'body': [],
            'footer': [],
            'columns': 1
        }
        
        if len(text_regions) == 0:
            return layout
        
        h, w = img.shape[:2]
        
        # Classify regions based on position
        for region in text_regions:
            x, y, rw, rh = region['bbox']
            y_ratio = y / h
            
            if y_ratio < 0.15:
                layout['header'].append(region)
            elif y_ratio > 0.85:
                layout['footer'].append(region)
            else:
                layout['body'].append(region)
        
        # Detect multi-column layout
        if len(layout['body']) > 0:
            x_positions = [r['bbox'][0] for r in layout['body']]
            
            # Simple column detection using x-position clustering
            if len(x_positions) > 1:
                x_sorted = sorted(x_positions)
                gaps = [x_sorted[i+1] - x_sorted[i] for i in range(len(x_sorted)-1)]
                
                if gaps:
                    avg_gap = np.mean(gaps)
                    large_gaps = sum(1 for g in gaps if g > avg_gap * 2)
                    layout['columns'] = large_gaps + 1
        
        return layout


def create_document_image():
    """Create synthetic document image"""
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Add some perspective distortion
    pts1 = np.float32([[50, 50], [750, 80], [730, 550], [70, 520]])
    pts2 = np.float32([[0, 0], [800, 0], [800, 600], [0, 600]])
    M = cv2.getPerspectiveTransform(pts2, pts1)
    
    # Create document content
    doc_content = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Title
    cv2.putText(doc_content, "DOCUMENT TITLE", (100, 80), 
               cv2.FONT_HERSHEY_BOLD, 1.5, (0, 0, 0), 3)
    cv2.line(doc_content, (100, 100), (700, 100), (0, 0, 0), 2)
    
    # Paragraphs
    y = 150
    for _ in range(3):
        for line in range(4):
            cv2.rectangle(doc_content, (100, y), (700, y + 15), (50, 50, 50), -1)
            y += 25
        y += 30
    
    # Table
    table_y = 400
    # Horizontal lines
    for i in range(4):
        cv2.line(doc_content, (100, table_y + i * 30), (700, table_y + i * 30), 
                (0, 0, 0), 2)
    # Vertical lines
    for i in range(5):
        cv2.line(doc_content, (100 + i * 150, table_y), (100 + i * 150, table_y + 90), 
                (0, 0, 0), 2)
    
    # Apply perspective
    warped = cv2.warpPerspective(doc_content, M, (800, 600))
    
    # Add some noise
    noise = np.random.normal(0, 10, warped.shape).astype(np.uint8)
    warped = cv2.add(warped, noise)
    
    return warped


def main():
    """Main function for document processing"""
    
    print("=" * 70)
    print("Document Processing and Analysis")
    print("=" * 70)
    
    # Create synthetic document
    print("\nCreating synthetic document...")
    img = create_document_image()
    print("   ✓ Document image created with perspective distortion")
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # 1. Document Detection
    print("\n1. Detecting document boundaries...")
    document_contour = processor.detect_document(img)
    
    if document_contour is not None:
        print(f"   ✓ Document detected with {len(document_contour)} corners")
        
        # Draw contour
        img_with_contour = img.copy()
        cv2.drawContours(img_with_contour, [document_contour], -1, (0, 255, 0), 3)
        
        # 2. Perspective Correction
        print("\n2. Applying perspective correction...")
        warped, M = processor.perspective_transform(img, document_contour)
        print("   ✓ Document rectified")
        
        # 3. Deskew
        print("\n3. Detecting and correcting skew...")
        deskewed, angle = processor.deskew_document(warped)
        print(f"   ✓ Skew corrected (angle: {angle:.2f}°)")
        
        # 4. Text Region Detection
        print("\n4. Detecting text regions...")
        text_regions = processor.detect_text_regions(deskewed)
        print(f"   ✓ Found {len(text_regions)} text regions")
        
        # Draw text regions
        img_with_text = deskewed.copy()
        for region in text_regions:
            x, y, w, h = region['bbox']
            cv2.rectangle(img_with_text, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 5. Table Detection
        print("\n5. Detecting tables...")
        tables, h_lines, v_lines = processor.extract_tables(deskewed)
        print(f"   ✓ Found {len(tables)} table(s)")
        
        # Draw tables
        img_with_tables = deskewed.copy()
        for (x, y, w, h) in tables:
            cv2.rectangle(img_with_tables, (x, y), (x + w, y + h), (255, 0, 0), 3)
        
        # 6. Layout Analysis
        print("\n6. Analyzing document layout...")
        layout = processor.analyze_layout(deskewed, text_regions)
        print(f"   ✓ Layout detected:")
        print(f"     - Header regions: {len(layout['header'])}")
        print(f"     - Body regions: {len(layout['body'])}")
        print(f"     - Footer regions: {len(layout['footer'])}")
        print(f"     - Column layout: {layout['columns']} column(s)")
        
        # Visualize pipeline
        print("\n7. Visualizing processing pipeline...")
        
        fig = plt.figure(figsize=(18, 10))
        
        # Original with contour
        plt.subplot(2, 4, 1)
        plt.imshow(cv2.cvtColor(img_with_contour, cv2.COLOR_BGR2RGB))
        plt.title('1. Document Detection')
        plt.axis('off')
        
        # Perspective corrected
        plt.subplot(2, 4, 2)
        plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        plt.title('2. Perspective Corrected')
        plt.axis('off')
        
        # Deskewed
        plt.subplot(2, 4, 3)
        plt.imshow(cv2.cvtColor(deskewed, cv2.COLOR_BGR2RGB))
        plt.title(f'3. Deskewed ({angle:.1f}°)')
        plt.axis('off')
        
        # Text regions
        plt.subplot(2, 4, 4)
        plt.imshow(cv2.cvtColor(img_with_text, cv2.COLOR_BGR2RGB))
        plt.title(f'4. Text Regions ({len(text_regions)})')
        plt.axis('off')
        
        # Table detection - horizontal lines
        plt.subplot(2, 4, 5)
        plt.imshow(h_lines, cmap='gray')
        plt.title('5a. Horizontal Lines')
        plt.axis('off')
        
        # Table detection - vertical lines
        plt.subplot(2, 4, 6)
        plt.imshow(v_lines, cmap='gray')
        plt.title('5b. Vertical Lines')
        plt.axis('off')
        
        # Tables detected
        plt.subplot(2, 4, 7)
        plt.imshow(cv2.cvtColor(img_with_tables, cv2.COLOR_BGR2RGB))
        plt.title(f'6. Tables ({len(tables)})')
        plt.axis('off')
        
        # Final processed document
        plt.subplot(2, 4, 8)
        plt.imshow(cv2.cvtColor(deskewed, cv2.COLOR_BGR2RGB))
        plt.title('7. Final Output')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('document_processing_pipeline.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("   ✓ Visualization saved")
        
    else:
        print("   ✗ No document detected in image")
    
    # Summary
    print("\n" + "=" * 70)
    print("Document Processing Summary")
    print("=" * 70)
    
    print("\nProcessing Steps:")
    print("  1. Document detection (edge detection + contours)")
    print("  2. Perspective correction (homography transformation)")
    print("  3. Skew correction (angle detection + rotation)")
    print("  4. Text region extraction (morphological operations)")
    print("  5. Table detection (line detection)")
    print("  6. Layout analysis (region classification)")
    
    print("\nApplications:")
    print("  • Automated document scanning")
    print("  • Invoice processing")
    print("  • Form recognition")
    print("  • Receipt digitization")
    print("  • Book scanning")
    print("  • ID card processing")
    
    print("\nAdvanced Techniques:")
    print("  • OCR integration (Tesseract, PaddleOCR)")
    print("  • Deep learning document understanding (LayoutLM, Donut)")
    print("  • Table structure recognition")
    print("  • Handwriting recognition")
    print("  • Multi-page document alignment")
    print("  • Document classification")
    
    print("\nFor Production Use:")
    print("  • Use pre-trained models (Document AI, AWS Textract)")
    print("  • Implement batch processing")
    print("  • Add quality checks and validation")
    print("  • Handle various document types")
    print("  • Optimize for mobile devices")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
