"""
Visual Language Models (VLM) Demonstration
Educational implementation showing VLM concepts and multimodal processing
Note: This is a simplified educational demo. Production VLMs use large transformer models.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

class SimpleVLM:
    """
    Simplified Visual Language Model for educational purposes
    Demonstrates the concept of processing both visual and textual information
    """
    
    def __init__(self):
        """Initialize VLM components"""
        # Visual feature extractors
        self.color_extractor = ColorFeatureExtractor()
        self.shape_extractor = ShapeFeatureExtractor()
        self.spatial_extractor = SpatialFeatureExtractor()
        
        # Simple vocabulary
        self.vocabulary = {
            'colors': ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'white', 'black'],
            'shapes': ['circle', 'rectangle', 'square', 'triangle', 'oval'],
            'sizes': ['small', 'medium', 'large'],
            'positions': ['top', 'bottom', 'left', 'right', 'center'],
            'quantities': ['one', 'two', 'three', 'several', 'many']
        }
        
    def encode_image(self, img):
        """
        Extract visual features from image
        Simulates vision encoder in VLMs (like CLIP)
        """
        # Extract different types of features
        color_features = self.color_extractor.extract(img)
        shape_features = self.shape_extractor.extract(img)
        spatial_features = self.spatial_extractor.extract(img)
        
        visual_encoding = {
            'color': color_features,
            'shape': shape_features,
            'spatial': spatial_features,
            'raw_image': img
        }
        
        return visual_encoding
    
    def generate_description(self, visual_encoding):
        """
        Generate textual description from visual features
        Simulates language decoder in VLMs
        """
        descriptions = []
        
        # Color description
        dominant_colors = visual_encoding['color']['dominant']
        if dominant_colors:
            color_desc = f"The image contains {', '.join(dominant_colors)} colors."
            descriptions.append(color_desc)
        
        # Shape description
        shapes = visual_encoding['shape']['detected_shapes']
        if shapes:
            shape_counts = {}
            for shape in shapes:
                shape_type = shape['type']
                shape_counts[shape_type] = shape_counts.get(shape_type, 0) + 1
            
            shape_desc = "There are "
            shape_parts = [f"{count} {shape}{'s' if count > 1 else ''}" 
                          for shape, count in shape_counts.items()]
            shape_desc += ", ".join(shape_parts) + " in the image."
            descriptions.append(shape_desc)
        
        # Spatial description
        spatial_desc = self._describe_spatial_layout(visual_encoding['spatial'])
        if spatial_desc:
            descriptions.append(spatial_desc)
        
        return " ".join(descriptions)
    
    def _describe_spatial_layout(self, spatial_features):
        """Describe spatial arrangement of objects"""
        regions = spatial_features['regions']
        
        if not regions:
            return ""
        
        descriptions = []
        for region_name, info in regions.items():
            if info['object_count'] > 0:
                descriptions.append(f"{info['object_count']} object(s) in the {region_name}")
        
        if descriptions:
            return "Spatial layout: " + ", ".join(descriptions) + "."
        return ""
    
    def answer_question(self, visual_encoding, question):
        """
        Answer questions about the image
        Simulates VQA (Visual Question Answering)
        """
        question_lower = question.lower()
        
        # Color questions
        if 'color' in question_lower or 'what color' in question_lower:
            colors = visual_encoding['color']['dominant']
            if colors:
                return f"The main colors are {', '.join(colors)}."
            return "I cannot determine the colors clearly."
        
        # Count questions
        if 'how many' in question_lower:
            shapes = visual_encoding['shape']['detected_shapes']
            
            # Check what to count
            for shape_type in ['circle', 'rectangle', 'square', 'triangle']:
                if shape_type in question_lower:
                    count = sum(1 for s in shapes if s['type'] == shape_type)
                    return f"There are {count} {shape_type}{'s' if count != 1 else ''}."
            
            # Count all objects
            total = len(shapes)
            return f"There are {total} object{'s' if total != 1 else ''} in total."
        
        # Position questions
        if 'where' in question_lower or 'position' in question_lower:
            spatial = visual_encoding['spatial']['regions']
            positions = [name for name, info in spatial.items() if info['object_count'] > 0]
            if positions:
                return f"Objects are located in: {', '.join(positions)}."
            return "I cannot determine positions clearly."
        
        # Size questions
        if 'size' in question_lower or 'big' in question_lower or 'small' in question_lower:
            shapes = visual_encoding['shape']['detected_shapes']
            if shapes:
                avg_size = np.mean([s['size'] for s in shapes])
                if avg_size > 5000:
                    return "The objects are large."
                elif avg_size > 1000:
                    return "The objects are medium-sized."
                else:
                    return "The objects are small."
            return "I cannot determine sizes."
        
        return "I'm not sure how to answer that question about this image."
    
    def visual_reasoning(self, visual_encoding):
        """
        Perform simple visual reasoning tasks
        """
        shapes = visual_encoding['shape']['detected_shapes']
        
        reasoning_results = {
            'total_objects': len(shapes),
            'shape_distribution': {},
            'color_distribution': {},
            'spatial_patterns': []
        }
        
        # Analyze shape distribution
        for shape in shapes:
            shape_type = shape['type']
            reasoning_results['shape_distribution'][shape_type] = \
                reasoning_results['shape_distribution'].get(shape_type, 0) + 1
        
        # Analyze color distribution
        for color, percentage in visual_encoding['color']['percentages'].items():
            if percentage > 5:  # More than 5%
                reasoning_results['color_distribution'][color] = percentage
        
        # Spatial reasoning
        regions = visual_encoding['spatial']['regions']
        if regions['top']['object_count'] > regions['bottom']['object_count']:
            reasoning_results['spatial_patterns'].append("Objects concentrated at top")
        
        return reasoning_results


class ColorFeatureExtractor:
    """Extract color features from images"""
    
    def extract(self, img):
        """Extract dominant colors and their percentages"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define color ranges
        color_ranges = {
            'red': [([0, 100, 100], [10, 255, 255]), ([170, 100, 100], [180, 255, 255])],
            'green': [([40, 40, 40], [80, 255, 255])],
            'blue': [([100, 50, 50], [130, 255, 255])],
            'yellow': [([20, 100, 100], [35, 255, 255])],
        }
        
        total_pixels = img.shape[0] * img.shape[1]
        color_percentages = {}
        
        for color_name, ranges in color_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                lower = np.array(lower)
                upper = np.array(upper)
                color_mask = cv2.inRange(hsv, lower, upper)
                mask = cv2.bitwise_or(mask, color_mask)
            
            pixels = np.sum(mask > 0)
            percentage = (pixels / total_pixels) * 100
            color_percentages[color_name] = percentage
        
        # Get dominant colors (> 10%)
        dominant = [color for color, pct in color_percentages.items() if pct > 10]
        
        return {
            'percentages': color_percentages,
            'dominant': dominant
        }


class ShapeFeatureExtractor:
    """Extract shape features from images"""
    
    def extract(self, img):
        """Detect and classify shapes"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        detected_shapes = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > 500:  # Minimum size
                # Approximate shape
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                
                x, y, w, h = cv2.boundingRect(contour)
                
                # Classify shape
                num_vertices = len(approx)
                aspect_ratio = w / float(h) if h != 0 else 0
                
                if num_vertices == 3:
                    shape_type = 'triangle'
                elif num_vertices == 4:
                    if 0.95 <= aspect_ratio <= 1.05:
                        shape_type = 'square'
                    else:
                        shape_type = 'rectangle'
                elif num_vertices > 8:
                    shape_type = 'circle'
                else:
                    shape_type = 'polygon'
                
                detected_shapes.append({
                    'type': shape_type,
                    'bbox': (x, y, w, h),
                    'size': area,
                    'vertices': num_vertices
                })
        
        return {
            'detected_shapes': detected_shapes,
            'count': len(detected_shapes)
        }


class SpatialFeatureExtractor:
    """Extract spatial layout features"""
    
    def extract(self, img):
        """Analyze spatial distribution of objects"""
        h, w = img.shape[:2]
        
        # Detect objects (simplified)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Define regions
        regions = {
            'top': {'bounds': (0, 0, w, h//3), 'object_count': 0},
            'center': {'bounds': (0, h//3, w, 2*h//3), 'object_count': 0},
            'bottom': {'bounds': (0, 2*h//3, w, h), 'object_count': 0},
        }
        
        # Count objects in each region
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, cw, ch = cv2.boundingRect(contour)
                center_y = y + ch // 2
                
                if center_y < h // 3:
                    regions['top']['object_count'] += 1
                elif center_y < 2 * h // 3:
                    regions['center']['object_count'] += 1
                else:
                    regions['bottom']['object_count'] += 1
        
        return {'regions': regions}


def create_multimodal_scene():
    """Create a scene for VLM testing"""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 240
    
    # Add various shapes with different colors
    # Red circles
    cv2.circle(img, (100, 100), 40, (0, 0, 255), -1)
    cv2.circle(img, (300, 100), 35, (0, 0, 255), -1)
    
    # Green rectangles
    cv2.rectangle(img, (450, 50), (550, 130), (0, 255, 0), -1)
    
    # Blue shapes
    cv2.rectangle(img, (80, 250), (180, 350), (255, 0, 0), -1)
    cv2.circle(img, (400, 300), 45, (255, 0, 0), -1)
    
    # Yellow triangle
    pts = np.array([[300, 250], [250, 350], [350, 350]], np.int32)
    cv2.fillPoly(img, [pts], (0, 255, 255))
    
    return img


def main():
    """Main demonstration of VLM concepts"""
    
    print("=" * 70)
    print("Visual Language Models (VLM) Educational Demo")
    print("=" * 70)
    
    print("\nNote: This is a simplified educational demonstration.")
    print("Production VLMs use large transformer models (CLIP, BLIP, LLaVA, etc.)")
    
    # Create test scene
    print("\nCreating multimodal test scene...")
    img = create_multimodal_scene()
    print("   ✓ Test scene created")
    
    # Initialize VLM
    print("\nInitializing Visual Language Model...")
    vlm = SimpleVLM()
    print("   ✓ VLM initialized")
    
    # Encode image
    print("\n1. Encoding visual information...")
    visual_encoding = vlm.encode_image(img)
    print("   ✓ Visual features extracted:")
    print(f"      - Colors detected: {visual_encoding['color']['dominant']}")
    print(f"      - Shapes detected: {visual_encoding['shape']['count']}")
    
    # Generate description
    print("\n2. Generating image description...")
    description = vlm.generate_description(visual_encoding)
    print(f"   Generated: \"{description}\"")
    
    # Visual Question Answering
    print("\n3. Visual Question Answering (VQA)...")
    questions = [
        "What colors are in the image?",
        "How many circles are there?",
        "Where are the objects located?",
        "How many red objects are there?",
        "What is the size of the objects?"
    ]
    
    answers = {}
    for question in questions:
        answer = vlm.answer_question(visual_encoding, question)
        answers[question] = answer
        print(f"   Q: {question}")
        print(f"   A: {answer}")
        print()
    
    # Visual Reasoning
    print("4. Visual Reasoning...")
    reasoning = vlm.visual_reasoning(visual_encoding)
    print(f"   Total objects: {reasoning['total_objects']}")
    print(f"   Shape distribution: {reasoning['shape_distribution']}")
    print(f"   Color distribution: {reasoning['color_distribution']}")
    print(f"   Spatial patterns: {reasoning['spatial_patterns']}")
    
    # Visualization
    print("\n5. Creating visualization...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Input Image')
    plt.axis('off')
    
    # Color distribution
    plt.subplot(2, 3, 2)
    colors = list(visual_encoding['color']['percentages'].keys())
    percentages = list(visual_encoding['color']['percentages'].values())
    plt.bar(colors, percentages)
    plt.title('Color Distribution (%)')
    plt.xticks(rotation=45)
    plt.ylabel('Percentage')
    plt.grid(True, alpha=0.3)
    
    # Shape detection visualization
    plt.subplot(2, 3, 3)
    img_with_shapes = img.copy()
    for shape in visual_encoding['shape']['detected_shapes']:
        x, y, w, h = shape['bbox']
        cv2.rectangle(img_with_shapes, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_with_shapes, shape['type'], (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(img_with_shapes, cv2.COLOR_BGR2RGB))
    plt.title('Shape Detection')
    plt.axis('off')
    
    # Shape distribution
    plt.subplot(2, 3, 4)
    shape_dist = reasoning['shape_distribution']
    if shape_dist:
        plt.bar(shape_dist.keys(), shape_dist.values())
        plt.title('Shape Count')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    # Generated description
    plt.subplot(2, 3, 5)
    plt.text(0.1, 0.5, f'Description:\n\n"{description}"',
            wrap=True, fontsize=10, verticalalignment='center')
    plt.title('Generated Caption')
    plt.axis('off')
    
    # VQA results
    plt.subplot(2, 3, 6)
    qa_text = "Visual Q&A:\n\n"
    for i, (q, a) in enumerate(list(answers.items())[:3]):
        qa_text += f"Q{i+1}: {q}\n"
        qa_text += f"A{i+1}: {a}\n\n"
    plt.text(0.1, 0.5, qa_text, wrap=True, fontsize=8, 
            verticalalignment='center', family='monospace')
    plt.title('Question Answering')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('vlm_demonstration.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("   ✓ Visualization saved")
    
    # Summary
    print("\n" + "=" * 70)
    print("VLM Concepts Summary")
    print("=" * 70)
    
    print("\nKey Components of VLMs:")
    print("  1. Vision Encoder - Extracts visual features from images")
    print("     • Typically uses CNNs or Vision Transformers (ViT)")
    print("     • Examples: CLIP vision encoder, ResNet, EfficientNet")
    
    print("\n  2. Language Decoder - Generates text from visual features")
    print("     • Uses transformer models (GPT-like)")
    print("     • Generates captions, answers questions")
    
    print("\n  3. Multimodal Fusion - Combines vision and language")
    print("     • Cross-attention mechanisms")
    print("     • Learned alignment between modalities")
    
    print("\nProduction VLM Models:")
    print("  • CLIP (OpenAI) - Vision-language pre-training")
    print("  • BLIP/BLIP-2 (Salesforce) - Bootstrapped language-image pre-training")
    print("  • LLaVA - Large Language and Vision Assistant")
    print("  • Flamingo (DeepMind) - Few-shot learning VLM")
    print("  • GPT-4V (OpenAI) - Multimodal GPT-4")
    print("  • Gemini (Google) - Native multimodal model")
    
    print("\nApplications:")
    print("  • Image captioning")
    print("  • Visual question answering (VQA)")
    print("  • Visual reasoning and inference")
    print("  • Image-text retrieval")
    print("  • Visual dialog systems")
    print("  • Accessibility (describing images for visually impaired)")
    print("  • Content moderation")
    print("  • Medical image analysis with clinical notes")
    
    print("\nFor Production Use:")
    print("  • Use pre-trained models (Hugging Face Transformers)")
    print("  • Fine-tune on domain-specific data")
    print("  • Consider computational requirements (GPU memory)")
    print("  • Implement efficient inference (quantization, pruning)")
    print("  • Use APIs (OpenAI GPT-4V, Google Gemini, Anthropic Claude)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
