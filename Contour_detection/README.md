# Contour Detection in Images

My most challenging project so far! Took me several attempts to get this working properly.

## What This Does

Detects and draws contours (boundaries) around shapes in an image. Shows both the original image and the image with detected contours highlighted in red.

## The Journey

### First Attempt (Failed)
Tried to detect contours directly on the color image. Got a ton of weird results - contours everywhere, nothing made sense. Learned that contours need a binary (black and white) image.

### Second Attempt (Better but Still Bad)
Converted to grayscale and used simple threshold with value 127. Better, but still picking up too much noise. Image had varying lighting, so a single threshold value didn't work well.

### Third Attempt (Almost There)
Added Gaussian blur before thresholding. This helped reduce noise! But still had issues with shadows and uneven lighting.

### Final Version (Success!)
Used adaptive thresholding instead of global threshold. This was the key! Adaptive threshold calculates threshold value locally, so it handles varying lighting conditions much better.

## How to Run

```bash
pip install -r requirements.txt
python contour_detection.py
```

Make sure `shape_for_test.jpeg` is in the same directory.

## The Code Pipeline

1. **Load and convert to RGB** - OpenCV loads as BGR, need RGB for matplotlib
2. **Convert to grayscale** - Contours work on single-channel images
3. **Apply Gaussian blur** - Reduces noise (kernel size 5x5)
4. **Adaptive threshold** - Creates binary image, handles varying lighting
5. **Find contours** - Detects boundaries in binary image
6. **Draw contours** - Visualizes results on original image

## Parameters I Experimented With

### Gaussian Blur Kernel Size
- 3x3: Still too noisy
- **5x5: Goldilocks zone** ✓
- 7x7: Losing some detail
- 9x9: Too blurry, missing edges

### Adaptive Threshold Block Size
- 7: Too small, overly sensitive
- **11: Works well** ✓
- 15: Too large, missing details
- Must be odd number!

### Adaptive Threshold Constant (C)
- 0: Too many false positives
- **2: Good balance** ✓
- 5: Missing some contours
- 10: Way too conservative

## What I Learned

### Technical Stuff
- Preprocessing is CRITICAL - can't skip blur and threshold steps
- Adaptive threshold > global threshold for real-world images
- `RETR_TREE` gets all contours with hierarchy info
- `CHAIN_APPROX_SIMPLE` saves memory by compressing contour points

### Debugging Skills
- Always visualize intermediate steps (saved gray, blurred, threshold images)
- Print shapes and types: `print(img.shape, img.dtype)`
- Check if image loaded: `if img is None`
- OpenCV 3 vs 4 difference: findContours returns 2 vs 3 values (handled both)

## Common Issues I Hit

1. **Contours all over the place**: Forgot to blur, image had too much noise
2. **No contours found**: Image wasn't binary, forgot threshold step
3. **Weird colors in matplotlib**: Forgot BGR to RGB conversion
4. **Empty contour list**: Wrong contour mode (RETR_EXTERNAL vs RETR_TREE)

## Future Improvements

- [ ] Filter contours by area (remove tiny noise contours)
- [ ] Calculate contour properties (area, perimeter, center)
- [ ] Draw bounding boxes with `cv2.boundingRect()`
- [ ] Shape detection (identify circles, triangles, squares)
- [ ] Contour approximation with `cv2.approxPolyDP()`
- [ ] Try different images (real-world photos vs simple shapes)

## Use Cases

Contour detection is useful for:
- Object detection and counting
- Shape analysis
- Document scanning (finding page boundaries)
- Gesture recognition (hand contours)
- Motion detection (changed regions)

## Requirements

See `requirements.txt`:
- opencv-python
- matplotlib  
- numpy

## Tips for Using This Code

1. **Try different images** - What works on simple shapes might fail on complex scenes
2. **Adjust parameters** - Different images need different blur/threshold values
3. **Filter by area** - Use `cv2.contourArea()` to remove noise
4. **Check hierarchy** - Useful for detecting nested shapes

## My Observations

The shapes test image is pretty simple - clean background, good contrast. Real-world images are much harder:
- Varying lighting
- Complex backgrounds
- Texture noise
- Shadows

For real applications, might need:
- Better preprocessing (bilateral filter, morphological ops)
- Color-based segmentation first
- Machine learning approaches for complex scenes

## Resources That Helped

- OpenCV docs on contours (dense but comprehensive)
- PyImageSearch article "Contours in OpenCV" (super helpful!)
- Stack Overflow answer on adaptive vs global threshold
- My own experimentation (trial and error taught me the most!)

---

**Started**: Early April 2024  
**Completed**: Mid April 2024 (after many iterations!)  
**Difficulty**: ⭐⭐ Intermediate  
**Time spent**: ~6-8 hours (including debugging and experimentation)

This project really taught me that preprocessing matters more than the algorithm itself sometimes!

