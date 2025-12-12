# Learning Journal

## Week 1 - Getting Started (March 2024)

### Day 1-2: Setup and Hello World
- Installed OpenCV, fought with Python versions for a bit (3.11 wasn't compatible with opencv-python initially)
- Created first simple program - just loading and displaying an image
- Learned the hard way that cv2.imshow() needs cv2.waitKey() or the window closes immediately

### Day 3-4: Mouse Events
- Worked on mouse_draw_circle project
- Started with drawing on click, then figured out how to draw circles
- Tried to make it draw on drag but that was messier than expected - sticking with click for now
- Fun fact: spent 30 mins debugging why circles weren't showing up... forgot to call cv2.imshow() in the loop 🤦

## Week 2-3 - Image Transformations (Late March)

### Morphological Operations
- Learned about erosion and dilation - initially confused about what they do
- The cameraman image is a classic! Found it in a tutorial
- Tried different kernel sizes - bigger kernels = more dramatic effects
- Gradient was interesting - shows edges using morphology instead of edge detection
- Saved intermediate results to compare effects

**Key insight**: Morphological ops are great for noise removal and feature extraction without losing too much shape info

## Week 4-5 - Contours (Early April)

### Contour Detection Project
- This one took longer than expected
- First attempt: used simple threshold - results were terrible
- Second attempt: added GaussianBlur - better but still noisy
- Final version: adaptive threshold worked much better for varying lighting

**Challenges faced**:
1. Finding the right blur kernel size (tried 3x3, 5x5, 7x7 - settled on 5x5)
2. Understanding RETR_TREE vs RETR_EXTERNAL hierarchy modes
3. Figuring out why contours were all over the place - preprocessing was the key!

**Things I learned**:
- Always preprocess: grayscale → blur → threshold
- Adaptive threshold > global threshold for real-world images
- cv2.findContours() returns different things in OpenCV 3 vs 4 (had to handle both)

### Debugging Notes
- When contours look weird, check:
  - Is image binary?
  - Is it actually grayscale (single channel)?
  - Threshold value might be too high/low

## Ongoing Learning (May onwards)

### Current experiments:
- Playing with different edge detection methods (Canny looks promising)
- Trying to understand when to use which preprocessing technique
- Want to build something practical - maybe a document scanner?

### Questions I still have:
- When to use morphological ops vs edge detection?
- How to handle really noisy images?
- What's the deal with different color spaces (HSV, LAB, etc.)?

### Resources that helped:
- OpenCV docs (though sometimes a bit dense)
- PyImageSearch blog - super practical examples
- This Stack Overflow answer on adaptive thresholding: [saved for reference]
- YouTube channel "sentdex" - good OpenCV tutorials

## Mistakes and Lessons

1. **Don't skip preprocessing** - I tried to detect contours on raw images. Didn't work. Always blur and threshold first.

2. **Test with different images** - What works on one image might fail on another. The shapes image was easy, real photos are harder.

3. **Start simple** - I wanted to jump straight into object detection. Going through basics first made everything make more sense.

4. **Read error messages carefully** - OpenCV errors can be cryptic but usually point to the right problem (wrong number of channels, wrong data type, etc.)

5. **Save intermediate results** - Helps in understanding where things go wrong in the pipeline

## Next Steps

- [ ] Learn feature detection (SIFT, SURF, ORB)
- [ ] Try template matching
- [ ] Build a practical project - maybe QR code detector?
- [ ] Understand how to optimize for real-time processing
- [ ] Eventually get into deep learning approaches (YOLO, etc.)

## Random Notes

- cv2.imshow() doesn't work in Jupyter notebooks - use matplotlib instead
- Color conversion is ALWAYS needed: BGR (OpenCV) → RGB (matplotlib)
- When in doubt, check the shape of your arrays with .shape
- Keep kernel sizes odd numbered
- The documentation is your friend, even if it's boring

---

*This is a living document - updating as I learn more*
