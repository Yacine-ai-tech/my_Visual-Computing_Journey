# My Visual Computing Journey 🖼️👁️

Hey there! This repo documents my journey learning computer vision and image processing. Started this in early 2024 as I realized how important CV is becoming in tech. 

## Why I'm Learning This

After working on a few projects, I realized computer vision is everywhere - from face filters on social media to autonomous vehicles. I wanted to understand the fundamentals before diving into deep learning approaches. So here I am, learning OpenCV and classical CV techniques from the ground up.

## What I've Learned So Far

This is roughly in the order I learned things (spoiler: I didn't plan this perfectly, just following my curiosity):

1. **Basic Mouse Interactions** - Started here to understand how OpenCV handles events and drawing. Pretty basic but helped me get comfortable with the library.

2. **Morphological Operations** - Learned about erosion, dilation, opening, closing. Really useful for cleaning up images and extracting features.

3. **Contour Detection** - This was harder than I expected! Took me a while to get the preprocessing right (blur + threshold). But once it clicked, I could see how powerful it is.

## Current Focus

Right now I'm exploring:
- Edge detection techniques (Canny, Sobel)
- Feature detection and matching
- Image segmentation

## Projects in This Repo

| Project | Difficulty | What I Learned |
|---------|-----------|----------------|
| mouse_draw_circle | ⭐ Beginner | OpenCV basics, event handling |
| morphological_operations | ⭐⭐ Intermediate | Morphological transforms, kernel operations |
| Contour_detection | ⭐⭐ Intermediate | Image preprocessing, contour detection |

## Setup

Most projects use similar dependencies:
```bash
pip install opencv-python numpy matplotlib
```

Check individual project folders for specific requirements.

## Resources I'm Using

- OpenCV official docs (obviously)
- "Learning OpenCV 3" book - pretty comprehensive
- PyImageSearch blog - Adrian's tutorials are gold
- Stack Overflow when I'm stuck (which is often lol)

## Notes to Self

- Remember to always convert BGR to RGB when using matplotlib!
- GaussianBlur kernel size must be odd
- cv2.waitKey(0) waits indefinitely, cv2.waitKey(1) waits 1ms
- Contours work on binary images, so threshold first

## What's Next?

Planning to work on:
- [ ] Feature matching (SIFT/ORB)
- [ ] Object tracking
- [ ] Face detection with Haar cascades
- [ ] Maybe some basic deep learning with YOLO?

Feel free to check out the code. I'm still learning, so if you spot mistakes or have suggestions, let me know!

---

*Last updated: December 2024*
