# My Visual Computing Journey 🖼️👁️

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Learning-orange)]()

Hey there! This repo documents my journey learning computer vision and image processing. Started this in early 2024 as I realized how important CV is becoming in tech. 

## Why I'm Learning This

After working on a few projects, I realized computer vision is everywhere - from face filters on social media to autonomous vehicles. I wanted to understand the fundamentals before diving into deep learning approaches. So here I am, learning OpenCV and classical CV techniques from the ground up.

## What I've Learned So Far

This is roughly in the order I learned things (spoiler: I didn't plan this perfectly, just following my curiosity):

1. **Basic Mouse Interactions** - Started here to understand how OpenCV handles events and drawing. Pretty basic but helped me get comfortable with the library.

2. **Morphological Operations** - Learned about erosion, dilation, opening, closing. Really useful for cleaning up images and extracting features.

3. **Contour Detection** - This was harder than I expected! Took me a while to get the preprocessing right (blur + threshold). But once it clicked, I could see how powerful it is.

4. **Edge Detection** - Compared Canny, Sobel, and Laplacian methods. Each has its strengths depending on the use case.

## Current Focus

Right now I'm exploring:
- Feature detection and matching (SIFT, ORB)
- Image segmentation techniques
- Color space transformations

## Long-term Roadmap

This repository is structured to grow with my learning journey:

**Phase 1 (Current): Classical Computer Vision** ✅
- OpenCV fundamentals
- Image processing operations
- Edge and contour detection
- Feature detection

**Phase 2 (Next): Intermediate Techniques** 🔄
- Object detection (Haar cascades, HOG)
- Video processing and tracking
- Camera calibration
- Template matching

**Phase 3 (Future): Deep Learning** 🎯
- CNNs for image classification
- Modern object detection (YOLO, R-CNN family)
- Semantic segmentation
- Transfer learning

**Phase 4 (Advanced): Cutting-Edge Models** 🚀
- Vision Transformers (ViT)
- Vision Language Models (CLIP, BLIP, LLaVA)
- Foundation models (SAM, DINO)
- Neural rendering (NeRF)
- Multimodal AI

*The repository structure is designed to scale from basics to advanced topics naturally.*

**📖 See [SCALABILITY_GUIDE.md](SCALABILITY_GUIDE.md) for details on how this repository accommodates progression from basic operations to VLMs, video processing, and neural rendering.**

## Projects in This Repo

| Project | Difficulty | What I Learned | Status |
|---------|-----------|----------------|--------|
| mouse_draw_circle | ⭐ Beginner | OpenCV basics, event handling | ✅ Complete |
| morphological_operations | ⭐⭐ Intermediate | Morphological transforms, kernel operations | ✅ Complete |
| Contour_detection | ⭐⭐ Intermediate | Image preprocessing, contour detection | ✅ Complete |
| edge_detection | ⭐⭐ Intermediate | Canny, Sobel, Laplacian methods | ✅ Complete |

## Repository Structure

```
📁 my_Visual-Computing_Journey/
├── 📄 README.md                    ← You are here
├── 📄 LEARNING_JOURNAL.md         ← My learning timeline and reflections
├── 📄 RESOURCES.md                ← Books, courses, tutorials I'm using
├── 📄 PROJECTS_INDEX.md           ← All projects organized by difficulty
├── 📄 DEBUGGING_NOTES.md          ← Common bugs I've encountered
├── 📄 TODO.md                     ← Future plans and ideas (includes VLMs, video processing)
├── 📄 SETUP_GUIDE.md              ← How to get started
├── 📄 SCALABILITY_GUIDE.md        ← How this repo scales from basics to advanced topics
├── 📁 mouse_draw_circle/          ← Project 1: Interactive drawing
├── 📁 morphological_operations/   ← Project 2: Image transformations
├── 📁 Contour_detection/          ← Project 3: Boundary detection
├── 📁 edge_detection/             ← Project 4: Edge detection comparison
└── 📁 experiments/                ← Quick tests and experiments
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Yacine-ai-tech/my_Visual-Computing_Journey.git
cd my_Visual-Computing_Journey

# Install dependencies
pip install -r requirements.txt

# Try a project
cd mouse_draw_circle
python draw_circle_onClick.py
```

For detailed setup instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md).

## Resources I'm Using

- OpenCV official docs (obviously)
- "Learning OpenCV 3" book - pretty comprehensive
- PyImageSearch blog - Adrian's tutorials are gold
- Stack Overflow when I'm stuck (which is often lol)

See [RESOURCES.md](RESOURCES.md) for the complete list with links and notes.

## Notes to Self

- Remember to always convert BGR to RGB when using matplotlib!
- GaussianBlur kernel size must be odd
- cv2.waitKey(0) waits indefinitely, cv2.waitKey(1) waits 1ms
- Contours work on binary images, so threshold first

More debugging notes in [DEBUGGING_NOTES.md](DEBUGGING_NOTES.md).

## What's Next?

Planning to work on:
- [ ] Feature matching (SIFT/ORB)
- [ ] Object tracking
- [ ] Face detection with Haar cascades
- [ ] Document scanner (practical project!)
- [ ] Video processing fundamentals
- [ ] Deep learning for CV (YOLO, etc.)
- [ ] Eventually: Vision Transformers and VLMs

*See [TODO.md](TODO.md) for the complete roadmap including advanced topics like Vision Language Models, Neural Rendering, and Foundation Models.*

Full TODO list in [TODO.md](TODO.md).

## Contributing

This is my personal learning repository, but if you spot mistakes or have suggestions for improvement, feel free to open an issue! I'm always looking to learn better approaches.

## License

This project is open source and available under the [MIT License](LICENSE).

## Connect

If you're also learning computer vision or have tips to share, I'd love to connect! Feel free to reach out.

---

*Last updated: December 2024*

**Note**: This is a learning repository. Code quality varies as I progress. Early projects are simpler, later ones show improved understanding. That's the point of a journey! 🚀

Feel free to check out the code. I'm still learning, so if you spot mistakes or have suggestions, let me know!
