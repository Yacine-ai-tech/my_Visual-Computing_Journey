# Setup Guide

Quick guide to get started with the projects in this repository.

## Prerequisites

- Python 3.8 or higher (I'm using 3.10)
- pip (Python package manager)
- Basic Python knowledge
- Familiarity with command line

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Yacine-ai-tech/my_Visual-Computing_Journey.git
cd my_Visual-Computing_Journey
```

### 2. Create Virtual Environment (Recommended)

This keeps dependencies isolated from your system Python.

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

**Option A: Install all common dependencies**
```bash
pip install -r requirements.txt
```

**Option B: Install for specific project**
```bash
cd Contour_detection
pip install -r requirements.txt
```

### 4. Verify Installation

Test if OpenCV is installed correctly:

```python
python -c "import cv2; print(cv2.__version__)"
```

Should print something like `4.8.0` or similar.

## Running Projects

Each project is in its own folder. Navigate to the folder and run the Python script:

```bash
cd mouse_draw_circle
python draw_circle_onClick.py
```

Check individual project READMEs for specific instructions.

## Common Issues

### Issue: "No module named 'cv2'"
**Solution**: OpenCV not installed. Run `pip install opencv-python`

### Issue: "Image not found" or imread returns None
**Solution**: Check your file paths. Use absolute paths or make sure you're running from the correct directory.

### Issue: Window doesn't show up
**Solution**: Make sure you have display capability. For remote servers, you might need X11 forwarding or use matplotlib instead of cv2.imshow().

### Issue: Colors look wrong
**Solution**: Remember OpenCV uses BGR, matplotlib uses RGB. Convert when needed:
```python
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
```

### Issue: Python version conflicts
**Solution**: Use virtual environment to isolate dependencies.

## Project Structure

```
my_Visual-Computing_Journey/
├── README.md                       # Main overview
├── LEARNING_JOURNAL.md            # My learning timeline
├── RESOURCES.md                   # Learning resources
├── PROJECTS_INDEX.md              # All projects organized
├── DEBUGGING_NOTES.md             # Common bugs and fixes
├── TODO.md                        # Future plans
├── requirements.txt               # Common dependencies
├── .gitignore                     # Git ignore file
├── mouse_draw_circle/             # Project 1
│   ├── README.md
│   ├── draw_circle_onClick.py
│   └── requirements.txt
├── morphological_operations/      # Project 2
│   ├── README.md
│   ├── morphological.py
│   ├── cameraman.tif
│   └── requirements.txt
├── Contour_detection/            # Project 3
│   ├── README.md
│   ├── contour_detection.py
│   ├── shape_for_test.jpeg
│   └── requirements.txt
├── edge_detection/               # Project 4
│   ├── README.md
│   ├── edge_detector.py
│   └── requirements.txt
└── experiments/                  # Experimental code
    ├── README.md
    ├── threshold_tests.py
    └── blur_kernel_tests.py
```

## Tips for Best Experience

1. **Start with simpler projects** - Try `mouse_draw_circle` first
2. **Read the project README** - Each project has specific notes
3. **Experiment with parameters** - Change values and see what happens
4. **Check the experiments folder** - See how I tested different approaches
5. **Use an IDE** - VS Code with Python extension is great
6. **Enable linting** - Helps catch errors early

## Recommended Learning Path

If you're new to OpenCV, follow this order:

1. Start with `mouse_draw_circle` - Learn basics
2. Try `morphological_operations` - Understand transformations
3. Move to `edge_detection` - See different algorithms
4. Finally `Contour_detection` - Put it all together

## Resources

Check out `RESOURCES.md` for learning materials I've found helpful.

## Contributing

This is my personal learning repository, but if you spot errors or have suggestions, feel free to open an issue!

## Questions?

If something doesn't work or isn't clear, check:
1. `DEBUGGING_NOTES.md` - I've probably hit that issue
2. Individual project READMEs
3. OpenCV documentation
4. Stack Overflow (my savior many times!)

---

Happy coding! 🚀

*Last updated: December 2024*
