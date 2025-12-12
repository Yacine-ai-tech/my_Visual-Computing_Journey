# Drawing Circles with Mouse Clicks

My first real OpenCV project! Simple but helped me understand event handling.

## What It Does

Creates a blank canvas where you can draw white circles by clicking. Press ESC to exit.

## How to Run

```bash
pip install -r requirements.txt
python draw_circle_onClick.py
```

## What I Learned

This was my "hello world" for OpenCV event handling. Learned about:
- Creating windows with `cv2.namedWindow()`
- Setting up mouse callbacks
- The event loop pattern in OpenCV
- Key press detection with `cv2.waitKey()`

## The Tricky Parts

- **Initially forgot `cv2.imshow()` in the loop** - window showed up blank and I was so confused
- **Figuring out the waitKey mask** - needed `& 0xFF` for proper key detection on some systems
- **Circle size experimentation** - tried 20 (too small), 60 (too big), settled on 40

## Ideas for Future

- Add different colors (maybe keyboard controls: 'r' for red, 'g' for green, etc.)
- Add brush size controls
- Implement an eraser (draw black circles?)
- Save the drawing to a file
- Add drag-to-draw functionality (tried this but got messy)

## Requirements

Check `requirements.txt` - just basic OpenCV and NumPy.

---

Started: March 2024  
Status: Working, could expand with more features
