# Interactive Circle Drawing

Interactive tool for drawing circles using mouse clicks.

## Description

Creates a blank canvas where circles can be drawn by clicking with the mouse. Demonstrates OpenCV event handling and window management.

## Usage

```bash
pip install -r requirements.txt
python draw_circle_onClick.py
```

Click anywhere on the canvas to draw a white circle. Press ESC to exit.

## Features

- Mouse event handling with callback functions
- Real-time drawing on canvas
- Keyboard input detection for program control

## Implementation Details

- Uses `cv2.namedWindow()` for window creation
- Mouse callbacks registered with `cv2.setMouseCallback()`
- Event loop with `cv2.waitKey()` for continuous display
- Key detection with bitwise mask for cross-platform compatibility

## Requirements

See `requirements.txt` for dependencies.
