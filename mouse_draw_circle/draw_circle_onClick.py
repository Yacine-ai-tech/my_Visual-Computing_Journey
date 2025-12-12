import cv2
import numpy as np

# Create a blank canvas - started with 256x256 but it was too small
# 512x512 seems like a good size for drawing
black_image = np.zeros((512, 512, 3), np.uint8)

# Mouse callback function
# Took me a while to understand the event system in OpenCV
def draw_circles(event, x, y, flags, param):
    # Only draw when left mouse button is clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # Draw white circles with radius 40
        # Tried different sizes: 20 was too small, 60 too big, 40 feels right
        cv2.circle(black_image, (x, y), 40, (255, 255, 255), -1)
        
        # TODO: Try adding different colors based on keyboard input
        # Maybe 'r' for red, 'g' for green, 'b' for blue?

# Create window and attach mouse callback
cv2.namedWindow('Interactive Drawing')
cv2.setMouseCallback('Interactive Drawing', draw_circles)

print("Click to draw circles. Press ESC to exit.")

# Main loop
while True:
    cv2.imshow('Interactive Drawing', black_image)
    
    # Wait for ESC key (key code 27)
    # Note: cv2.waitKey(1) returns full key code, need & 0xFF for proper comparison
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Clean up
cv2.destroyAllWindows()

# Future improvements:
# - Add ability to change brush size
# - Add color picker
# - Add eraser functionality
# - Save the drawing to file
