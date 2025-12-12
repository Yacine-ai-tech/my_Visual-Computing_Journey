# Debugging Notes & Common Errors

Keeping track of bugs I've encountered and how I fixed them. Future me will thank present me!

## OpenCV Installation Issues

### Problem: opencv-python wouldn't install on Python 3.11
**Error**: Some cryptic build error about wheels
**Solution**: Downgraded to Python 3.10 temporarily, then later found out opencv-python updated to support 3.11
**Date**: March 2024
**Lesson**: Check package compatibility with Python version first

---

## Image Not Loading

### Problem: imread() returning None
**Code**: `img = cv2.imread('image.jpg')`
**Error**: Silent failure, img is None, then later operations crash
**Solution**: 
1. Check file path (was using wrong relative path)
2. Add error checking: `if img is None: print("Error loading")`
**Lesson**: ALWAYS check if imread() succeeded before processing

---

## Window Closes Immediately

### Problem: cv2.imshow() shows window for split second then closes
**Code**: 
```python
cv2.imshow('Image', img)
```
**Solution**: Need to add `cv2.waitKey(0)` to keep window open
**Explanation**: waitKey(0) waits indefinitely, waitKey(1) waits 1ms
**Lesson**: Always pair imshow with waitKey

---

## BGR vs RGB Color Issues

### Problem: Colors look wrong in matplotlib
**Symptoms**: Blue sky looks orange, red things look blue
**Cause**: OpenCV uses BGR, matplotlib expects RGB
**Solution**: `img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`
**Frequency**: Happened like 5 times before I remembered
**Lesson**: Convert BGR→RGB when using matplotlib!

---

## Kernel Size Must Be Odd

### Problem: GaussianBlur crashes with even kernel size
**Error**: `error: (-215:Assertion failed) ksize.width > 0 && ksize.width % 2 == 1`
**Bad code**: `cv2.GaussianBlur(img, (4, 4), 0)`
**Fixed code**: `cv2.GaussianBlur(img, (5, 5), 0)`
**Lesson**: Kernel dimensions must be positive odd integers (3, 5, 7, 9, etc.)

---

## Contours Not Found

### Problem: findContours() returns empty list
**Tried**:
1. Image was RGB not grayscale → converted to gray
2. Image wasn't binary → added threshold step
3. Threshold value was wrong → switched to adaptive threshold
**Working pipeline**: grayscale → blur → threshold → findContours
**Lesson**: Contours need binary input, preprocessing matters!

---

## Type Conversion Issues

### Problem: Sobel returns negative values, can't display
**Error**: Image looks weird or crashes
**Cause**: Sobel can produce negative gradients, uint8 can't handle that
**Solution**: 
```python
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)  # Use CV_64F
# Process...
sobelx = np.uint8(np.abs(sobelx))  # Convert back
```
**Lesson**: Use CV_64F for operations that might produce negative values

---

## Threshold Output is All White/Black

### Problem: Adaptive threshold giving binary but wrong results
**First attempt**: Tried cv2.THRESH_BINARY with global threshold
**Values tried**: 100, 127, 150 - none worked well
**Solution**: Used adaptive threshold instead
```python
cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                      cv2.THRESH_BINARY, 11, 2)
```
**Lesson**: Global threshold sucks for real images, adaptive is better

---

## Python Syntax Errors

### Problem: Forgetting colons, wrong indentation
**Examples**:
- `if event == cv2.EVENT_LBUTTONDOWN` → forgot colon
- Mixed tabs and spaces (VS Code auto-fix saved me)
**Solution**: Use a good IDE with linting (VS Code + Python extension)
**Lesson**: Let the tools help you!

---

## Import Errors

### Problem: Module not found even after pip install
**Cause**: Multiple Python installations, installed in wrong environment
**Solution**: 
1. Use virtual environment: `python -m venv venv`
2. Activate it: `source venv/bin/activate` (Linux/Mac)
3. Install packages in venv
**Lesson**: Always use virtual environments for projects

---

## Performance Issues

### Problem: Processing large images was SLOW
**Issue**: 4K image taking forever to process
**Solution**: 
1. Resize before processing: `cv2.resize(img, (800, 600))`
2. Only process region of interest if possible
**Lesson**: Don't process full resolution unless necessary

---

## Matplotlib Not Showing Images

### Problem: plt.show() called but no window appears
**Tried**:
- Checked if image was loaded correctly ✓
- Checked if image was right format ✓
**Solution**: Was running in wrong environment (needed matplotlib backend)
**For Jupyter**: Use `%matplotlib inline`
**For scripts**: Make sure matplotlib backend is installed
**Lesson**: Different environments need different setups

---

## Can't Save Images

### Problem: cv2.imwrite() silently fails
**Cause**: Directory doesn't exist
**Solution**: Create directory first or use existing path
```python
import os
os.makedirs('output', exist_ok=True)
cv2.imwrite('output/result.png', img)
```
**Lesson**: Check/create output directories

---

## Array Shape Mismatches

### Problem: Operations fail due to wrong dimensions
**Error**: `shapes (512,512,3) and (512,512) not aligned`
**Cause**: Mixing grayscale (2D) and color (3D) images
**Solution**: Convert consistently, check shape with `img.shape`
**Lesson**: Be aware of image dimensions (H,W) vs (H,W,C)

---

## Tips to Avoid Bugs

1. **Print shapes often**: `print(img.shape)` saves time
2. **Check None values**: After imread, findContours, etc.
3. **Visualize intermediate steps**: Don't wait until end to check
4. **Use small test images first**: Faster iteration
5. **Read error messages carefully**: They usually tell you what's wrong
6. **Google the exact error**: Someone else has had it
7. **Check OpenCV version**: v3 vs v4 have differences

---

*Updated regularly as I encounter new issues*
