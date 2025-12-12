# Morphological Operations

Implementation of basic morphological operations for image processing.

## Description

Demonstrates morphological operations on grayscale images:
- **Erosion** - Shrinks bright regions, removes small noise
- **Dilation** - Expands bright regions, fills small holes
- **Gradient** - Difference between dilation and erosion, highlights edges

## Usage

```bash
pip install -r requirements.txt
python morphological.py
```

## Operations

### Erosion
Removes small objects and separates connected regions. Uses a structuring element (kernel) to probe the image and shrink bright areas.

### Dilation
Fills gaps and connects nearby objects. Expands bright regions using a structuring element.

### Gradient
Computed as the difference between dilation and erosion results. Effective for edge detection without explicit edge detection algorithms.

### Advanced Operations
- **Opening** - Erosion followed by dilation, removes noise while preserving shape
- **Closing** - Dilation followed by erosion, fills gaps in objects

## Output

The script saves results as:
- `erosion_result.tif`
- `dilation_result.tif`
- `gradient_result.tif`

## Requirements

See `requirements.txt` for dependencies.
