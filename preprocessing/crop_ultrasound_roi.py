"""
Automatic ROI detection and cropping for ultrasound frames.

Ultrasound images typically contain a bright fan-shaped sector (the actual
image data) surrounded by dead black space, UI legends, depth markers, and
color bars. This module detects the sector region and crops frames to it,
so downstream models (DINOv2, RAFT) only process meaningful content.

Algorithm:
    1. Convert to grayscale
    2. Threshold to separate bright content from black surround
    3. Morphological close → open to merge the fan sector and remove
       small legend text / UI elements
    4. Find the largest contour (= the ultrasound sector)
    5. Return its axis-aligned bounding rectangle

The bounding box is computed once from the first frame of a video and reused
for all subsequent frames — the sector position is fixed within a recording.
"""

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def detect_ultrasound_roi(
    frame: np.ndarray,
    threshold: int = 10,
    morph_kernel_size: int = 25,
    pad: int = 5,
) -> tuple:
    """
    Detect the bounding box of the ultrasound fan sector in a frame.

    Args:
        frame: Input image as numpy array (H, W) or (H, W, 3), uint8.
        threshold: Pixel intensity threshold (0-255). Pixels above this
                   are considered ultrasound content. Default 10 works well
                   for bright-on-black ultrasound displays.
        morph_kernel_size: Size of the structuring element for morphological
                          operations. Larger = more aggressive merging of
                          nearby bright regions and removal of small text.
        pad: Pixels of padding to keep around the detected ROI as a safety
             margin. Set to 0 for a tight crop.

    Returns:
        Tuple (y_min, y_max, x_min, x_max) defining the crop rectangle,
        or None if no significant content is found.
    """
    if cv2 is None:
        raise ImportError(
            "OpenCV (cv2) is required for ROI detection. "
            "Install it with: pip install opencv-python"
        )

    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame.copy()

    # Binary threshold — ultrasound content is brighter than the black surround
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Morphological close to merge nearby bright regions (fills small gaps
    # within the fan sector), then open to remove small isolated bright
    # spots like legend text, markers, and UI elements
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find contours and pick the largest one (the fan sector)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)

    # Minimum area sanity check — the fan should be at least 10% of the frame
    frame_area = gray.shape[0] * gray.shape[1]
    if cv2.contourArea(largest) < 0.10 * frame_area:
        # Likely just noise, not a real ultrasound sector
        return None

    x, y, w, h = cv2.boundingRect(largest)

    # Apply padding with bounds clamping
    img_h, img_w = gray.shape[:2]
    y_min = max(0, y - pad)
    y_max = min(img_h, y + h + pad)
    x_min = max(0, x - pad)
    x_max = min(img_w, x + w + pad)

    return (y_min, y_max, x_min, x_max)


def crop_to_roi(frame: np.ndarray, bbox: tuple) -> np.ndarray:
    """
    Crop a frame to the given bounding box.

    Args:
        frame: Input image (H, W) or (H, W, C).
        bbox: Tuple (y_min, y_max, x_min, x_max) from detect_ultrasound_roi.

    Returns:
        Cropped image array.
    """
    y_min, y_max, x_min, x_max = bbox
    return frame[y_min:y_max, x_min:x_max]


def detect_and_crop(
    frame: np.ndarray,
    threshold: int = 10,
    morph_kernel_size: int = 25,
    pad: int = 5,
) -> tuple:
    """
    Convenience function: detect ROI and crop in one call.

    Returns:
        Tuple of (cropped_frame, bbox). If detection fails, returns
        (original_frame, None).
    """
    bbox = detect_ultrasound_roi(
        frame, threshold=threshold, morph_kernel_size=morph_kernel_size, pad=pad
    )
    if bbox is None:
        return frame, None
    return crop_to_roi(frame, bbox), bbox
