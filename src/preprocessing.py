### These are preprocessing functions to make each frame ready for lane detection stage
import cv2
import numpy as np

def to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Increases the contrast of the grayscale image using the CLAHE method.
# This makes the white road lane markings appear more prominent against the dark asphalt.
def apply_clahe(frame, clip_limit = 2.0, grid_size= (8, 8)):
    clahe = cv2.createCLAHE(clipLimit= clip_limit, tileGridSize= grid_size)
    enhanced_frame = clahe.apply(frame)
    return enhanced_frame

# Noise Reduction
def apply_blur(frame, kernel_size = 5):
    return cv2.GaussianBlur(
        frame, 
        (kernel_size, kernel_size), 
        0
    )

# Edge detection
def apply_canny(frame, low_thresh= 50, high_thresh= 150):
    return cv2.Canny(frame, threshold1= low_thresh, threshold2= high_thresh)

# Morphological operations are used to remove noise and improve the continuity and shape of detected lane lines.
def apply_morphological_closing(frame, kernel_size=(5, 5)):
    structuring_element = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        ksize=kernel_size
    )

    return cv2.morphologyEx(
        frame,
        cv2.MORPH_CLOSE,
        structuring_element
    )

# Region of interest
def apply_roi(frame):
    height, width = frame.shape
    mask = np.zeros_like(frame)

    roi_bottom_y = int(height * 0.90)
    polygon = np.array([
        [
            (int(width * 0.20), roi_bottom_y),                 # Bottom_left
            (int(width * 0.50), int(height * 0.65)),     # Top_left
            (int(width * 0.58), int(height * 0.65)),     # Top_right
            (int(width * 0.72), roi_bottom_y)                  # Bottom_right 
        ]
    ], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)

    
    roi = cv2.bitwise_and(frame, mask)

    return roi