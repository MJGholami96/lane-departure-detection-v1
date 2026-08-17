import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.lane_detection import fit_selected_segments, keep_parallel_to_longest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

video_path = PROJECT_ROOT / "data" / "input" / "AS3.mp4"


results_dir = PROJECT_ROOT / "results"

results_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_path = results_dir / "lane_detection_result.png"

# 1. Reading Video

cap = cv2.VideoCapture(str(video_path))

frame_number = 284

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    frame_number
)

ret, frame = cap.read()

if not ret:
    print("Error reading the video!")
    cap.release()
    exit()

cap.release()


# Basic Information

orig_image = frame.copy()

height, width = frame.shape[:2]


# 2. Preprocessing

# Grayscale

gray = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2GRAY
)


# CLAHE

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

contrast = clahe.apply(gray)


# Gaussian Blur

blurred = cv2.GaussianBlur(
    contrast,
    (5, 5),
    0
)


# Canny Edge Detection

edge_image = cv2.Canny(
    blurred,
    50,
    150
)


# Morphological Closing

structuring_element = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (5, 5)
)

closing = cv2.morphologyEx(
    edge_image,
    cv2.MORPH_CLOSE,
    structuring_element
)


# 3. ROI

mask = np.zeros_like(closing)

roi_bottom_y = int(
    height * 0.90
)

polygon = np.array(
    [
        [
            (
                int(width * 0.20),
                roi_bottom_y
            ),

            (
                int(width * 0.50),
                int(height * 0.65)
            ),

            (
                int(width * 0.58),
                int(height * 0.65)
            ),

            (
                int(width * 0.72),
                roi_bottom_y
            )
        ]
    ],
    dtype=np.int32
)

cv2.fillPoly(
    mask,
    polygon,
    255
)


# Apply ROI

roi = cv2.bitwise_and(
    closing,
    mask
)


# 4. Hough Line Detection

lines = cv2.HoughLinesP(
    roi,
    rho=1,
    theta=np.pi / 180,
    threshold=25,
    minLineLength=10,
    maxLineGap=15
)


# 5. Hough Lines Visualization

frame_with_lines = frame.copy()

if lines is not None:

    print(
        f"Number of detected lines: {len(lines)}"
    )

    for line in lines:

        x1, y1, x2, y2 = line.ravel()

        cv2.line(
            frame_with_lines,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            1
        )

else:

    print("No line is detected!")


# 6. Slope Classification

negative_slope_lines = []
positive_slope_lines = []

pos_slopes = []
neg_slopes = []

min_abs_slope = 0.40
max_abs_slope = 1.15


if lines is not None:

    for line in lines:

        x1, y1, x2, y2 = line.ravel()

        dx = x2 - x1
        dy = y2 - y1

        # Nearly vertical lines.
        if abs(dx) < 2:
            continue

        slope = dy / dx

        # Nearly horizontal lines.
        if (
            abs(slope) < min_abs_slope
            or
            abs(slope) > max_abs_slope
        ):
            continue

        if slope < 0:

            negative_slope_lines.append(
                (x1, y1, x2, y2)
            )

            neg_slopes.append(slope)

        else:

            positive_slope_lines.append(
                (x1, y1, x2, y2)
            )

            pos_slopes.append(slope)


print(
    f"Lines with a negative slope: "
    f"{len(negative_slope_lines)}"
)

print(
    f"Lines with a positive slope: "
    f"{len(positive_slope_lines)}"
)

print("\nPositive slopes:")
print(pos_slopes)

print("\nNegative slopes:")
print(neg_slopes)


# 7. Slope Groups Visualization

slope_groups_vis = frame.copy()


# Negative slope = Blue

for x1, y1, x2, y2 in negative_slope_lines:

    cv2.line(
        slope_groups_vis,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2
    )


# Positive slope = Red

for x1, y1, x2, y2 in positive_slope_lines:

    cv2.line(
        slope_groups_vis,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        2
    )


# 8. Reference Y and Segment Positions

reference_y = int(
    height * 0.70
)

negative_x_positions = []
positive_x_positions = []


# Negative slope positions

for x1, y1, x2, y2 in negative_slope_lines:

    if y2 == y1:
        continue

    x_reference = (
        x1
        +
        (reference_y - y1)
        *
        (x2 - x1)
        /
        (y2 - y1)
    )

    length = np.hypot(
        x2 - x1,
        y2 - y1
    )

    if 0 <= x_reference < width:

        negative_x_positions.append(
            (
                int(x_reference),
                length
            )
        )


# Positive slope positions

for x1, y1, x2, y2 in positive_slope_lines:

    if y2 == y1:
        continue

    x_reference = (
        x1
        +
        (reference_y - y1)
        *
        (x2 - x1)
        /
        (y2 - y1)
    )

    length = np.hypot(
        x2 - x1,
        y2 - y1
    )

    if 0 <= x_reference < width:

        positive_x_positions.append(
            (
                int(x_reference),
                length
            )
        )


# 9. Segment Position Visualization

segment_positions_vis = frame.copy()


# Reference horizontal line

cv2.line(
    segment_positions_vis,
    (0, reference_y),
    (width - 1, reference_y),
    (0, 255, 255),
    2
)


# Negative slope positions = Blue

for x, length in negative_x_positions:

    cv2.circle(
        segment_positions_vis,
        (x, reference_y),
        7,
        (255, 0, 0),
        -1
    )

# Positive slope positions = Red

for x, length in positive_x_positions:

    cv2.circle(
        segment_positions_vis,
        (x, reference_y),
        7,
        (0, 0, 255),
        -1
    )


# 10. Select Main Clusters

center_x = width / 2

cluster_tolerance = 35


# Negative / left candidates

negative_left_candidates = [
    (x, length)
    for x, length in negative_x_positions
    if x < center_x
]


if len(negative_left_candidates) > 0:

    (
        negative_cluster_center,
        negative_max_length
    ) = max(
        negative_left_candidates,
        key=lambda item: item[1]
    )

else:

    negative_cluster_center = None
    negative_max_length = None


# Positive / right candidates

positive_right_candidates = [
    (x, length)
    for x, length in positive_x_positions
    if x > center_x
]


if len(positive_right_candidates) > 0:

    (
        positive_cluster_center,
        positive_max_length
    ) = max(
        positive_right_candidates,
        key=lambda item: item[1]
    )

else:

    positive_cluster_center = None
    positive_max_length = None


# 11. Select Segments Near Main Cluster

selected_negative_lines = []
selected_positive_lines = []


# Negative slope selected segments

for x1, y1, x2, y2 in negative_slope_lines:

    if y2 == y1:
        continue

    x_reference = (
        x1
        +
        (reference_y - y1)
        *
        (x2 - x1)
        /
        (y2 - y1)
    )

    if (
        negative_cluster_center is not None
        and
        abs(
            x_reference
            -
            negative_cluster_center
        )
        <= cluster_tolerance
    ):

        selected_negative_lines.append(
            (x1, y1, x2, y2)
        )


# Positive slope selected segments

for x1, y1, x2, y2 in positive_slope_lines:

    if y2 == y1:
        continue

    x_reference = (
        x1
        +
        (reference_y - y1)
        *
        (x2 - x1)
        /
        (y2 - y1)
    )

    if (
        positive_cluster_center is not None
        and
        abs(
            x_reference
            -
            positive_cluster_center
        )
        <= cluster_tolerance
    ):

        selected_positive_lines.append(
            (x1, y1, x2, y2)
        )


# Debug Information

print(
    f"\nBlue cluster center: "
    f"{negative_cluster_center}"
)

print(
    f"Blue segment length: "
    f"{negative_max_length}"
)

print(
    f"Red cluster center: "
    f"{positive_cluster_center}"
)

print(
    f"Red segment length:"
    f"{positive_max_length}"
)

print(
    f"Selected blue segments: "
    f"{len(selected_negative_lines)}"
)

print(
    f"Selected Red segments: "
    f"{len(selected_positive_lines)}"
)


# 12. Selected Segments Visualization

selected_segments_vis = frame.copy()


# Reference line

cv2.line(
    selected_segments_vis,
    (0, reference_y),
    (width - 1, reference_y),
    (0, 255, 255),
    2
)


# Selected negative segments = Blue

for x1, y1, x2, y2 in selected_negative_lines:

    cv2.line(
        selected_segments_vis,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        4
    )


# Selected positive segments = Red

for x1, y1, x2, y2 in selected_positive_lines:

    cv2.line(
        selected_segments_vis,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        4
    )


# 13. Direction Filtering

# Apply direction filter to positive group

positive_lines_for_fit = (
    keep_parallel_to_longest(
        selected_positive_lines,
        max_angle_difference=10
    )
)


# Apply direction filter to negative group

negative_lines_for_fit = (
    keep_parallel_to_longest(
        selected_negative_lines,
        max_angle_difference=10
    )
)

# 14. Direction Filter Visualization

direction_filter_vis = frame.copy()


# All selected positive segments = Gray

for x1, y1, x2, y2 in selected_positive_lines:

    cv2.line(
        direction_filter_vis,
        (x1, y1),
        (x2, y2),
        (130, 130, 130),
        2
    )


# Direction-filtered segments = Red

for x1, y1, x2, y2 in positive_lines_for_fit:

    cv2.line(
        direction_filter_vis,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        4
    )


# 15. Fit Selected Segments

# Fit left and right boundaries

current_left_model = (
    fit_selected_segments(
        negative_lines_for_fit
    )
)


current_right_model = (
    fit_selected_segments(
        positive_lines_for_fit
    )
)


# 16. Fitted Boundaries Visualization

current_lane_fit_vis = frame.copy()


y_top = int(
    height * 0.60
)

y_bottom = roi_bottom_y


# Original selected negative segments

for x1, y1, x2, y2 in selected_negative_lines:

    cv2.line(
        current_lane_fit_vis,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2
    )


# Original selected positive segments

for x1, y1, x2, y2 in selected_positive_lines:

    cv2.line(
        current_lane_fit_vis,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        2
    )


# Fitted LEFT boundary

if current_left_model is not None:

    a, b = current_left_model

    x_top = int(
        np.clip(
            a * y_top + b,
            0,
            width - 1
        )
    )

    x_bottom = int(
        np.clip(
            a * y_bottom + b,
            0,
            width - 1
        )
    )

    cv2.line(
        current_lane_fit_vis,
        (x_bottom, y_bottom),
        (x_top, y_top),
        (255, 255, 0),
        6
    )


# Fitted RIGHT boundary

if current_right_model is not None:

    a, b = current_right_model

    x_top = int(
        np.clip(
            a * y_top + b,
            0,
            width - 1
        )
    )

    x_bottom = int(
        np.clip(
            a * y_bottom + b,
            0,
            width - 1
        )
    )

    cv2.line(
        current_lane_fit_vis,
        (x_bottom, y_bottom),
        (x_top, y_top),
        (0, 255, 255),
        6
    )


# Final Subplot

fig, axes = plt.subplots(
    4,
    3,
    figsize=(18, 25)
)


# 1. Original

axes[0, 0].imshow(
    cv2.cvtColor(
        orig_image,
        cv2.COLOR_BGR2RGB
    )
)

axes[0, 0].set_title(
    "1. Original Image"
)

axes[0, 0].axis("off")


# 2. Grayscale

axes[0, 1].imshow(
    gray,
    cmap="gray"
)

axes[0, 1].set_title(
    "2. Grayscale"
)

axes[0, 1].axis("off")

# 3. CLAHE

axes[0, 2].imshow(
    contrast,
    cmap="gray"
)

axes[0, 2].set_title(
    "3. CLAHE Contrast"
)

axes[0, 2].axis("off")


# 4. Gaussian Blur

axes[1, 0].imshow(
    blurred,
    cmap="gray"
)

axes[1, 0].set_title(
    "4. Gaussian Blur"
)

axes[1, 0].axis("off")


# 5. Canny

axes[1, 1].imshow(
    edge_image,
    cmap="gray"
)

axes[1, 1].set_title(
    "5. Canny Edges"
)

axes[1, 1].axis("off")


# 6. Morphological Closing

axes[1, 2].imshow(
    closing,
    cmap="gray"
)

axes[1, 2].set_title(
    "6. Morphological Closing"
)

axes[1, 2].axis("off")


# 7. ROI

axes[2, 0].imshow(
    roi,
    cmap="gray"
)

axes[2, 0].set_title(
    "7. ROI Masked"
)

axes[2, 0].axis("off")


# 8. Hough Lines

axes[2, 1].imshow(
    cv2.cvtColor(
        frame_with_lines,
        cv2.COLOR_BGR2RGB
    )
)

axes[2, 1].set_title(
    f"8. Hough Lines "
    f"({len(lines) if lines is not None else 0})"
)

axes[2, 1].axis("off")


# 9. Slope Groups

axes[2, 2].imshow(
    cv2.cvtColor(
        slope_groups_vis,
        cv2.COLOR_BGR2RGB
    )
)

axes[2, 2].set_title(
    "9. Slope Groups"
)

axes[2, 2].axis("off")


# 10. Segment Positions

axes[3, 0].imshow(
    cv2.cvtColor(
        segment_positions_vis,
        cv2.COLOR_BGR2RGB
    )
)

axes[3, 0].set_title(
    "10. Segment Positions"
)

axes[3, 0].axis("off")


# 11. Selected Segments

axes[3, 1].imshow(
    cv2.cvtColor(
        selected_segments_vis,
        cv2.COLOR_BGR2RGB
    )
)

axes[3, 1].set_title(
    "11. Selected Segments"
)

axes[3, 1].axis("off")


# 12. Fitted Boundaries

axes[3, 2].imshow(
    cv2.cvtColor(
        current_lane_fit_vis,
        cv2.COLOR_BGR2RGB
    )
)

axes[3, 2].set_title(
    "12. Fitted Boundaries of the Current Lane"
)

axes[3, 2].axis("off")


plt.tight_layout()

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print(
    f"\nFinal result saved to:\n{output_path}"
)

plt.show()