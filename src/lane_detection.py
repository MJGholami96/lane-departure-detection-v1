import cv2
import numpy as np


# Detects line segments in the ROI using the Probabilistic Hough Transform
def detect_lines(
    roi,
    rho=1,
    theta=np.pi / 180,
    threshold=25,
    min_line_length=10,
    max_line_gap=15
):
    lines = cv2.HoughLinesP(
    roi,
    rho=rho,
    theta=theta,
    threshold=threshold,
    minLineLength=min_line_length,
    maxLineGap=max_line_gap
)
    return lines

# Filters and classifies Hough lines based on their slope.
def classify_lines_by_slope(lines, min_abs_slope=0.4,  max_abs_slope = 2.0):

    negative_slope_lines = []
    positive_slope_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.ravel()

            dx = x2 - x1
            dy = y2 - y1

            # Ignore nearly vertical lines
            if abs(dx) < 2:
                continue

            slope = dy / dx

            # Ignore nearly horizontal lines
            if abs(slope) < min_abs_slope or abs(slope) > max_abs_slope:
                continue

            if slope < 0:
                negative_slope_lines.append((x1, y1, x2, y2))
            else:
                positive_slope_lines.append((x1, y1, x2, y2))

    return negative_slope_lines, positive_slope_lines

# Finds the x-position of line segments at a fixed height (reference_y).
def get_x_positions(lines, reference_y, width):

    x_positions = []

    for x1, y1, x2, y2 in lines:

        if y1 == y2:
            continue

        x = x1 + (reference_y - y1) * (x2 - x1) / (y2 - y1)
        length = np.hypot(x2 - x1, y2 - y1)
        if 0 <= x < width:
            x_positions.append((int(x), length))

    return x_positions

# Selects segments belonging to the two main clusters.
def select_lane_segments(
    negative_slope_lines,
    positive_slope_lines,
    reference_y,
    width
):

    center_x = width / 2

    # Negative slope
    negative_candidates = []

    for line in negative_slope_lines:

        x1, y1, x2, y2 = map(int, line)

        if y1 == y2:
            continue

        x_reference = (
            x1
            + (reference_y - y1)
            * (x2 - x1)
            / (y2 - y1)
        )

        if 0 <= x_reference < center_x:

            length = np.hypot(
                x2 - x1,
                y2 - y1
            )

            negative_candidates.append(
                (
                    length,
                    x_reference,
                    (x1, y1, x2, y2)
                )
            )

    if negative_candidates:

        negative_best = max(
            negative_candidates,
            key=lambda item: item[0]
        )

        negative_max_length = negative_best[0]
        negative_cluster_center = negative_best[1]

        selected_negative_lines = [
            negative_best[2]
        ]

    else:

        negative_max_length = None
        negative_cluster_center = None
        selected_negative_lines = []


    # Positive slope
    positive_candidates = []

    for line in positive_slope_lines:

        x1, y1, x2, y2 = map(int, line)

        if y1 == y2:
            continue

        x_reference = (
            x1
            + (reference_y - y1)
            * (x2 - x1)
            / (y2 - y1)
        )

        if center_x < x_reference < width:

            length = np.hypot(
                x2 - x1,
                y2 - y1
            )

            positive_candidates.append(
                (
                    length,
                    x_reference,
                    (x1, y1, x2, y2)
                )
            )

    if positive_candidates:

        positive_best = max(
            positive_candidates,
            key=lambda item: item[0]
        )

        positive_max_length = positive_best[0]
        positive_cluster_center = positive_best[1]

        selected_positive_lines = [
            positive_best[2]
        ]

    else:

        positive_max_length = None
        positive_cluster_center = None
        selected_positive_lines = []


    return (
        selected_negative_lines,
        selected_positive_lines,
        negative_cluster_center,
        positive_cluster_center
    )


# Filter segment directions before fitting
def keep_parallel_to_longest(segments, max_angle_difference=10):
    """
    Keeps only segments whose direction is close
    to the direction of the longest segment.
    """
    if not segments:
        return None

    lengths = []

    for x1, y1, x2, y2 in segments:
        length = np.hypot(x2 - x1, y2 - y1)
        lengths.append(length)

    longest_index = int(np.argmax(lengths))
    x1, y1, x2, y2 = segments[longest_index]

    # % 180 makes the order of the segment endpoints irrelevant
    reference_angle = (
        np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
    )

    filtered_segments = []

    for x1, y1, x2, y2 in segments:

        angle = (
            np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        )

        angle_difference = abs(angle - reference_angle)
        angle_difference = min(angle_difference, 180 - angle_difference)

        if angle_difference <= max_angle_difference:
            filtered_segments.append((x1, y1, x2, y2))

    return filtered_segments

# Fits a line model to the selected lane segments.
def fit_selected_segments(segments):
    """
    Fits the selected segments using the model x = a*y + b.
    """
    if len(segments) == 0:
        return None

    points = []

    for x1, y1, x2, y2 in segments:
        points.append((x1, y1))
        points.append((x2, y2))

    points = np.array(points)

    x = points[:, 0]
    y = points[:, 1]

    a, b = np.polyfit(y, x, 1)

    return a, b


