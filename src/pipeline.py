from .preprocessing import (
    to_grayscale,
    apply_clahe,
    apply_blur,
    apply_canny,
    apply_morphological_closing,
    apply_roi
)

from .lane_detection import (
    detect_lines,
    classify_lines_by_slope,
    get_x_positions,
    select_lane_segments,
    keep_parallel_to_longest,
    fit_selected_segments
)

from .lane_departure import (
    calculate_lane_offset,
    update_lane_state
)

from .visualization import (
    draw_lane_position,
    draw_departure_warning
)


def preprocess_frame(frame):

    gray = to_grayscale(frame)

    clahe = apply_clahe(gray)

    blurred = apply_blur(clahe)

    edges = apply_canny(blurred)

    closed = apply_morphological_closing(edges)

    roi = apply_roi(closed)

    return roi



def detect_lane(roi):

    height, width = roi.shape[:2]

    # Reference position used for lane selection
    reference_y = int(height * 0.72)

    # Detect Hough line segments
    lines = detect_lines(roi)

    # Classify line segments by slope
    negative_slope_lines, positive_slope_lines = (
        classify_lines_by_slope(lines)
    )

    # Get x-positions of the line segments at reference_y
    negative_x_positions = get_x_positions(
        negative_slope_lines,
        reference_y,
        width
    )

    positive_x_positions = get_x_positions(
        positive_slope_lines,
        reference_y,
        width
    )

    # Select lane-related segments
    (
        selected_negative_lines,
        selected_positive_lines,
        _,
        _
    ) = select_lane_segments(
        negative_slope_lines,
        positive_slope_lines,
        reference_y,
        width
    )

    # Keep segments parallel to the longest segment
    negative_lane_segments = keep_parallel_to_longest(
        selected_negative_lines
    )

    positive_lane_segments = keep_parallel_to_longest(
        selected_positive_lines
    )

    # Fit final lane lines
    left_lane_line = fit_selected_segments(
        negative_lane_segments or []
    )

    right_lane_line = fit_selected_segments(
        positive_lane_segments or []
    )

    return (
        left_lane_line,
        right_lane_line,
        reference_y
    )


def detect_lane_departure(
    left_lane_line,
    right_lane_line,
    reference_y,
    width,
    lane_state
):

    # Calculate normalized vehicle offset
    normalized_offset = calculate_lane_offset(
        left_lane_line,
        right_lane_line,
        reference_y,
        width
    )

    # Update lane departure state
    departure_status = update_lane_state(
        normalized_offset,
        left_lane_line,
        right_lane_line,
        lane_state
    )

    return (
        normalized_offset,
        departure_status,
        lane_state
    )


def visualize_result(
    frame,
    left_lane_line,
    right_lane_line,
    reference_y,
    normalized_offset,
    departure_active,
    departure_direction,
    departure_status
):

    # The Normal / Departure status and overlay.
    departure_vis = draw_departure_warning(
        frame,
        departure_active,
        departure_direction
    )

    # Then draw the detected lanes (Solid Green)
    result = draw_lane_position(
        departure_vis,
        left_lane_line,
        right_lane_line,
        reference_y
    )

    return result