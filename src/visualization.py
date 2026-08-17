import cv2
import numpy as np


def draw_lane_position(
    frame,
    left_lane_line,
    right_lane_line,
    reference_y
):

    lane_position_vis = frame.copy()

    height, width = frame.shape[:2]

    y_top = int(height * 0.65)
    y_bottom = int(height * 0.88)

    # LEFT LANE

    if left_lane_line is not None:

        left_a, left_b = left_lane_line

        left_top_x = int(
            np.clip(
                left_a * y_top + left_b,
                0,
                width - 1
            )
        )

        left_bottom_x = int(
            np.clip(
                left_a * y_bottom + left_b,
                0,
                width - 1
            )
        )

        cv2.line(
            lane_position_vis,
            (left_bottom_x, y_bottom),
            (left_top_x, y_top),
            (0, 255, 0),
            6
        )


    # RIGHT LANE

    if right_lane_line is not None:

        right_a, right_b = right_lane_line

        right_top_x = int(
            np.clip(
                right_a * y_top + right_b,
                0,
                width - 1
            )
        )

        right_bottom_x = int(
            np.clip(
                right_a * y_bottom + right_b,
                0,
                width - 1
            )
        )

        cv2.line(
            lane_position_vis,
            (right_bottom_x, y_bottom),
            (right_top_x, y_top),
            (0, 255, 0),
            6
        )

    return lane_position_vis

def draw_departure_warning(
    frame,
    departure_active,
    departure_direction
):

    departure_vis = frame.copy()

    if departure_active:

        # Light red overlay

        red_overlay = np.full_like(
            departure_vis,
            (0, 0, 255)
        )

        departure_vis = cv2.addWeighted(
            departure_vis,
            0.75,
            red_overlay,
            0.25,
            0
        )

        # Departure text

        cv2.putText(
            departure_vis,
            f"DEPARTURE {departure_direction}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )

    else:

        # Normal text

        cv2.putText(
            departure_vis,
            "NORMAL",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3
        )

    return departure_vis