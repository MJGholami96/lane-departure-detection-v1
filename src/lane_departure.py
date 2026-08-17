def create_lane_state():
    return {
        "departure_active": False,
        "departure_direction": None,

        # Number of frames in which no lane was detected
        "missing_lane_frames": 0,

        # Number of consecutive frames in which both lanes were detected
        "both_lanes_detected_frames": 0,

        # Last valid direction
        "last_lane_direction": None
    }



# Calculate vehicle position relative to the current lane
def calculate_lane_offset(
    left_lane_line,
    right_lane_line,
    reference_y,
    width
):

    vehicle_reference_x = width / 2

    normalized_offset = None

    if left_lane_line is not None and right_lane_line is not None:

        left_a, left_b = left_lane_line
        right_a, right_b = right_lane_line

        left_x = left_a * reference_y + left_b
        right_x = right_a * reference_y + right_b

        lane_center_x = (left_x + right_x) / 2
        lane_half_width = (right_x - left_x) / 2

        if lane_half_width > 0:

            normalized_offset = (
                vehicle_reference_x - lane_center_x
            ) / lane_half_width

    return normalized_offset



# Thresholds
MISSING_LANE_FRAMES_THRESHOLD = 5

BOTH_LANES_DETECTED_FRAMES_THRESHOLD = 8



# Lane departure state machine
def update_lane_state(
    normalized_offset,
    left_lane_line,
    right_lane_line,
    lane_state
):

    # Detection status

    both_lanes_missing = (
        left_lane_line is None
        and
        right_lane_line is None
    )

    both_lanes_detected = (
        left_lane_line is not None
        and
        right_lane_line is not None
    )


    # If no lane is detected.

    if both_lanes_missing:

        lane_state["missing_lane_frames"] += 1

        # Because both lanes were not detected, the counter for returning to Normal is reset to zero.
        lane_state["both_lanes_detected_frames"] = 0

    else:

        # At least one lane is detected.
        lane_state["missing_lane_frames"] = 0


    
    # If both lanes are detected.
    
    if both_lanes_detected:

        lane_state["both_lanes_detected_frames"] += 1

    else:

        # Both lanes must be detected consecutively to return to Normal.
        lane_state["both_lanes_detected_frames"] = 0


    # Determine the direction based on the offset.

    if normalized_offset is not None:

        if normalized_offset > 0:

            lane_state["last_lane_direction"] = "RIGHT"

        elif normalized_offset < 0:

            lane_state["last_lane_direction"] = "LEFT"


    # NORMAL STATE

    if not lane_state["departure_active"]:

        # If both lanes are absent for 5 consecutive frames.
        if lane_state["missing_lane_frames"] >= 5:

            lane_state["departure_active"] = True

            # Departure direction
            if lane_state["last_lane_direction"] is not None:

                lane_state["departure_direction"] = (
                    lane_state["last_lane_direction"]
                )

            else:

                # If there is still no valid direction.
                lane_state["departure_direction"] = "LEFT"


            return (
                f"DEPARTURE "
                f"{lane_state['departure_direction']}"
            )


        return "NORMAL"


    # DEPARTURE STATE

    # Return to NORMAL only when both lanes are detected for 3 consecutive frames.
    if lane_state["both_lanes_detected_frames"] >= 3:

        lane_state["departure_active"] = False

        lane_state["departure_direction"] = None

        lane_state["missing_lane_frames"] = 0

        lane_state["both_lanes_detected_frames"] = 0

        return "NORMAL"


    # Departure is still active.
    return (
        f"DEPARTURE "
        f"{lane_state['departure_direction']}"
    )