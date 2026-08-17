import cv2

from pathlib import Path
from src.lane_departure import create_lane_state

from src.pipeline import (
    preprocess_frame,
    detect_lane,
    detect_lane_departure,
    visualize_result
)


# Video

PROJECT_ROOT = Path(__file__).resolve().parent.parent

video_path = PROJECT_ROOT / "data" / "input" / "AS3.mp4"

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError("Could not open video.")

output_dir = PROJECT_ROOT / "results"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "result.mp4"


fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    str(output_path),
    fourcc,
    fps,
    (width, height)
)


# Lane state

lane_state = create_lane_state()


# Video processing
frame_num = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break

    height, width = frame.shape[:2]

    # 1. Preprocessing
    roi = preprocess_frame(frame)

    # 2. Lane detection
    (
        left_lane_line,
        right_lane_line,
        reference_y
    ) = detect_lane(roi)
    

    # 3. Lane departure
    (
        normalized_offset,
        departure_status,
        lane_state
    ) = detect_lane_departure(
        left_lane_line,
        right_lane_line,
        reference_y,
        width,
        lane_state
    )

    # 4. Visualization
    result = visualize_result(
        frame,
        left_lane_line,
        right_lane_line,
        reference_y,
        normalized_offset,
        lane_state["departure_active"],
        lane_state["departure_direction"],
        departure_status
    )
    frame_num +=1
    

    out.write(result)
    # Display result
    cv2.imshow("Lane Detection", result)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()