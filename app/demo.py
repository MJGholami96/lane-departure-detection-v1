import cv2
from PIL import Image
from pathlib import Path

from src.lane_departure import create_lane_state

from src.pipeline import (
    preprocess_frame,
    detect_lane,
    detect_lane_departure,
    visualize_result
)


# Input videos

PROJECT_ROOT = Path(__file__).resolve().parent.parent

input_dir = PROJECT_ROOT / "data" / "input"

video_paths = [
    input_dir / "AS2.mp4",
    input_dir / "AS3.mp4",
    input_dir / "AS4.mp4"
]


# Output GIF

results_dir = PROJECT_ROOT / "results"
results_dir.mkdir(exist_ok=True)

gif_path = results_dir / "demo.gif"


# GIF settings

# Every N frames, add one frame to the GIF.
frame_step = 3

# GIF width
gif_width = 640

# Frame display duration in milliseconds.
gif_duration = 100



# GIF frames
gif_frames = []


# Process all videos

for video_index, video_path in enumerate(video_paths, start=1):

    print("\n==========================================")
    print(f"Processing video {video_index}:")
    print(video_path)
    print("==========================================")


    # Open video

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        print("Could not open video.")
        continue


    # IMPORTANT:
    # New lane state for each video

    lane_state = create_lane_state()


    # Frame counter
    frame_num = 0


    # Read video

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


        frame_num += 1


        # Add frame to ONE common GIF

        if frame_num % frame_step == 0:

            # BGR -> RGB
            rgb = cv2.cvtColor(
                result,
                cv2.COLOR_BGR2RGB
            )


            # Resize
            gif_height = int(
                height * gif_width / width
            )


            rgb = cv2.resize(
                rgb,
                (gif_width, gif_height)
            )


            gif_frame = Image.fromarray(rgb)

            gif_frames.append(gif_frame)


        # Display

        cv2.imshow(
            "Lane Detection",
            result
        )


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    # Close current video

    cap.release()

    cv2.destroyAllWindows()


# Save ONE combined GIF

if len(gif_frames) > 0:

    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=gif_duration,
        loop=0
    )


    print("\n==========================================")
    print("Combined GIF saved:")
    print(gif_path)
    print("==========================================")


else:

    print("\nNo frames were collected.")