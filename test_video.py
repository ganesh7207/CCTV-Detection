
import cv2
import os

# Video file location
video_path = "input/cctv_video.mp4"

# Check whether video exists
if not os.path.exists(video_path):
    print("Video file not found!")
else:
    # Open CCTV video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Unable to open video")
    else:
        # Get video information
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        duration = frame_count / fps if fps > 0 else 0

        print("Video opened successfully!")
        print("FPS:", fps)
        print("Total Frames:", frame_count)
        print("Width:", width)
        print("Height:", height)
        print("Duration:", duration, "seconds")

    cap.release()
