
import cv2
from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolov8n.pt")

# Input CCTV video
video_path = "input/cctv_video.mp4"

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Unable to open video")
    exit()

print("ByteTrack Person Tracking Started")

while True:

    # Read video frame
    ret, frame = cap.read()

    if not ret:
        break

    # Detect and track people using ByteTrack
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=0.35,
        verbose=False
    )

    # Get tracking result
    result = results[0]

    # Draw bounding boxes and tracking IDs
    annotated_frame = result.plot()

    # Get current tracking IDs
    if result.boxes is not None and result.boxes.id is not None:

        track_ids = result.boxes.id.int().cpu().tolist()

        # Display current tracked people count
        cv2.putText(
            annotated_frame,
            f"Tracked People: {len(track_ids)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    else:
        track_ids = []

    # Display video
    cv2.imshow("ByteTrack Person Tracking", annotated_frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Person Tracking Completed")
