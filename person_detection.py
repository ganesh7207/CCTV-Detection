
import cv2
from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolov8n.pt")

# Input CCTV video
video_path = "input/cctv_video.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Unable to open video")
    exit()

print("YOLO Person Detection Started")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Detect only people (COCO class ID 0)
    results = model(frame, classes=[0], conf=0.35, verbose=False)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Display video
    cv2.imshow("YOLO Person Detection", annotated_frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Detection Completed")
