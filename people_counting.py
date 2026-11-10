
import cv2
import csv
import os
from ultralytics import YOLO

# ==========================================
# 1. LOAD YOLO MODEL
# ==========================================

model = YOLO("yolov8n.pt")

# ==========================================
# 2. INPUT AND OUTPUT PATHS
# ==========================================

video_path = "input/cctv_video.mp4"

output_video = "output/people_counting_output.mp4"
output_csv = "output/people_events.csv"

os.makedirs("output", exist_ok=True)

# ==========================================
# 3. OPEN CCTV VIDEO
# ==========================================

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Unable to open CCTV video")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 25

# Create output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    raise RuntimeError("Unable to create output video")

# ==========================================
# 4. ENTRY / EXIT LINE
# ==========================================

line_start = (745, 305)
line_end = (635, 475)

# Ignore small movements near the line
line_margin = 8

# ==========================================
# 5. INITIALIZE COUNTERS
# ==========================================

total_entered = 0
total_exited = 0
headcount = 0

previous_side = {}

frame_number = 0

# ==========================================
# 6. FIND PERSON'S SIDE OF THE LINE
# ==========================================

def get_side(point, start, end, margin):

    x, y = point
    x1, y1 = start
    x2, y2 = end

    line_length = (
        (x2 - x1) ** 2 + (y2 - y1) ** 2
    ) ** 0.5

    if line_length == 0:
        return 0

    distance = (
        (x2 - x1) * (y - y1)
        - (y2 - y1) * (x - x1)
    ) / line_length

    if distance > margin:
        return 1

    elif distance < -margin:
        return -1

    return 0


# ==========================================
# 7. FORMAT VIDEO TIMESTAMP
# ==========================================

def format_timestamp(seconds):

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    return (
        f"{hours:02d}:{minutes:02d}:"
        f"{secs:06.3f}"
    )


# ==========================================
# 8. PROCESS VIDEO AND SAVE CSV
# ==========================================

print("People Counting Started")

with open(output_csv, "w", newline="") as file:

    csv_writer = csv.DictWriter(
        file,
        fieldnames=[
            "timestamp",
            "person_id",
            "event",
            "headcount"
        ]
    )

    csv_writer.writeheader()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        timestamp = format_timestamp(frame_number / fps)

        # YOLO + ByteTrack
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=0.35,
            verbose=False
        )

        result = results[0]

        # Draw entry/exit line
        cv2.line(
            frame,
            line_start,
            line_end,
            (0, 255, 255),
            3
        )

        # Check detected people
        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):

            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):

                x1, y1, x2, y2 = map(int, box)

                # Bottom-center point
                center_x = (x1 + x2) // 2
                bottom_y = y2

                point = (center_x, bottom_y)

                # Determine current side
                current_side = get_side(
                    point,
                    line_start,
                    line_end,
                    line_margin
                )

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Display person ID
                cv2.putText(
                    frame,
                    f"ID: {track_id}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # Draw bottom-center point
                cv2.circle(
                    frame,
                    point,
                    5,
                    (0, 0, 255),
                    -1
                )

                # Entry / exit logic
                if current_side != 0:

                    if track_id not in previous_side:

                        previous_side[track_id] = current_side

                    else:

                        old_side = previous_side[track_id]

                        if old_side != current_side:

                            event = None

                            # Top to bottom = IN
                            if old_side == -1 and current_side == 1:

                                total_entered += 1
                                headcount += 1
                                event = "IN"

                            # Bottom to top = OUT
                            elif old_side == 1 and current_side == -1:

                                total_exited += 1
                                headcount = max(0, headcount - 1)
                                event = "OUT"

                            # Save crossing event
                            if event is not None:

                                csv_writer.writerow({
                                    "timestamp": timestamp,
                                    "person_id": track_id,
                                    "event": event,
                                    "headcount": headcount
                                })

                                print(
                                    f"{timestamp} | "
                                    f"Person {track_id} | "
                                    f"{event} | "
                                    f"Inside: {headcount}"
                                )

                        previous_side[track_id] = current_side

        # ==================================
        # 9. DISPLAY LIVE COUNTS
        # ==================================

        cv2.putText(
            frame,
            f"ENTERED: {total_entered}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"EXITED: {total_exited}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"INSIDE: {headcount}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        # Save processed frame to output video
        out.write(frame)

        # Display processed video
        cv2.imshow("People Counting System", frame)

        frame_number += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# ==========================================
# 10. RELEASE RESOURCES
# ==========================================

cap.release()
out.release()
cv2.destroyAllWindows()

print("\nProcessing Completed!")
print("Total Entered:", total_entered)
print("Total Exited:", total_exited)
print("People Inside:", headcount)

print("Output Video:", output_video)
print("CSV File:", output_csv)
