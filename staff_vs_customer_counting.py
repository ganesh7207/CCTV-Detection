
import cv2
import csv
import os
from ultralytics import YOLO

# ==========================================
# 1. LOAD MODEL AND VIDEO
# ==========================================

model = YOLO("yolov8n.pt")

video_path = "input/cctv_video.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Cannot open input/cctv_video.mp4")

fps = cap.get(cv2.CAP_PROP_FPS) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video opened successfully!")
print("Width:", width)
print("Height:", height)
print("FPS:", fps)
print("Total Frames:", total_frames)

# ==========================================
# 2. CREATE OUTPUT FOLDER AND VIDEO
# ==========================================

os.makedirs("output", exist_ok=True)

output_video = "output/staff_customer_counting_output.mp4"
output_csv = "output/people_events_with_roles.csv"

out = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not out.isOpened():
    raise RuntimeError("Cannot create output video!")

# ==========================================
# 3. VIRTUAL ENTRY / EXIT LINE
# ==========================================

line_start = (745, 305)
line_end = (635, 475)

line_margin = 8

# ==========================================
# 4. STAFF WORKING AREA
# ==========================================

STAFF_COUNTER_X = 115

# ==========================================
# 5. INITIALIZE COUNTERS
# ==========================================

entered = 0
exited = 0
headcount = 0

previous_side = {}

staff_ids = set()

frame_idx = 0

# ==========================================
# 6. FUNCTION TO FIND SIDE OF LINE
# ==========================================

def get_side(pt, p1, p2, margin):

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    length = (dx * dx + dy * dy) ** 0.5 or 1

    dist = (
        dx * (pt[1] - p1[1])
        - dy * (pt[0] - p1[0])
    ) / length

    return 1 if dist > margin else (-1 if dist < -margin else 0)


# ==========================================
# 7. PROCESS VIDEO AND GENERATE CSV
# ==========================================

print("\nProcessing CCTV Video...")

with open(output_csv, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "timestamp",
        "person_id",
        "role",
        "event",
        "headcount"
    ])

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # Calculate video timestamp
        sec = frame_idx / fps

        timestamp = (
            f"{int(sec // 3600):02d}:"
            f"{int((sec % 3600) // 60):02d}:"
            f"{sec % 60:06.3f}"
        )

        # ==================================
        # 8. YOLO DETECTION + BYTETRACK
        # ==================================

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=0.35,
            verbose=False
        )

        res = results[0]

        # Draw yellow entrance / exit line
        cv2.line(
            frame,
            line_start,
            line_end,
            (0, 255, 255),
            3
        )

        staff_count = 0
        customer_count = 0

        # ==================================
        # 9. DETECT AND TRACK PEOPLE
        # ==================================

        if res.boxes is not None and res.boxes.id is not None:

            boxes = res.boxes.xyxy.cpu().numpy()

            ids = res.boxes.id.int().cpu().tolist()

            for box, tid in zip(boxes, ids):

                x1, y1, x2, y2 = map(int, box)

                # Bottom-center point
                pt = (
                    (x1 + x2) // 2,
                    y2
                )

                # ==================================
                # 10. STAFF VS CUSTOMER CLASSIFICATION
                # ==================================

                # Identify staff near the left counter
                if (
                    pt[0] <= STAFF_COUNTER_X
                    and 100 <= pt[1] <= 460
                ):
                    staff_ids.add(tid)

                is_staff = tid in staff_ids

                if is_staff:

                    staff_count += 1

                    color = (0, 165, 255)

                    label = f"ID: {tid} [STAFF]"

                else:

                    customer_count += 1

                    color = (0, 255, 0)

                    label = f"ID: {tid} [CUSTOMER]"

                # ==================================
                # 11. DRAW BOXES AND TRACKING IDs
                # ==================================

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 6, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

                cv2.circle(
                    frame,
                    pt,
                    4,
                    (0, 0, 255),
                    -1
                )

                # ==================================
                # 12. ENTRY / EXIT COUNTING
                # ==================================

                side = get_side(
                    pt,
                    line_start,
                    line_end,
                    line_margin
                )

                if side != 0:

                    if tid not in previous_side:

                        previous_side[tid] = side

                    elif previous_side[tid] != side:

                        event = None

                        old_side = previous_side[tid]

                        # Person entering
                        if old_side == -1 and side == 1:

                            event = "IN"

                            if not is_staff:
                                entered += 1
                                headcount += 1

                        # Person exiting
                        elif old_side == 1 and side == -1:

                            event = "OUT"

                            if not is_staff:
                                exited += 1
                                headcount = max(0, headcount - 1)

                        # Save event in CSV
                        if event:

                            role = (
                                "STAFF"
                                if is_staff
                                else "CUSTOMER"
                            )

                            writer.writerow([
                                timestamp,
                                tid,
                                role,
                                event,
                                headcount
                            ])

                            print(
                                f"{timestamp} | "
                                f"{role} {tid} | "
                                f"{event} | "
                                f"Headcount: {headcount}"
                            )

                        previous_side[tid] = side

        # ==================================
        # 13. DISPLAY LIVE COUNTERS
        # ==================================

        cv2.putText(
            frame,
            f"ENTERED: {entered}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"EXITED: {exited}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"INSIDE: {headcount}",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"STAFF: {staff_count}",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 165, 255),
            2
        )

        cv2.putText(
            frame,
            f"CUSTOMERS: {customer_count}",
            (20, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # ==================================
        # 14. DISPLAY LIVE VIDEO  [ADDED]
        # ==================================

        cv2.imshow(
            "Staff vs Customer People Counting",
            frame
        )

        # ==================================
        # 15. SAVE PROCESSED VIDEO
        # ==================================

        out.write(frame)

        # Press Q to stop processing
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Video processing stopped by user.")
            break

        frame_idx += 1

        if frame_idx % 250 == 0:

            print(
                f"Processed {frame_idx}/{total_frames} frames..."
            )


# ==========================================
# 16. RELEASE RESOURCES  [UPDATED]
# ==========================================

cap.release()
out.release()
cv2.destroyAllWindows()


# ==========================================
# 17. FINAL RESULTS
# ==========================================

print("\n========== FINAL RESULTS ==========")

print("Total Entered:", entered)
print("Total Exited:", exited)
print("People Inside:", headcount)

print("Output Video:", output_video)
print("CSV Report:", output_csv)

print("\nPeople Counting Completed Successfully!")
