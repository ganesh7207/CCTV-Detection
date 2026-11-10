# People Counting System Using YOLO and ByteTrack

## 1. Project Overview

This project is a computer vision-based people counting system that processes CCTV footage to detect and track people entering and exiting a store.

The system uses YOLO for person detection and ByteTrack for tracking individuals across video frames.

It identifies entry and exit events using a virtual counting line and calculates the live number of people inside the store.

## 2. Technologies Used

- Python – Main programming language
- YOLOv8 – Person detection
- ByteTrack – Multi-object tracking
- OpenCV – Video processing and visualization
- NumPy – Numerical operations
- CSV – Event data storage

## 3. Project Structure

```text
People_Counting_System/
│
├── input/
│   └── cctv_video.mp4
│
├── output/
│   ├── people_counting_output.mp4
│   └── people_events.csv
│
├── yolov8n.pt
├── test_video.py
├── person_detection.py
├── person_tracking.py
├── people_counting.py
├── staff_vs_customer_counting.py
├── requirements.txt
└── README.md
```

## 4. Installation

### Step 1: Install Python

Install Python 3.10 or a compatible version.

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

### Step 3: Activate the environment on Windows

```bash
.venv\Scripts\activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Input Video

Place the CCTV video inside the `input` folder.

Use the filename:

```text
cctv_video.mp4
```

If your video has a different filename, update the video path in `people_counting.py`.

## 6. Running the Project

Execute the following command from the project directory:

```bash
python people_counting.py
```

### Option B: Run Staff vs. Customer Classification
```bash
python staff_vs_customer_counting.py
```

The application will:

1. Read the CCTV video.
2. Detect people using YOLOv8.
3. Track detected people using ByteTrack.
4. Identify entry and exit line crossings.
5. Calculate the live headcount.
6. Generate the processed video.
7. Save entry and exit events in a CSV file.

## 7. Output Files

### Output Video

`output/people_counting_output.mp4`

The processed video displays:
- Person bounding boxes
- Tracking IDs
- Entry/exit line
- Total entered count
- Total exited count
- Current headcount

### CSV File

`output/people_events.csv`

The CSV contains the following columns:

| Column | Description |
|---|---|
| timestamp | Time of the crossing event |
| person_id | Unique tracking ID |
| event | IN or OUT |
| headcount | Number of people inside after the event |

## 8. Counting Logic

A virtual line is configured across the store entrance.

The bottom-center point of each detected person's bounding box is used to determine their position relative to the line.

When a tracked person crosses the line:

- Entry direction: Increase the entered count.
- Exit direction: Increase the exited count and decrease the headcount.

The headcount is calculated as:

**Current Headcount = Total Entered - Total Exited**

The line coordinates and direction must be configured according to the CCTV camera view.

## 9. Model Information

The project uses the pretrained YOLOv8n model for person detection.

ByteTrack is used through the Ultralytics tracking interface.

Only the person class is selected for detection and tracking.

The pretrained model does not require training from scratch.

## 10. Limitations and Future Improvements

- Occlusion can affect detection and tracking accuracy.
- People entering together may cause missed or incorrect crossings.
- Tracking IDs can change when a person disappears for a long time.
- Staff and customers are not automatically classified in the current implementation.
- Initial occupancy is assumed to be zero.
- Counting accuracy depends on camera position and counting-line configuration.

Future improvements include better tracking recovery, improved group counting, staff classification, and more robust crossing confirmation.

## 11. Conclusion

This project demonstrates a computer vision-based approach to people counting using object detection, multi-object tracking, and line-crossing logic.

It generates an annotated CCTV video and a CSV event log containing timestamps, tracking IDs, entry/exit events, and headcount information.