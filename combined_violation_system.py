import cv2
from ultralytics import YOLO
import easyocr
import os
from datetime import datetime
import csv
import time
import contextlib

# === CONFIGURATION ===
TEST_MODE = True  # ✅ True = use test video, False = use live IP camera
TEST_VIDEO_PATH = "samples/red_light_violation.mp4"

IP_CAMERA_URL = "http://10.52.2.150:8080/video"  # for live camera
CONF_THRESHOLD = 0.5

# Speed detection setup (optional)
ZONE_1_Y = 250
ZONE_2_Y = 450
DISTANCE_METERS = 5  # approx distance between two lines
SPEED_LIMIT = 40  # km/h

# File storage setup
VIOLATIONS_IMG_DIR = "violations"
VIOLATIONS_VIDEO_DIR = "violations_videos"
CSV_FILE = os.path.join(os.getcwd(), "violations_log.csv")


os.makedirs(VIOLATIONS_IMG_DIR, exist_ok=True)
os.makedirs(VIOLATIONS_VIDEO_DIR, exist_ok=True)

# Create CSV if not already exists
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp", "ViolationID", "Type", "LicensePlate",
            "Image", "Video", "Speed (km/h)"
        ])

# === Initialize YOLO + OCR ===
model = YOLO("yolov8n.pt")
reader = easyocr.Reader(['en'])

# === Choose input source ===
if TEST_MODE:
    print("▶️ Running in TEST MODE using sample video.")
    cap = cv2.VideoCapture(TEST_VIDEO_PATH)
    cap = cv2.VideoCapture(TEST_VIDEO_PATH)

# Loop the video if it ends
    def reset_video(cap):
        cap.release()
        new_cap = cv2.VideoCapture(TEST_VIDEO_PATH)
        return new_cap

else:
    print("🎥 Running with IP camera stream.")
    cap = cv2.VideoCapture(IP_CAMERA_URL)

if not cap.isOpened():
    print("❌ Unable to open video/camera stream. Check file or IP.")
    exit()

# === Runtime setup ===
violation_count = 0
CLIP_DURATION = 15 # seconds
FPS = 20
speed_records = {}

# === Main Loop ===
cv2.namedWindow("Traffic Violation System", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        if not ret:
            print("🔁 Restarting test video...")
            cap = reset_video(cap)
            continue
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("🛑 Exiting system...")
            break
        

    frame = cv2.resize(frame, (800, 600))
    height, width, _ = frame.shape

    # Simulated signal for test mode
    SIGNAL_STATE = "RED" if int(time.time()) % 10 < 5 else "GREEN"

    # Draw zones and signal info
    cv2.line(frame, (0, ZONE_1_Y), (width, ZONE_1_Y), (255, 255, 0), 2)
    cv2.line(frame, (0, ZONE_2_Y), (width, ZONE_2_Y), (255, 255, 0), 2)
    color = (0, 0, 255) if SIGNAL_STATE == "RED" else (0, 255, 0)
    cv2.putText(frame, f"Signal: {SIGNAL_STATE}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # YOLO detection
    with contextlib.redirect_stdout(None):
        results = model(frame, verbose=False)

    violation_type = None
    plate_text = ""
    speed_kmph = None

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id]

            if conf > CONF_THRESHOLD and cls_name in ["car", "bus", "truck", "motorbike"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cy = (y1 + y2) // 2
                center_x = (x1 + x2) // 2

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, cls_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # === Red Light Violation ===
                if SIGNAL_STATE == "RED" and cy < ZONE_2_Y:
                    violation_type = "RED LIGHT"
                    vehicle_roi = frame[y1:y2, x1:x2]
                    plate_results = reader.readtext(vehicle_roi, detail=0)
                    if plate_results:
                        plate_text = ' '.join(plate_results)
                    break

                # === Speed Violation ===
                vehicle_id = f"{center_x}"
                now = time.time()

                if ZONE_1_Y - 10 < cy < ZONE_1_Y + 10:
                    speed_records[vehicle_id] = now
                elif ZONE_2_Y - 10 < cy < ZONE_2_Y + 10 and vehicle_id in speed_records:
                    t1 = speed_records.pop(vehicle_id)
                    time_diff = now - t1
                    speed_kmph = (DISTANCE_METERS / time_diff) * 3.6
                    if speed_kmph > SPEED_LIMIT:
                        violation_type = "SPEED"
                        plate_roi = frame[y2 - int((y2 - y1) * 0.3):y2, x1:x2]
                        plate_results = reader.readtext(plate_roi, detail=0)

                        plate_results = reader.readtext(vehicle_roi, detail=0)
                        print(f"OCR results: {plate_results}")

                        if plate_results:
                            plate_text = ' '.join(plate_results)
                        break

    cv2.imshow("Traffic Violation System", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("🛑 Exiting system...")
        break

    # === Handle Violations ===
    if violation_type:
        violation_count += 1
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        img_filename = f"{VIOLATIONS_IMG_DIR}/violation_{violation_count}_{timestamp}.jpg"
        video_filename = f"{VIOLATIONS_VIDEO_DIR}/violation_{violation_count}_{timestamp}.mp4"

        # Save image
        cv2.imwrite(img_filename, frame)
        print(f"🚨 {violation_type} violation detected! Saved image: {img_filename}")
        if plate_text:
            print(f"📃 License Plate: {plate_text}")
        if speed_kmph:
            print(f"🏎️ Speed: {speed_kmph:.2f} km/h")

        # Save short video clip
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_filename, fourcc, FPS, (width, height))
        start_time = time.time()
        while time.time() - start_time < CLIP_DURATION:
            ret, clip_frame = cap.read()
            if not ret:
                break
            clip_frame = cv2.resize(clip_frame, (800, 600))
            out.write(clip_frame)
        out.release()

        # Log data to CSV
        print(f"📄 Logging violation type: {violation_type}, plate: {plate_text}")

        with open(CSV_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, violation_count, violation_type, plate_text,
                img_filename, video_filename,
                f"{speed_kmph:.2f}" if speed_kmph else "N/A"
            ])

        print(f"✅ Logged to CSV and saved video: {video_filename}")

cap.release()
cv2.destroyAllWindows()
