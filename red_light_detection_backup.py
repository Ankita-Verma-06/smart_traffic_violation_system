import cv2
from ultralytics import YOLO
import easyocr
import os
from datetime import datetime
import serial
import time
import contextlib
import csv

# === CONFIGURATION ===
IP_CAMERA_URL = "http://192.168.1.102:8080/video"
LINE_Y = 350
CONF_THRESHOLD = 0.5
VIOLATIONS_IMG_DIR = "violations"
VIOLATIONS_VIDEO_DIR = "violations_videos"
CSV_FILE = "violations_log.csv"

os.makedirs(VIOLATIONS_IMG_DIR, exist_ok=True)
os.makedirs(VIOLATIONS_VIDEO_DIR, exist_ok=True)

SERIAL_PORT = "/dev/tty.usbmodem14101"
BAUD_RATE = 9600

# YOLO + OCR
model = YOLO("yolov8n.pt")
reader = easyocr.Reader(['en'])

# Open CSV for logging
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "ViolationID", "LicensePlate", "Image", "Video"])

cap = cv2.VideoCapture(IP_CAMERA_URL)
if not cap.isOpened():
    print("❌ Unable to open camera stream.")
    exit()

try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
except:
    arduino = None

violation_count = 0
CLIP_DURATION = 3  # seconds
FPS = 20

def get_signal_state():
    if arduino:
        try:
            arduino.write(b'STATE\n')
            line = arduino.readline().decode().strip()
            if line in ["RED", "GREEN"]:
                return line
        except:
            return "RED"
    return "RED"

cv2.namedWindow("Red Light Violation Detection", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (800, 600))
    height, width, _ = frame.shape

    SIGNAL_STATE = get_signal_state()

    # Draw stop line
    color = (0, 0, 255) if SIGNAL_STATE == "RED" else (0, 255, 0)
    cv2.line(frame, (0, LINE_Y), (width, LINE_Y), color, 3)
    cv2.putText(frame, f"Signal: {SIGNAL_STATE}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # Run YOLO silently
    with contextlib.redirect_stdout(None):
        results = model(frame, verbose=False)

    violation_detected = False
    plate_text = ""

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id]

            if conf > CONF_THRESHOLD and cls_name in ["car", "bus", "truck", "motorbike"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cy = (y1 + y2) // 2

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, cls_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if SIGNAL_STATE == "RED" and cy < LINE_Y:
                    violation_detected = True
                    # Crop vehicle region for plate detection
                    vehicle_roi = frame[y1:y2, x1:x2]
                    plate_results = reader.readtext(vehicle_roi, detail=0)
                    if plate_results:
                        plate_text = ' '.join(plate_results)
                    break  # only record first violating vehicle per frame

    cv2.imshow("Red Light Violation Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if violation_detected:
        violation_count += 1
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        img_filename = f"{VIOLATIONS_IMG_DIR}/violation_{violation_count}_{timestamp}.jpg"
        video_filename = f"{VIOLATIONS_VIDEO_DIR}/violation_{violation_count}_{timestamp}.mp4"

        # Save snapshot
        cv2.imwrite(img_filename, frame)
        print(f"🚨 Violation detected! Saved image: {img_filename}")
        if plate_text:
            print(f"📃 License Plate: {plate_text}")

        # Record short video
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
        print(f"🎬 Video saved: {video_filename}")

        # Log violation to CSV
        with open(CSV_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, violation_count, plate_text, img_filename, video_filename])

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
