import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import os
from datetime import datetime

# === CONFIGURATION ===
IP_CAMERA_URL = "http://10.52.2.150:8080/video"  # <-- replace with your phone's IP link
CONF_THRESHOLD = 0.5
SIGNAL_STATE = "RED"   # can be RED or GREEN manually, or automated later
LINE_Y = 350            # position of stop line (adjust based on your camera angle)

# === SETUP ===
os.makedirs("violations", exist_ok=True)
model = YOLO("yolov8n.pt")
reader = easyocr.Reader(['en'])
cap = cv2.VideoCapture(IP_CAMERA_URL)

if not cap.isOpened():
    print("❌ Unable to open camera stream. Check IP or Wi-Fi connection.")
    exit()

print("✅ Camera connected. Press 'r' to toggle signal, 'q' to quit.")

violation_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))
    height, width, _ = frame.shape

    # Draw stop line
    cv2.line(frame, (0, LINE_Y), (width, LINE_Y), (0, 0, 255) if SIGNAL_STATE == "RED" else (0, 255, 0), 3)
    cv2.putText(frame, f"Signal: {SIGNAL_STATE}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if SIGNAL_STATE == "RED" else (0, 255, 0), 3)

    results = model(frame)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id]

            if conf > CONF_THRESHOLD and cls_name in ["car", "bus", "truck", "motorbike"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
                cv2.putText(frame, cls_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 🚨 Red light violation detection
                if SIGNAL_STATE == "RED" and cy < LINE_Y:
                    violation_count += 1
                    filename = f"violations/violation_{violation_count}_{datetime.now().strftime('%H%M%S')}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"🚨 Violation detected! Saved: {filename}")

                    # Read number plate (optional)
                    vehicle_roi = frame[y1:y2, x1:x2]
                    text = reader.readtext(vehicle_roi, detail=0)
                    if text:
                        print("Plate Detected:", ' '.join(text))

    cv2.imshow("Red Light Violation Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # toggle signal manually (for demo)
        SIGNAL_STATE = "GREEN" if SIGNAL_STATE == "RED" else "RED"

cap.release()
cv2.destroyAllWindows()
