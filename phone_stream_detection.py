"""
phone_stream_detection.py
Use phone IP camera MJPEG stream as a video source, detect vehicles with YOLOv8,
attempt to find license-plate-like regions in vehicle ROIs, and run OCR.
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
import easyocr

# ------------------ Configuration ------------------
# Put your phone stream URL here (example for IP Webcam app)
STREAM_URL = "http://10.52.2.150.8080/video"  # <<--- change to your phone IP + stream path

# YOLO model (use small one for speed). You can change to 'yolov8n.pt' or a custom model
YOLO_MODEL = "yolov8n.pt"
CONFIDENCE = 0.35       # detection confidence threshold
VEHICLE_CLASSES = {2, 3, 5, 7}  # COCO class ids for vehicle-like: 2=car,3=motorbike,5=bus,7=truck
# (If using names instead of ids, YOLO returns both; but here we use ids)

OCR_LANGS = ['en']      # languages for EasyOCR
MIN_PLATE_AREA = 2000   # min area of candidate plate contour to consider
ASPECT_RATIO_RANGE = (2.0, 6.5)  # plate width/height range
DOWNSCALE = 1.0         # set <1.0 to speed up processing at cost of accuracy

# Optional: virtual line (y coordinate) — for simple violation rules later
VIOLATION_LINE_Y = None  # e.g. 0.7 * frame_height (set after first frame if you want)

# ------------------ Init ------------------
print("Loading YOLO model...")
model = YOLO(YOLO_MODEL)
print("Loading OCR model (easyocr)...")
ocr = easyocr.Reader(OCR_LANGS, gpu=False)  # set gpu=True if you have CUDA + easyocr compiled

cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    raise SystemExit(f"Unable to open stream: {STREAM_URL}")

# Optionally get FPS (not reliable for MJPEG) and set display window
cv2.namedWindow("Traffic", cv2.WINDOW_NORMAL)

last_time = time.time()
frame_count = 0

def find_plate_candidates(vehicle_roi):
    """Return list of bounding boxes (x,y,w,h) that look like license plates using edge+contour heuristics."""
    gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # Emphasize rectangular shapes
    grad = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    # Adaptive threshold + morphology
    _, th = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_roi, w_roi = vehicle_roi.shape[:2]
    candidates = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h
        if area < MIN_PLATE_AREA:
            continue
        ar = (w / float(h)) if h>0 else 0
        # filter by aspect ratio and relative width
        if ASPECT_RATIO_RANGE[0] <= ar <= ASPECT_RATIO_RANGE[1] and w > 0.2*w_roi:
            candidates.append((x,y,w,h))
    # sort by area descending
    candidates = sorted(candidates, key=lambda b: b[2]*b[3], reverse=True)
    return candidates

def ocr_plate_image(plate_img):
    """Run easyocr on plate image and return best text or empty string."""
    try:
        # EasyOCR prefers grayscale but works with color too
        result = ocr.readtext(plate_img)
        # result = [(bbox, text, conf), ...]
        texts = [res[1] for res in result if res[2] > 0.3]
        if texts:
            # return the longest text candidate (simple heuristic)
            texts = sorted(texts, key=lambda t: len(t), reverse=True)
            return texts[0]
    except Exception as e:
        print("OCR error:", e)
    return ""

print("Start streaming and detecting. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        # Could be a temporary hiccup; you may retry open or break
        print("Frame not received, retrying...")
        time.sleep(0.5)
        continue

    frame_count += 1
    if DOWNSCALE != 1.0:
        frame = cv2.resize(frame, (0,0), fx=DOWNSCALE, fy=DOWNSCALE)

    # Set violation line after getting frame size if not set
    if VIOLATION_LINE_Y is None:
        h = frame.shape[0]
        VIOLATION_LINE_Y = int(0.75 * h)  # default line at 75% of frame height

    # Run YOLO inference (fast: smallest model)
    results = model.predict(frame, imgsz=640, conf=CONFIDENCE, verbose=False)

    # results is a list; take first (single image)
    r = results[0]
    boxes = r.boxes  # Boxes object (ultralytics)
    # r.boxes.xyxy, r.boxes.conf, r.boxes.cls are available

    display = frame.copy()

    # Loop over detections
    for box in boxes:
        xyxy = box.xyxy[0].numpy()  # [x1,y1,x2,y2]
        conf = float(box.conf[0].numpy())
        cls_id = int(box.cls[0].numpy())
        # Only proceed if class is a vehicle
        if cls_id not in VEHICLE_CLASSES:
            continue

        x1,y1,x2,y2 = map(int, xyxy)
        cv2.rectangle(display, (x1,y1), (x2,y2), (0,255,0), 2)
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        label = f"ID:{cls_id} {conf:.2f}"
        cv2.putText(display, label, (x1, max(10,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Crop ROI for plate detection
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        candidates = find_plate_candidates(roi)
        plate_text = ""
        plate_box_abs = None
        if candidates:
            # Try best candidate(s)
            for (px,py,pw,ph) in candidates[:3]:
                plate_img = roi[py:py+ph, px:px+pw]
                # Optional: preprocess plate image (resize, denoise)
                plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                plate_gray = cv2.resize(plate_gray, (0,0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                txt = ocr_plate_image(plate_gray)
                if txt and len(txt) >= 4:   # heuristic: plates longer than 3 chars
                    plate_text = txt
                    plate_box_abs = (x1+px, y1+py, pw, ph)
                    break

        # draw plate detection / OCR results
        if plate_text:
            bx,by,bw,bh = plate_box_abs
            cv2.rectangle(display, (bx,by), (bx+bw, by+bh), (0,0,255), 2)
            cv2.putText(display, plate_text, (bx, max(10,by-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        else:
            # if no plate found, optionally draw top candidate
            if candidates:
                px,py,pw,ph = candidates[0]
                cv2.rectangle(display, (x1+px, y1+py), (x1+px+pw, y1+py+ph), (255,0,0), 1)

        # Example: simple violation rule (placeholder)
        # If you want to detect e.g., crossing a line, check centroid vs VIOLATION_LINE_Y:
        # if cy > VIOLATION_LINE_Y:
        #     # mark as crossed
        #     cv2.putText(display, "CROSSED", (cx, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # draw the violation line
    cv2.line(display, (0, VIOLATION_LINE_Y), (display.shape[1], VIOLATION_LINE_Y), (0,0,255), 2)

    # FPS display
    now = time.time()
    if now - last_time >= 1.0:
        fps = frame_count / (now - last_time)
        last_time = now
        frame_count = 0
    else:
        fps = None
    if fps:
        cv2.putText(display, f"FPS: {fps:.1f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Traffic", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
