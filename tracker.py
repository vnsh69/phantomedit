import cv2
import numpy as np
from ultralytics import YOLO
import sys
import torch

# DeepSORT Import
try:
    sys.path.insert(0, './deep_sort_pytorch')
    from deep_sort.deep_sort import DeepSort
except ImportError:
    print("Error: Could not import DeepSort.")
    print("Make sure 'deep_sort_pytorch' folder exists.")
    sys.exit(1)

# Paths
YOLO_MODEL_PATH = 'yolov8n.pt'  # or any model like yolov8s.pt
DEEPSORT_WEIGHTS_PATH = 'deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7'
INPUT_VIDEO_PATH = 'input.mp4'
OUTPUT_VIDEO_PATH = 'output_tracked.mp4'

# Load YOLOv8 model (CPU-only)
print("Loading YOLOv8 model on CPU...")
yolo_model = YOLO(YOLO_MODEL_PATH)

# Initialize DeepSORT
print("Initializing DeepSORT...")
tracker = DeepSort(model_path=DEEPSORT_WEIGHTS_PATH, max_age=70, use_cuda=False)

# Video I/O
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open {INPUT_VIDEO_PATH}")
    sys.exit(1)

# Video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
print(f"Saving output to {OUTPUT_VIDEO_PATH}")

# Processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1: Run detection (track ALL classes, so no class filter)
    results = yolo_model(frame, verbose=False)

    bboxes_xywh = []
    confidences = []
    class_ids = []

    # Step 2: Parse detections
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item()
            cls = int(box.cls[0].cpu().item())

            # Convert xyxy to xywh (center format) for DeepSORT
            w, h = x2 - x1, y2 - y1
            center_x = x1 + w / 2
            center_y = y1 + h / 2
            bboxes_xywh.append([center_x, center_y, w, h])
            confidences.append(float(conf))
            class_ids.append(cls)

    # Step 3: DeepSORT update
    if len(bboxes_xywh) > 0:
        np_bboxes = np.array(bboxes_xywh)
        np_confidences = np.array(confidences)
        np_class_ids = np.array(class_ids)
        tracks, _ = tracker.update(np_bboxes, np_confidences, np_class_ids, frame)
    else:
        tracks = []

    # Step 4: Draw tracks
    if len(tracks) > 0:
        for track in tracks:
            # track format: [x1, y1, x2, y2, class_id, track_id]
            x1, y1, x2, y2, class_id, track_id = track.astype(int)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw tracking ID and class
            label = f"ID: {track_id} Class: {class_id}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Step 5: Write to video
    out.write(frame)

    # Optional: Display live
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
print("Done. Cleaning up...")
cap.release()
out.release()
cv2.destroyAllWindows()