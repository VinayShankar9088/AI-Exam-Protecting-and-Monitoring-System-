from ultralytics import YOLO

model = YOLO("models/yolov8n.pt")  # ✅ nano model for CPU

def detect_objects(frame):
    results = model(frame, device="cpu")
    return results[0].plot()
