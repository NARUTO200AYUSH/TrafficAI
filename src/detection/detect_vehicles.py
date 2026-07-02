from ultralytics import YOLO

# Load the YOLO model
model = YOLO("models/yolov8n.pt")

# Run detection on the video
results = model.predict(
    source="data/videos/traffic.mp4",
    save=True,
    conf=0.4
)

print("✅ Detection completed!")