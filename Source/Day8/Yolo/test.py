from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("last.pt")

# Run inference on 'bus.jpg' with arguments
model.predict("1.jpg", save=True, imgsz=640, conf=0.01)
