from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

model.train(
    data="data.yaml",   # ✅ fixed
    epochs=10,
    imgsz=640
)



