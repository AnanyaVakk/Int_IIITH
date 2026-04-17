from ultralytics import YOLO

model = YOLO("runs/segment/train2/weights/best.pt")

metrics = model.val(data="data.yaml")
print(metrics)
