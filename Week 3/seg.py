from ultralytics import YOLO

# Load segmentation model
model = YOLO("yolo26n-seg.pt")

# Run on your folder
results = model(
    "/Users/ananya/Desktop/IIITH_Int/3frames/",
    save=True,       # saves annotated images
    save_txt=True,   # saves labels
    project="runs/segmentation",
    name="seg_output",
    conf=0.25
)

print("Segmentation complete!")
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

metrics = model.val(data="coco.yaml")
print(metrics)
