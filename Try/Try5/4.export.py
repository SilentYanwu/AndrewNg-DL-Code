from ultralytics import YOLO

model = YOLO("runs/train/exp_yolo/weights/best.pt")

model.export(
    format="onnx",
    opset=12,
    simplify=True
)
