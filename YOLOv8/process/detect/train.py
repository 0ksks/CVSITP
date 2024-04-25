from ultralytics import YOLO
detectModel = YOLO("yolov8n.pt")
results = detectModel.train(task="detect", data="process/detect_data/detect.yaml", epochs=10, verbose=True)