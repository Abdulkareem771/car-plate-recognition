from ultralytics import YOLO
model = YOLO("src\models\yolov8s.pt")
## training the model on yemen plate


results = model.train(
        data="src/data/yemen-plate/data.yaml", 
        epochs=30, 
        imgsz=640, 
        batch=9,
        device=0,
        name="yolov8s",
        patience=5,
        amp=False,  # Disable AMP to avoid the yolo11n.pt download issue
        workers=0,  # Disable multiprocessing to avoid Windows issues
        )

