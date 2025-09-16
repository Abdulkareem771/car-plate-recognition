from ultralytics import YOLO

def main():
    model = YOLO("src\models\yolov8s.pt")

    results = model.train(
        data=r"src\data\archive\data.yaml", 
        epochs=20, 
        imgsz=640, 
        batch=16,
        device=0,
        name="yolov8s",
        patience=1,
            
        workers=0,  
        )

if __name__ == "__main__":
    main()