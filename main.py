from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    results = model.train(
        data="archive/data.yaml", 
        epochs=20, 
        imgsz=640, 
        batch=16,
        device=0,
        name="yolov8n",
        patience=1,
            
        workers=0,  
        )

if __name__ == "__main__":
    main()