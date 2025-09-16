from ultralytics import YOLO
import cv2
import os


MODEL_PATH = "runs/detect/yolov8n14/weights/best.pt"        
IMAGES_DIR = "archive/test/test"    
OUT_DIR = "results"           


os.makedirs(OUT_DIR, exist_ok=True)


model = YOLO(MODEL_PATH)


image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

for file in image_files:
    img_path = os.path.join(IMAGES_DIR, file)
    results = model(img_path)   

    
    for r in results:
        annotated = r.plot()
        out_path = os.path.join(OUT_DIR, file)
        cv2.imwrite(out_path, annotated)

print(f" Done Annotated results are saved in '{OUT_DIR}' folder.")
