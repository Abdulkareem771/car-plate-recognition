import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

class PlateDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def detect(self, image, conf_threshold=0.5):
        """Detect license plates in an image"""
        results = self.model(image)
        detections = []
        
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf >= conf_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'crop': image[y1:y2, x1:x2]
                    })
        
        return detections