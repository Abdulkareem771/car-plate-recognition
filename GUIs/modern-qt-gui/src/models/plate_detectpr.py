from ultralytics import YOLO
from src.config.settings import MODEL_PLATE, MODEL_PLATE_DETAILS
from src.utils.helpers import ensure_model_paths

class PlateDetector:
    def __init__(self):
        # Ensure models exist before loading
        ensure_model_paths()
        
        self.plate_model = YOLO(str(MODEL_PLATE))
        self.details_model = YOLO(str(MODEL_PLATE_DETAILS))
        
    def detect_plates(self, image, conf_threshold=0.5):
        return self.plate_model(image, conf=conf_threshold)
    
    def detect_plate_details(self, plate_image, conf_threshold=0.3):
        return self.details_model(plate_image, conf=conf_threshold)