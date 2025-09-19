import cv2
import numpy as np
import requests
from PIL import Image
import torch
import torchvision.transforms as transforms
from urllib.parse import urlparse
import time
import os
from pathlib import Path

class CarPlateDetector:
    def __init__(self, model_path, stream_url, conf_threshold=0.5):
        """
        Initialize the car plate detector
        
        Args:
            model_path: Path to your trained YOLO model weights
            stream_url: HTTP stream URL (RTSP, HTTP, etc.)
            conf_threshold: Confidence threshold for detection
        """
        # Load YOLO model
        self.model = self.load_model(model_path)
        self.stream_url = stream_url
        self.conf_threshold = conf_threshold
        self.cap = None
        
        # Check if stream is valid
        self.validate_stream_url(stream_url)
        
    def load_model(self, model_path):
        """Load YOLOv8 model"""
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            return model
        except Exception as e:
            raise Exception(f"Could not load YOLO model: {e}")
    
    def validate_stream_url(self, url):
        """Validate the stream URL"""
        parsed = urlparse(url)
        if not parsed.scheme in ['http', 'https', 'rtsp', 'rtmp']:
            raise ValueError("Unsupported stream protocol")
    
    def connect_to_stream(self):
        """Connect to the HTTP video stream"""
        try:
            self.cap = cv2.VideoCapture(self.stream_url)
            if not self.cap.isOpened():
                raise Exception(f"Could not open stream: {self.stream_url}")
            print(f"Successfully connected to stream: {self.stream_url}")
        except Exception as e:
            print(f"Error connecting to stream: {e}")
            return False
        return True
    
    def process_frame(self, frame):
        """Process a single frame for car plate detection using YOLOv8"""
        # Perform detection
        results = self.model(frame)

        # Parse results (YOLOv8 format)
        detections = results[0].boxes.data.cpu().numpy()  # numpy array of detections

        # Draw bounding boxes
        processed_frame = self.draw_boxes(frame, detections)

        return processed_frame, detections

    def draw_boxes(self, frame, detections):
        """Draw bounding boxes and labels on the frame for YOLOv8 results"""
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection[:6]  # extract values
            if conf > self.conf_threshold:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Create label with confidence
                label = f"Plate: {conf:.2f}"
                
                # Draw label background
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(frame, (x1, y1 - label_height - baseline),
                             (x1 + label_width, y1), (0, 255, 0), -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1, y1 - baseline),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame
    
    def run_detection(self):
        """Main function to run car plate detection on live stream"""
        if not self.connect_to_stream():
            return
        
        print("Starting car plate detection... Press 'q' to quit.")
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                print("Failed to grab frame. Reconnecting...")
                time.sleep(2)
                self.connect_to_stream()
                continue
            
            # Process frame
            processed_frame, detections = self.process_frame(frame)
            
            # Display frame with detections
            cv2.imshow('Car Plate Detection', processed_frame)
            
            # Print detection info (optional)
            if len(detections) > 0:
                print(f"Detected {len(detections)} car plate(s)")
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

# Usage example
if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = Path(__file__).parent
    
    # Construct the path to the model (relative to the script location)
    model_path = script_dir.parent / "runs" / "detect" / "yolov8n14" / "weights" / "best.pt"
    
    # Convert to absolute path and string (for compatibility)
    model_path_str = str(model_path.resolve())
    
    # Check if the model file exists
    if not os.path.exists(model_path_str):
        raise FileNotFoundError(f"Model not found at: {model_path_str}")
    
    # Configuration
    STREAM_URL = "http://192.168.8.135:8080/video"  # Update with your stream URL
    
    # Initialize detector
    detector = CarPlateDetector(
        model_path=model_path_str,
        stream_url=STREAM_URL,
        conf_threshold=0.6  # Adjust based on your model's performance
    )
    
    # Start detection
    detector.run_detection()