from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit, 
    QLineEdit, QHBoxLayout, QGroupBox
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QTimer
import cv2
from urllib.parse import urlparse

class LiveStreamTab(QWidget):
    def __init__(self, plate_detector, plate_recognizer):
        super().__init__()
        self.detector = plate_detector
        self.recognizer = plate_recognizer
        self.cap = None
        self.is_streaming = False
        self.conf_threshold = 0.5
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Stream URL input
        url_group = QGroupBox("Stream Configuration")
        url_layout = QHBoxLayout()
        
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter stream URL (http, rtsp, etc.)")
        self.url_input.setText("http://192.168.1.33:8080/video")  # Default example
        
        url_layout.addWidget(QLabel("Stream URL:"))
        url_layout.addWidget(self.url_input)
        url_group.setLayout(url_layout)
        
        # Confidence threshold
        conf_group = QGroupBox("Detection Settings")
        conf_layout = QHBoxLayout()
        
        self.conf_input = QLineEdit()
        self.conf_input.setPlaceholderText("Confidence threshold (0.1-0.9)")
        self.conf_input.setText("0.5")
        
        conf_layout.addWidget(QLabel("Confidence:"))
        conf_layout.addWidget(self.conf_input)
        conf_group.setLayout(conf_layout)
        
        # Buttons
        self.btn_start = QPushButton("Start Stream")
        self.btn_stop = QPushButton("Stop Stream")
        self.btn_stop.setEnabled(False)
        
        self.btn_start.clicked.connect(self.start_stream)
        self.btn_stop.clicked.connect(self.stop_stream)
        
        # Image display
        self.image_label = QLabel("Stream not started")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        
        # Text output
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setMaximumHeight(100)
        
        # Add widgets to layout
        layout.addWidget(url_group)
        layout.addWidget(conf_group)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_start)
        button_layout.addWidget(self.btn_stop)
        layout.addLayout(button_layout)
        
        layout.addWidget(self.image_label)
        layout.addWidget(QLabel("Detected Plates:"))
        layout.addWidget(self.text_output)
        
        self.setLayout(layout)
        
        # Timer for updating the video feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
    
    def validate_stream_url(self, url):
        """Validate the stream URL"""
        parsed = urlparse(url)
        return parsed.scheme in ['http', 'https', 'rtsp', 'rtmp', 'tcp']
    
    def start_stream(self):
        """Start the live stream"""
        stream_url = self.url_input.text().strip()
        conf_text = self.conf_input.text().strip()
        
        # Validate URL
        if not self.validate_stream_url(stream_url):
            self.text_output.setPlainText("Error: Invalid stream URL format")
            return
            
        # Validate confidence
        try:
            self.conf_threshold = float(conf_text)
            if not 0.1 <= self.conf_threshold <= 0.9:
                raise ValueError("Confidence out of range")
        except ValueError:
            self.text_output.setPlainText("Error: Confidence must be between 0.1 and 0.9")
            return
        
        # Try to connect to the stream
        self.cap = cv2.VideoCapture(stream_url)
        if not self.cap.isOpened():
            self.text_output.setPlainText("Error: Could not connect to stream")
            self.cap = None
            return
        
        self.is_streaming = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.text_output.setPlainText("Connected to stream. Processing...")
        
        # Start the timer to update frames
        self.timer.start(30)  # Update every 30ms
    
    def stop_stream(self):
        """Stop the live stream"""
        self.is_streaming = False
        self.timer.stop()
        
        if self.cap:
            self.cap.release()
            self.cap = None
            
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.image_label.setText("Stream stopped")
        self.text_output.setPlainText("Stream stopped")
    
    def update_frame(self):
        """Process and update the current frame"""
        if not self.is_streaming or not self.cap:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.text_output.setPlainText("Error: Failed to read frame from stream")
            return
        
        # Process the frame
        processed_frame, detected_texts = self.process_frame(frame)
        
        # Convert to QImage
        rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Update the image label
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.width(), 
            self.image_label.height(),
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))
        
        # Update the text output
        if detected_texts:
            self.text_output.setPlainText("\n".join(detected_texts))
    
    def process_frame(self, frame):
        """Process a single frame for car plate detection"""
        detections = self.detector.detect(frame, self.conf_threshold)
        detected_texts = []
        annotated = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            crop = detection['crop']

            if crop.size > 0:
                text_detected = self.recognizer.recognize(crop)
                detected_texts.append(text_detected)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, text_detected, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return annotated, detected_texts