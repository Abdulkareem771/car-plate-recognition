import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFileDialog, QTextEdit, QGroupBox, QGridLayout, QLineEdit, QFrame,
    QTabWidget, QComboBox, QCheckBox, QSlider, QProgressBar, QSizePolicy,
    QMessageBox
)
from PySide6.QtGui import QPixmap, QImage, QFont, QIcon, QPalette, QColor
from PySide6.QtCore import Qt, QTimer

from src.models.plate_detector import PlateDetector
from src.models.text_recognizer import TextRecognizer
from src.utils.camera import CameraThread
from src.utils.helpers import is_valid_ip, is_valid_url
from src.gui.styles import DARK_STYLE

import time
import paho.mqtt.client as mqtt
import json
from datetime import datetime

class PlateApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚔 Yemeni Traffic Plate Recognition System")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize models
        try:
            self.plate_detector = PlateDetector()
            self.text_recognizer = TextRecognizer()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize models: {str(e)}")
            raise
        
        # Initialize camera thread
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.update_frame)
        self.camera_thread.error_occurred.connect(self.handle_camera_error)
        
        # MQTT client
        self.mqtt_client = None
        self.mqtt_connected = False
        
        # Frame processing variables
        self.current_frame = None
        self.annotated_frame = None
        self.processing_frame = False
        self.last_processed_time = 0
        self.processing_interval = 2.0

        # Initialize processing timer
        self.processing_timer = QTimer()
        self.processing_timer.timeout.connect(self.process_camera_frame)
        self.processing_timer.start(100)
        
        # Apply styles
        self.setStyleSheet(DARK_STYLE)
        
        self.setup_ui()
        
    def setup_ui(self):
        main_layout = QHBoxLayout()

        # Left Panel: Image and Controls
        left_panel = QVBoxLayout()
        
        # Create tabs
        self.tabs = QTabWidget()
        
        # Image tab
        image_tab = QWidget()
        image_tab_layout = QVBoxLayout()
        
        # Upload button
        self.btn_upload = QPushButton("📷 Upload Vehicle Image")
        self.btn_upload.setMinimumHeight(50)
        self.btn_upload.clicked.connect(self.upload_image)
        image_tab_layout.addWidget(self.btn_upload)
        
        # Camera selection
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("Camera Source:"))
        
        self.camera_type_combo = QComboBox()
        self.camera_type_combo.addItems(["Webcam", "IP Camera"])
        self.camera_type_combo.currentIndexChanged.connect(self.camera_type_changed)
        camera_layout.addWidget(self.camera_type_combo)
        
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Camera 0", "Camera 1", "Camera 2"])
        camera_layout.addWidget(self.camera_combo)
        
        # IP camera input
        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("Enter camera URL (rtsp://, http://, or IP address)")
        self.ip_input.setVisible(False)
        camera_layout.addWidget(self.ip_input)
        
        self.btn_start_cam = QPushButton("Start Camera")
        self.btn_start_cam.clicked.connect(self.start_camera)
        camera_layout.addWidget(self.btn_start_cam)
        
        self.btn_stop_cam = QPushButton("Stop Camera")
        self.btn_stop_cam.clicked.connect(self.stop_camera)
        self.btn_stop_cam.setEnabled(False)
        camera_layout.addWidget(self.btn_stop_cam)
        
        image_tab_layout.addLayout(camera_layout)

        # Image display
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setStyleSheet("border: 2px solid #555555; background-color: #404040; border-radius: 8px;")
        image_tab_layout.addWidget(self.image_label)
        
        # Processing options
        options_group = QGroupBox("Processing Options")
        options_layout = QVBoxLayout()
        
        self.enable_ocr_check = QCheckBox("Enable OCR")
        self.enable_ocr_check.setChecked(True)
        options_layout.addWidget(self.enable_ocr_check)
        
        self.save_results_check = QCheckBox("Save Results")
        options_layout.addWidget(self.save_results_check)
        
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Confidence Threshold:"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self.update_confidence_label)
        confidence_layout.addWidget(self.confidence_slider)
        self.confidence_label = QLabel("0.5")
        confidence_layout.addWidget(self.confidence_label)
        options_layout.addLayout(confidence_layout)
        
        # MQTT Settings - Compact version
        mqtt_group = QGroupBox("MQTT Settings")
        mqtt_group.setMaximumHeight(200)
        mqtt_layout = QGridLayout()
        
        # MQTT Enable checkbox
        self.mqtt_enable_check = QCheckBox("Enable MQTT")
        self.mqtt_enable_check.setChecked(True)
        self.mqtt_enable_check.stateChanged.connect(self.toggle_mqtt_settings)
        mqtt_layout.addWidget(self.mqtt_enable_check, 0, 0, 1, 2)
        
        # Broker address
        mqtt_layout.addWidget(QLabel("Broker:"), 1, 0)
        self.broker_input = QLineEdit("192.168.8.101")
        self.broker_input.setPlaceholderText("MQTT broker IP")
        self.broker_input.setEnabled(False)
        mqtt_layout.addWidget(self.broker_input, 1, 1)
        
        # Topic for publishing
        mqtt_layout.addWidget(QLabel("Topic:"), 2, 0)
        self.topic_input = QLineEdit("car/summary")
        self.topic_input.setPlaceholderText("MQTT topic")
        self.topic_input.setEnabled(False)
        mqtt_layout.addWidget(self.topic_input, 2, 1)
        
        # Connection status and button
        self.mqtt_status_label = QLabel("Disconnected")
        self.mqtt_status_label.setStyleSheet("color: #ff6666; font-size: 10px;")
        mqtt_layout.addWidget(self.mqtt_status_label, 3, 0)
        
        self.mqtt_connect_btn = QPushButton("Connect")
        self.mqtt_connect_btn.clicked.connect(self.connect_mqtt)
        self.mqtt_connect_btn.setEnabled(True)
        self.mqtt_connect_btn.setMaximumWidth(80)
        mqtt_layout.addWidget(self.mqtt_connect_btn, 3, 1)
        
        mqtt_group.setLayout(mqtt_layout)
        options_layout.addWidget(mqtt_group)
        
        options_group.setLayout(options_layout)
        image_tab_layout.addWidget(options_group)
        
        image_tab.setLayout(image_tab_layout)
        self.tabs.addTab(image_tab, "Image Processing")
        
        # History tab
        history_tab = QWidget()
        history_tab_layout = QVBoxLayout()
        
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        history_tab_layout.addWidget(self.history_text)
        
        history_tab.setLayout(history_tab_layout)
        self.tabs.addTab(history_tab, "History")
        
        left_panel.addWidget(self.tabs)

        # Right Panel: Results
        right_panel = QVBoxLayout()
        
        # Title
        title = QLabel("🚔 Plate Recognition Results")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffffff; margin: 15px;")
        right_panel.addWidget(title)

        # Class-based results (compact)
        class_group = QGroupBox("📋 Detected Classes")
        class_layout = QGridLayout()
        
        # City field
        class_layout.addWidget(QLabel("City Number:"), 0, 0)
        self.city_field = QLineEdit()
        self.city_field.setReadOnly(True)
        self.city_field.setPlaceholderText("City code")
        class_layout.addWidget(self.city_field, 0, 1)
        
        # Text field
        class_layout.addWidget(QLabel("Vehicle Type:"), 1, 0)
        self.text_field = QLineEdit()
        self.text_field.setReadOnly(True)
        self.text_field.setPlaceholderText("خصوصي نقل اجرة")
        class_layout.addWidget(self.text_field, 1, 1)
        
        # Arabic Number field
        class_layout.addWidget(QLabel("Arabic Number:"), 2, 0)
        self.arabic_number_field = QLineEdit()
        self.arabic_number_field.setReadOnly(True)
        self.arabic_number_field.setPlaceholderText("٠١٢٣٤٥")
        self.arabic_number_field.setStyleSheet("font-size: 16px; padding: 8px; font-weight: bold;")
        class_layout.addWidget(self.arabic_number_field, 2, 1)
        
        # English Number field
        class_layout.addWidget(QLabel("English Number:"), 3, 0)
        self.english_number_field = QLineEdit()
        self.english_number_field.setReadOnly(True)
        self.english_number_field.setPlaceholderText("012345")
        self.english_number_field.setStyleSheet("font-size: 16px; padding: 8px; font-weight: bold;")
        class_layout.addWidget(self.english_number_field, 3, 1)
        
        class_group.setLayout(class_layout)
        right_panel.addWidget(class_group)

        # Full plate OCR results
        full_ocr_group = QGroupBox("🔍 Full Plate OCR (Fallback)")
        full_ocr_layout = QVBoxLayout()
        
        self.full_ocr_output = QTextEdit()
        self.full_ocr_output.setReadOnly(True)
        self.full_ocr_output.setMaximumHeight(150)
        self.full_ocr_output.setPlaceholderText("Complete text from entire plate...")
        full_ocr_layout.addWidget(self.full_ocr_output)
        
        full_ocr_group.setLayout(full_ocr_layout)
        right_panel.addWidget(full_ocr_group)

        # Summary for traffic police
        summary_group = QGroupBox("📝 Traffic Summary")
        summary_layout = QVBoxLayout()
        
        self.summary_output = QTextEdit()
        self.summary_output.setReadOnly(True)
        self.summary_output.setMaximumHeight(100)
        self.summary_output.setPlaceholderText("Vehicle information summary...")
        summary_layout.addWidget(self.summary_output)
        
        summary_group.setLayout(summary_layout)
        right_panel.addWidget(summary_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_panel.addWidget(self.progress_bar)

        # Add panels to main layout
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)

        self.setLayout(main_layout)
        
    def toggle_mqtt_settings(self, state):
        """Enable/disable MQTT settings based on checkbox"""
        enabled = (state == Qt.Checked)
        self.broker_input.setEnabled(enabled)
        self.topic_input.setEnabled(enabled)
        self.mqtt_connect_btn.setEnabled(enabled)
        
        if not enabled and self.mqtt_client:
            self.disconnect_mqtt()

    def connect_mqtt(self):
        """Connect to MQTT broker"""
        if self.mqtt_client and self.mqtt_connected:
            return
        
        broker = self.broker_input.text().strip()
        if not broker:
            QMessageBox.warning(self, "MQTT Error", "Please enter a broker address")
            return
        
        try:
            # Create MQTT client
            client_id = f"plate_recognition_{int(time.time())}"
            self.mqtt_client = mqtt.Client(client_id=client_id)
            
            # Connect to broker
            self.mqtt_client.connect(broker, 1883, 60)
            self.mqtt_client.loop_start()
            
            self.mqtt_connected = True
            self.mqtt_status_label.setText("Connected")
            self.mqtt_status_label.setStyleSheet("color: #66ff66; font-size: 10px;")
            self.mqtt_connect_btn.setText("Disconnect")
            self.mqtt_connect_btn.clicked.disconnect()
            self.mqtt_connect_btn.clicked.connect(self.disconnect_mqtt)
            
        except Exception as e:
            QMessageBox.critical(self, "MQTT Error", f"Failed to connect to MQTT broker: {str(e)}")
            self.mqtt_status_label.setText("Connection Failed")
            self.mqtt_status_label.setStyleSheet("color: #ff6666; font-size: 10px;")

    def disconnect_mqtt(self):
        """Disconnect from MQTT broker"""
        if self.mqtt_client:
            try:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
            except:
                pass
            
        self.mqtt_client = None
        self.mqtt_connected = False
        self.mqtt_status_label.setText("Disconnected")
        self.mqtt_status_label.setStyleSheet("color: #ff6666; font-size: 10px;")
        self.mqtt_connect_btn.setText("Connect")
        self.mqtt_connect_btn.clicked.disconnect()
        self.mqtt_connect_btn.clicked.connect(self.connect_mqtt)

    def publish_to_mqtt(self, summary_data):
        """Publish summary data to MQTT topic"""
        if not self.mqtt_connected or not self.mqtt_client:
            return False
        
        try:
            topic = self.topic_input.text().strip() or "car/summary"
            
            # Convert summary data to JSON
            payload = json.dumps(summary_data, ensure_ascii=False)
            
            # Publish to MQTT
            result = self.mqtt_client.publish(topic, payload)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"Published to MQTT topic {topic}: {payload}")
                return True
            else:
                print(f"Failed to publish to MQTT: {result.rc}")
                return False
                
        except Exception as e:
            print(f"MQTT publish error: {e}")
            return False
        
    def camera_type_changed(self, index):
        is_ip_camera = (index == 1)  # IP Camera selected
        self.camera_combo.setVisible(not is_ip_camera)
        self.ip_input.setVisible(is_ip_camera)
        
    def update_confidence_label(self, value):
        self.confidence_label.setText(f"{value/100:.2f}")
        
    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select vehicle image", "", "Images (*.jpg *.jpeg *.png)"
        )
        if not file_path:
            return

        self.process_image(file_path)
        
    def process_image(self, file_path):
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(10)
        
        img = cv2.imread(file_path)
        if img is None:
            self.summary_output.setPlainText("Error: Could not load image")
            self.progress_bar.setVisible(False)
            return
            
        annotated = img.copy()

        # Initialize results
        city_val = ""
        arabic_text = ""
        arabic_number = ""
        english_number = ""
        full_plate_text = ""

        # ---- Stage 1: Detect Plate ----
        self.progress_bar.setValue(30)
        conf_threshold = self.confidence_slider.value() / 100
        results_plate = self.plate_detector.detect_plates(img, conf_threshold)
        plate_detected = False
        
        for r in results_plate:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                plate_crop = img[y1:y2, x1:x2]
                plate_detected = True

                # Draw bounding box around the entire plate
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(annotated, "License Plate", (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # ---- Stage 2: Detect details inside plate ----
                self.progress_bar.setValue(60)
                results_details = self.plate_detector.detect_plate_details(plate_crop, conf_threshold/2)

                for dr in results_details:
                    for dbox in dr.boxes:
                        dx1, dy1, dx2, dy2 = map(int, dbox.xyxy[0].cpu().numpy())
                        cls_id = int(dbox.cls[0].cpu().numpy())
                        conf = float(dbox.conf[0].cpu().numpy())

                        detail_crop = plate_crop[dy1:dy2, dx1:dx2]
                        label = dr.names[cls_id]  # "text", "city", "number"

                        if detail_crop.size > 0 and self.enable_ocr_check.isChecked():
                            # Run OCR on each class
                            ocr_results = self.text_recognizer.recognize_text(detail_crop)
                            text_detected = " ".join([res[1] for res in ocr_results])

                            if label == "text":
                                arabic_text = self.text_recognizer.process_plate_text(text_detected)
                                color = (255, 0, 0)  # Red for text
                            elif label == "number":
                                arabic_num, eng_num, arabic_to_eng = self.text_recognizer.extract_numbers(text_detected)
                                arabic_number = arabic_num
                                english_number = eng_num if eng_num else arabic_to_eng
                                color = (0, 0, 255)  # Blue for numbers
                            elif label == "city":
                                city_arabic, city_eng, city_arabic_to_eng = self.text_recognizer.extract_numbers(text_detected)
                                city_val = city_arabic if city_arabic else (city_eng if city_eng else text_detected)
                                color = (255, 255, 0)  # Yellow for city

                            # Draw rectangle on annotated plate
                            cv2.rectangle(plate_crop, (dx1, dy1), (dx2, dy2), color, 2)
                            cv2.putText(plate_crop, f"{label}: {text_detected}", 
                                      (dx1, dy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # ---- Fallback: Full plate OCR ----
                self.progress_bar.setValue(80)
                if self.enable_ocr_check.isChecked():
                    full_ocr_results = self.text_recognizer.recognize_text(plate_crop)
                    full_plate_text = " ".join([res[1] for res in full_ocr_results])
                    
                    # If no specific numbers found, try to extract from full plate
                    if not arabic_number and not english_number:
                        arabic_num, eng_num, arabic_to_eng = self.text_recognizer.extract_numbers(full_plate_text)
                        arabic_number = arabic_num
                        english_number = eng_num if eng_num else arabic_to_eng
                    
                    # If no city found, try to extract from full plate
                    if not city_val:
                        city_arabic, city_eng, city_arabic_to_eng = self.text_recognizer.extract_numbers(full_plate_text)
                        city_val = city_arabic if city_arabic else (city_eng if city_eng else "Not detected")

                # Replace original image plate with annotated plate
                annotated[y1:y2, x1:x2] = plate_crop

        # Convert to QImage for display
        rgb_image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        self.image_label.setPixmap(
            pixmap.scaled(600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

        # Update GUI fields
        self.city_field.setText(city_val if city_val else "Not detected")
        self.text_field.setText(arabic_text if arabic_text else "Not detected")
        self.arabic_number_field.setText(arabic_number if arabic_number else "Not detected")
        self.english_number_field.setText(english_number if english_number else "Not detected")
        
        # Full plate OCR results
        self.full_ocr_output.setPlainText(full_plate_text if full_plate_text else "No text detected")
        
        # Traffic summary
        summary_text = ""
        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "image_file": file_path.split('/')[-1],
            "plate_detected": plate_detected,
            "city": city_val if city_val else "Unknown",
            "vehicle_type": arabic_text if arabic_text else "Unknown",
            "arabic_number": arabic_number if arabic_number else "Not detected",
            "english_number": english_number if english_number else "Not detected",
            "full_text": full_plate_text if full_plate_text else "No text detected"
        }
        
        if plate_detected:
            summary_text += f"✅ Plate Detected\n"
            summary_text += f"City: {summary_data['city']}\n"
            summary_text += f"Type: {summary_data['vehicle_type']}\n"
            if arabic_number:
                summary_text += f"Arabic: {arabic_number}\n"
            if english_number:
                summary_text += f"English: {english_number}\n"
        else:
            summary_text = "❌ No plate detected in image"
            summary_data["plate_detected"] = False
        
        self.summary_output.setPlainText(summary_text)
        
        # Publish to MQTT if enabled
        if plate_detected and self.mqtt_enable_check.isChecked() and self.mqtt_connected:
            self.publish_to_mqtt(summary_data)
        
        # Add to history
        if plate_detected and self.save_results_check.isChecked():
            history_entry = f"Image: {file_path.split('/')[-1]}\n{summary_text}\n{'-'*50}\n"
            self.history_text.append(history_entry)
            
        self.progress_bar.setValue(100)
        
        # Hide progress bar after a short delay without affecting other UI elements
        def hide_progress():
            self.progress_bar.setVisible(False)
            
        QTimer.singleShot(500, hide_progress)
        
    def start_camera(self):
        camera_type = self.camera_type_combo.currentIndex()
    
        if camera_type == 0:  # Webcam
            camera_index = self.camera_combo.currentIndex()
            self.camera_thread.set_camera_source(camera_index)
        else:  # IP Camera/URL
            camera_url = self.ip_input.text().strip()
    
            if not camera_url:
                self.summary_output.setPlainText("Please enter a camera URL or IP address")
                return
    
            # If it's an IP without protocol, construct default URL
            if not any(camera_url.startswith(proto) for proto in ['rtsp://', 'http://', 'https://']):
                if is_valid_ip(camera_url):
                    camera_url = f"rtsp://{camera_url}:554/stream1"
                    print(f"Using default RTSP URL: {camera_url}")
                else:
                    self.summary_output.setPlainText("Please enter a valid URL or IP address")
                    return
    
            self.camera_thread.set_camera_source(camera_url)
    
        # Reset processing state
        self.current_frame = None
        self.annotated_frame = None
        self.processing_frame = False
        self.last_processed_time = 0
    
        self.camera_thread.start()
        self.btn_start_cam.setEnabled(False)
        self.btn_stop_cam.setEnabled(True)
        
    def stop_camera(self):
        self.camera_thread.stop()
        self.annotated_frame = None
        self.btn_start_cam.setEnabled(True)
        self.btn_stop_cam.setEnabled(False)
        
    def update_frame(self, frame):
        """Display the frame and store it for processing"""
        try:
            # Store the current frame for processing
            self.current_frame = frame.copy()
            
            # Display the annotated frame if available, otherwise display the original frame
            display_frame = self.annotated_frame if self.annotated_frame is not None else frame
            
            # Convert to QImage for display
            rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            # Scale and display
            self.image_label.setPixmap(
                pixmap.scaled(600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            
        except Exception as e:
            print(f"Error in update_frame: {e}")
        
    def handle_camera_error(self, error_msg):
        self.summary_output.setPlainText(f"Camera Error: {error_msg}")
        self.stop_camera()
        
    def closeEvent(self, event):
        self.camera_thread.stop()
        self.processing_timer.stop()
        self.disconnect_mqtt()
        event.accept()
    
    def process_camera_frame(self):
        """Process the current camera frame directly without temp files"""
        if self.processing_frame or self.current_frame is None:
            return
        
        try:
            self.processing_frame = True
            frame = self.current_frame.copy()
            annotated = frame.copy()

            # Initialize results
            city_val = ""
            arabic_text = ""
            arabic_number = ""
            english_number = ""
            full_plate_text = ""

            # ---- Stage 1: Detect Plate ----
            conf_threshold = self.confidence_slider.value() / 100
            results_plate = self.plate_detector.detect_plates(frame, conf_threshold)
            plate_detected = False
            
            for r in results_plate:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    plate_crop = frame[y1:y2, x1:x2]
                    plate_detected = True

                    # Draw bounding box around the entire plate
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(annotated, "License Plate", (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # ---- Stage 2: Detect details inside plate ----
                    results_details = self.plate_detector.detect_plate_details(plate_crop, conf_threshold/2)

                    for dr in results_details:
                        for dbox in dr.boxes:
                            dx1, dy1, dx2, dy2 = map(int, dbox.xyxy[0].cpu().numpy())
                            cls_id = int(dbox.cls[0].cpu().numpy())
                            conf = float(dbox.conf[0].cpu().numpy())

                            detail_crop = plate_crop[dy1:dy2, dx1:dx2]
                            label = dr.names[cls_id]  # "text", "city", "number"

                            if detail_crop.size > 0 and self.enable_ocr_check.isChecked():
                                # Run OCR on each class
                                ocr_results = self.text_recognizer.recognize_text(detail_crop)
                                text_detected = " ".join([res[1] for res in ocr_results])

                                if label == "text":
                                    arabic_text = self.text_recognizer.process_plate_text(text_detected)
                                    color = (255, 0, 0)  # Red for text
                                elif label == "number":
                                    arabic_num, eng_num, arabic_to_eng = self.text_recognizer.extract_numbers(text_detected)
                                    arabic_number = arabic_num
                                    english_number = eng_num if eng_num else arabic_to_eng
                                    color = (0, 0, 255)  # Blue for numbers
                                elif label == "city":
                                    city_arabic, city_eng, city_arabic_to_eng = self.text_recognizer.extract_numbers(text_detected)
                                    city_val = city_arabic if city_arabic else (city_eng if city_eng else text_detected)
                                    color = (255, 255, 0)  # Yellow for city

                                # Draw rectangle on annotated plate
                                cv2.rectangle(plate_crop, (dx1, dy1), (dx2, dy2), color, 2)
                                cv2.putText(plate_crop, f"{label}: {text_detected}", 
                                          (dx1, dy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    # ---- Fallback: Full plate OCR ----
                    if self.enable_ocr_check.isChecked():
                        full_ocr_results = self.text_recognizer.recognize_text(plate_crop)
                        full_plate_text = " ".join([res[1] for res in full_ocr_results])
                        
                        # If no specific numbers found, try to extract from full plate
                        if not arabic_number and not english_number:
                            arabic_num, eng_num, arabic_to_eng = self.text_recognizer.extract_numbers(full_plate_text)
                            arabic_number = arabic_num
                            english_number = eng_num if eng_num else arabic_to_eng
                        
                        # If no city found, try to extract from full plate
                        if not city_val:
                            city_arabic, city_eng, city_arabic_to_eng = self.text_recognizer.extract_numbers(full_plate_text)
                            city_val = city_arabic if city_arabic else (city_eng if city_eng else "Not detected")

                    # Replace original image plate with annotated plate
                    annotated[y1:y2, x1:x2] = plate_crop

            # Store the annotated frame for display
            self.annotated_frame = annotated.copy()

            # Update GUI fields
            self.city_field.setText(city_val if city_val else "Not detected")
            self.text_field.setText(arabic_text if arabic_text else "Not detected")
            self.arabic_number_field.setText(arabic_number if arabic_number else "Not detected")
            self.english_number_field.setText(english_number if english_number else "Not detected")
            
            # Full plate OCR results
            self.full_ocr_output.setPlainText(full_plate_text if full_plate_text else "No text detected")
            
            # Traffic summary
            # Traffic summary
            summary_text = ""
            summary_data = {
                "timestamp": datetime.now().isoformat(),
                "source": "live_camera",
                "plate_detected": plate_detected,
                "city": city_val if city_val else "Unknown",
                "vehicle_type": arabic_text if arabic_text else "Unknown",
                "arabic_number": arabic_number if arabic_number else "Not detected",
                "english_number": english_number if english_number else "Not detected",
                "full_text": full_plate_text if full_plate_text else "No text detected"
            }
            
            if plate_detected:
                summary_text += f"✅ Plate Detected (Live)\n"
                summary_text += f"City: {summary_data['city']}\n"
                summary_text += f"Type: {summary_data['vehicle_type']}\n"
                if arabic_number:
                    summary_text += f"Arabic: {arabic_number}\n"
                if english_number:
                    summary_text += f"English: {english_number}\n"
            else:
                summary_text = "❌ No plate detected in live feed"
                summary_data["plate_detected"] = False
            
            self.summary_output.setPlainText(summary_text)
            
            # Publish to MQTT if enabled and plate detected
            if plate_detected and self.mqtt_enable_check.isChecked() and self.mqtt_connected:
                self.publish_to_mqtt(summary_data)
            
            # Add to history
            if plate_detected and self.save_results_check.isChecked():
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                history_entry = f"Live Camera - {timestamp}\n{summary_text}\n{'-'*50}\n"
                self.history_text.append(history_entry)
                    
        except Exception as e:
            print(f"Error processing camera frame: {e}")
        finally:
            self.processing_frame = False