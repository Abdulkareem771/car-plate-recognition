import sys
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFileDialog, QTextEdit, QGroupBox, QGridLayout, QLineEdit, QFrame
)
from PySide6.QtGui import QPixmap, QImage, QFont
from PySide6.QtCore import Qt

# ===== USER SETTINGS =====
# First model: detects plates
MODEL_PLATE = "src/models/yolov8s14/weights/best.pt"  #new path

# Second model: detects 'text', 'city', 'number' inside the plate
MODEL_PLATE_DETAILS = "src/models/yolov8s5/weights/best.pt"

# Load models
plate_model = YOLO(MODEL_PLATE)
details_model = YOLO(MODEL_PLATE_DETAILS)

# OCR
reader = easyocr.Reader(['en', 'ar'])

# Helper: convert Arabic-Indic digits to English
def arabic_to_english_digits(text):
    mapping = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    return text.translate(mapping)

def clean_arabic_text(text):
    """Clean Arabic text from OCR artifacts"""
    if not text:
        return text
    
    # Remove common OCR artifacts
    text = text.replace('?', '').replace('؟', '')
    text = text.replace('be', '').replace('4ق', 'ق').replace('nب', 'ب').replace('itق', 'ق')
    
    # Remove English letters mixed with Arabic
    import re
    text = re.sub(r'[a-zA-Z]', '', text)
    
    # Fix common Arabic text patterns
    if "خصوصي" in text or "نقل" in text:
        text = "خصوصي نقل اجرة"
    
    return text.strip()

def extract_clean_numbers(text):
    """Extract and clean numbers from text - returns both Arabic and English"""
    if not text:
        return "", "", ""
    
    # Extract Arabic-Indic digits
    arabic_digits = ''.join([c for c in text if c in "٠١٢٣٤٥٦٧٨٩"])
    # Extract English digits
    english_digits = ''.join([c for c in text if c.isdigit()])
    
    # Convert Arabic to English
    arabic_to_eng = arabic_to_english_digits(arabic_digits) if arabic_digits else ""
    
    return arabic_digits, english_digits, arabic_to_eng

class PlateApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚔 Yemeni Traffic Plate Recognition System")
        self.setGeometry(100, 100, 1400, 900)

        # Apply dark mode theme
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QPushButton {
                background-color: #404040;
                border: 2px solid #555555;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #505050;
                border-color: #666666;
            }
            QPushButton:pressed {
                background-color: #353535;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 8px;
                margin: 5px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLineEdit {
                background-color: #404040;
                border: 2px solid #555555;
                border-radius: 5px;
                padding: 8px;
                color: #ffffff;
                font-size: 14px;
            }
            QLineEdit:focus {
                border-color: #0078d4;
            }
            QTextEdit {
                background-color: #404040;
                border: 2px solid #555555;
                border-radius: 5px;
                color: #ffffff;
                font-size: 12px;
            }
            QLabel {
                color: #ffffff;
            }
        """)

        main_layout = QHBoxLayout()

        # Left Panel: Image and Controls
        left_panel = QVBoxLayout()
        
        # Upload button
        self.btn_upload = QPushButton("📷 Upload Vehicle Image")
        self.btn_upload.setMinimumHeight(50)
        self.btn_upload.clicked.connect(self.upload_image)
        left_panel.addWidget(self.btn_upload)

        # Image display
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setStyleSheet("border: 2px solid #555555; background-color: #404040; border-radius: 8px;")
        left_panel.addWidget(self.image_label)

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

        # Add panels to main layout
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)

        self.setLayout(main_layout)

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select vehicle image", "", "Images (*.jpg *.jpeg *.png)"
        )
        if not file_path:
            return

        img = cv2.imread(file_path)
        annotated = img.copy()

        # Initialize results
        city_val = ""
        arabic_text = ""
        arabic_number = ""
        english_number = ""
        full_plate_text = ""

        # ---- Stage 1: Detect Plate ----
        results_plate = plate_model(img)
        plate_detected = False
        
        for r in results_plate:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                plate_crop = img[y1:y2, x1:x2]
                plate_detected = True

                # ---- Stage 2: Detect details inside plate ----
                results_details = details_model(plate_crop, conf=0.1)

                for dr in results_details:
                    for dbox in dr.boxes:
                        dx1, dy1, dx2, dy2 = map(int, dbox.xyxy[0].cpu().numpy())
                        cls_id = int(dbox.cls[0].cpu().numpy())
                        conf = float(dbox.conf[0].cpu().numpy())

                        detail_crop = plate_crop[dy1:dy2, dx1:dx2]
                        label = dr.names[cls_id]  # "text", "city", "number"

                        if detail_crop.size > 0:
                            # Run OCR on each class
                            ocr_results = reader.readtext(detail_crop)
                            text_detected = " ".join([res[1] for res in ocr_results])

                            if label == "text":
                                arabic_text = clean_arabic_text(text_detected)
                            elif label == "number":
                                arabic_num, eng_num, arabic_to_eng = extract_clean_numbers(text_detected)
                                arabic_number = arabic_num
                                english_number = eng_num if eng_num else arabic_to_eng
                            elif label == "city":
                                # Extract numbers from city text
                                city_arabic, city_eng, city_arabic_to_eng = extract_clean_numbers(text_detected)
                                city_val = city_arabic if city_arabic else (city_eng if city_eng else text_detected)

                            # Draw rectangle on annotated plate
                            cv2.rectangle(plate_crop, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
                            cv2.putText(plate_crop, f"{label}: {text_detected}", 
                                      (dx1, dy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # ---- Fallback: Full plate OCR ----
                print("Running full plate OCR...")
                full_ocr_results = reader.readtext(plate_crop)
                full_plate_text = " ".join([res[1] for res in full_ocr_results])
                
                # If no specific numbers found, try to extract from full plate
                if not arabic_number and not english_number:
                    arabic_num, eng_num, arabic_to_eng = extract_clean_numbers(full_plate_text)
                    arabic_number = arabic_num
                    english_number = eng_num if eng_num else arabic_to_eng
                
                # If no city found, try to extract from full plate
                if not city_val:
                    city_arabic, city_eng, city_arabic_to_eng = extract_clean_numbers(full_plate_text)
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
        if plate_detected:
            summary_text += f"✅ Plate Detected\n"
            summary_text += f"City: {city_val if city_val else 'Unknown'}\n"
            summary_text += f"Type: {arabic_text if arabic_text else 'Unknown'}\n"
            if arabic_number:
                summary_text += f"Arabic: {arabic_number}\n"
            if english_number:
                summary_text += f"English: {english_number}\n"
        else:
            summary_text = "❌ No plate detected in image"
        
        self.summary_output.setPlainText(summary_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlateApp()
    window.show()
    sys.exit(app.exec())
