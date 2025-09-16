import sys
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QGroupBox
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

# ===== USER SETTINGS =====
# First model: detects plates
MODEL_PLATE = "src/models/yolov8s14/weights/best.pt"##new path

# Second model: detects 'text', 'city', 'number'
MODEL_PLATE_DETAILS = "src/models/yolov8s5/weights/best.pt"

# Load models
plate_model = YOLO(MODEL_PLATE)
details_model = YOLO(MODEL_PLATE_DETAILS)

# OCR with better settings for Arabic
reader = easyocr.Reader(['en', 'ar'], gpu=False)

# Helper: convert Arabic-Indic digits to English
def arabic_to_english_digits(text):
    mapping = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    return text.translate(mapping)

def preprocess_for_arabic_ocr(image):
    """Enhanced preprocessing specifically for Arabic text recognition"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize image if too small for better OCR
    height, width = gray.shape
    if height < 50 or width < 100:
        scale_factor = max(50/height, 100/width, 2.0)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    
    # Apply bilateral filter to preserve edges
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Apply sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(filtered, -1, kernel)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Morphological operations to clean up
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Convert back to BGR
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

def extract_text_from_image(image, confidence_threshold=0.2):
    """Extract all text from image with multiple preprocessing approaches"""
    results = []
    
    # Try multiple preprocessing approaches for better Arabic recognition
    approaches = [
        ("preprocessed", preprocess_for_arabic_ocr(image)),
        ("original", image),
        ("grayscale", cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)),
        ("enhanced", cv2.convertScaleAbs(image, alpha=1.5, beta=30)),
        ("inverted", cv2.bitwise_not(image))
    ]
    
    for approach_name, processed_img in approaches:
        try:
            # Use different OCR settings for Arabic
            ocr_results = reader.readtext(processed_img, 
                                        paragraph=False,  # Don't group into paragraphs
                                        width_ths=0.7,   # Lower width threshold
                                        height_ths=0.7,  # Lower height threshold
                                        detail=1)        # Get detailed results
            
            for result in ocr_results:
                if result[2] > confidence_threshold:  # confidence threshold
                    results.append({
                        'text': result[1],
                        'confidence': result[2],
                        'approach': approach_name
                    })
        except Exception as e:
            print(f"OCR error with {approach_name}: {e}")
    
    return results

def clean_arabic_text(text):
    """Clean and fix common Arabic OCR errors"""
    if not text:
        return text
    
    # Remove common OCR artifacts and noise
    text = text.replace('?', '').replace('؟', '')
    text = text.replace('be', '').replace('ق', 'ق')  # Keep Arabic characters
    text = text.replace('4ق', 'ق').replace('nب', 'ب').replace('itق', 'ق')
    
    # Remove English letters mixed with Arabic
    import re
    text = re.sub(r'[a-zA-Z]', '', text)
    
    # Fix common Arabic text patterns
    replacements = {
        'خصوصي نقل اجرة': 'خصوصي نقل اجرة',
        'نقل اجرة': 'خصوصي نقل اجرة',
        'خصوصي': 'خصوصي نقل اجرة',
        'نقل': 'خصوصي نقل اجرة'
    }
    
    for wrong, correct in replacements.items():
        if wrong in text:
            text = correct
            break
    
    return text.strip()

def extract_numbers_from_text(text):
    """Extract all numbers from text (both Arabic and English)"""
    if not text:
        return []
    
    numbers = []
    
    # Extract Arabic-Indic digits (٠١٢٣٤٥٦٧٨٩)
    arabic_digits = ''.join([c for c in text if c in "٠١٢٣٤٥٦٧٨٩"])
    if arabic_digits:
        numbers.append(('arabic', arabic_digits))
    
    # Extract English digits
    english_digits = ''.join([c for c in text if c.isdigit()])
    if english_digits:
        numbers.append(('english', english_digits))
    
    # Also look for mixed numbers (Arabic + English)
    mixed_digits = ''.join([c for c in text if c.isdigit() or c in "٠١٢٣٤٥٦٧٨٩"])
    if mixed_digits and mixed_digits != arabic_digits and mixed_digits != english_digits:
        numbers.append(('mixed', mixed_digits))
    
    return numbers

class PlateApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚗 Yemeni Plate Recognition")
        self.setGeometry(200, 200, 1200, 800)

        main_layout = QVBoxLayout()

        # ---- Button ----
        self.btn_upload = QPushButton("Upload Image")
        self.btn_upload.clicked.connect(self.upload_image)
        main_layout.addWidget(self.btn_upload)

        # ---- Image display ----
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.image_label)

        # ---- Info fields ----
        info_layout = QHBoxLayout()

        # Left column: City and Fallback
        left_group = QGroupBox("Detection Results")
        left_layout = QVBoxLayout()

        # City number field
        city_layout = QVBoxLayout()
        city_layout.addWidget(QLabel("City Number (from classes)"))
        self.city_output = QTextEdit()
        self.city_output.setReadOnly(True)
        self.city_output.setMaximumHeight(60)
        city_layout.addWidget(self.city_output)

        # Fallback field
        fallback_layout = QVBoxLayout()
        fallback_layout.addWidget(QLabel("Fallback Text (direct OCR)"))
        self.fallback_output = QTextEdit()
        self.fallback_output.setReadOnly(True)
        self.fallback_output.setMaximumHeight(100)
        fallback_layout.addWidget(self.fallback_output)

        left_layout.addLayout(city_layout)
        left_layout.addLayout(fallback_layout)
        left_group.setLayout(left_layout)

        # Right column: Text and Numbers
        right_group = QGroupBox("Plate Details")
        right_layout = QVBoxLayout()

        self.text_output = QTextEdit()
        self.text_output.setPlaceholderText("Arabic Text")
        self.text_output.setReadOnly(True)
        self.text_output.setMaximumHeight(80)

        self.number_output = QTextEdit()
        self.number_output.setPlaceholderText("Plate Number (Arabic / English)")
        self.number_output.setReadOnly(True)
        self.number_output.setMaximumHeight(80)

        right_layout.addWidget(QLabel("Arabic Text"))
        right_layout.addWidget(self.text_output)
        right_layout.addWidget(QLabel("Plate Number"))
        right_layout.addWidget(self.number_output)
        right_group.setLayout(right_layout)

        info_layout.addWidget(left_group, 1)
        info_layout.addWidget(right_group, 1)

        main_layout.addLayout(info_layout)
        self.setLayout(main_layout)

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select an image", "", "Images (*.jpg *.jpeg *.png)"
        )
        if not file_path:
            return

        img = cv2.imread(file_path)
        annotated = img.copy()

        # Store results for all plates
        all_results = []

        # ---- Stage 1: Detect Plate ----
        results_plate = plate_model(img)
        print(f"Detected {len(results_plate)} plate results")
        
        plate_count = 0
        for r in results_plate:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    plate_crop = img[y1:y2, x1:x2]
                    
                    if plate_crop.size > 0:
                        plate_count += 1
                        print(f"Processing plate {plate_count}")

                        # Create a clean copy for fallback OCR (no labels)
                        clean_plate_crop = plate_crop.copy()
                        
                        # ---- Stage 2: Detect details with classes ----
                        city_val = ""
                        arabic_text = ""
                        arabic_number = ""
                        
                        results_details = details_model(plate_crop, conf=0.1)  # Lower confidence
                        print(f"Details model found {len(results_details)} results")
                        
                        for dr in results_details:
                            if dr.boxes is not None:
                                for dbox in dr.boxes:
                                    dx1, dy1, dx2, dy2 = map(int, dbox.xyxy[0].cpu().numpy())
                                    cls_id = int(dbox.cls[0].cpu().numpy())
                                    label = dr.names[cls_id]
                                    confidence = float(dbox.conf[0].cpu().numpy())
                                    
                                    print(f"Found {label} with confidence {confidence:.2f}")

                                    detail_crop = plate_crop[dy1:dy2, dx1:dx2]
                                    
                                    if detail_crop.size > 0:
                                        # Enhanced OCR for each class
                                        ocr_results = extract_text_from_image(detail_crop)
                                        
                                        if ocr_results:
                                            # Get the best result
                                            best_result = max(ocr_results, key=lambda x: x['confidence'])
                                            text_detected = best_result['text']
                                            
                                            print(f"OCR result for {label}: '{text_detected}' (confidence: {best_result['confidence']:.2f})")
                                            
                                            if label == "text":
                                                arabic_text = clean_arabic_text(text_detected)
                                            elif label == "number":
                                                arabic_number = text_detected
                                            elif label == "city":
                                                city_val = text_detected

                                        # Draw rectangle on annotated version
                                        cv2.rectangle(plate_crop, (dx1, dy1), (dx2, dy2), (255, 0, 0), 2)
                                        cv2.putText(
                                            plate_crop, f"{label}: {text_detected if 'text_detected' in locals() else 'No text'}",
                                            (dx1, dy1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (255, 0, 0), 1
                                        )

                        # ---- Fallback: Direct OCR on clean cropped plate (no labels) ----
                        print("Running fallback OCR on clean plate (no labels)...")
                        fallback_results = extract_text_from_image(clean_plate_crop, confidence_threshold=0.2)
                        fallback_text = ""
                        fallback_numbers = []
                        
                        for result in fallback_results:
                            fallback_text += result['text'] + " "
                            # Extract numbers from fallback text
                            numbers = extract_numbers_from_text(result['text'])
                            fallback_numbers.extend(numbers)
                        
                        fallback_text = fallback_text.strip()
                        
                        # Store results
                        plate_result = {
                            'plate_num': plate_count,
                            'city': city_val,
                            'text': arabic_text,
                            'number': arabic_number,
                            'fallback_text': fallback_text,
                            'fallback_numbers': fallback_numbers
                        }
                        all_results.append(plate_result)

                        # Replace annotated plate back
                        annotated[y1:y2, x1:x2] = plate_crop

        # ---- Update GUI fields ----
        if all_results:
            # Display results from all plates
            city_text = ""
            text_text = ""
            number_text = ""
            fallback_text = ""
            
            for result in all_results:
                plate_num = result['plate_num']
                city = result['city'] if result['city'] else "Not detected"
                text = result['text'] if result['text'] else "Not detected"
                number = result['number'] if result['number'] else "Not detected"
                fallback = result['fallback_text'] if result['fallback_text'] else "No text detected"
                fallback_nums = result['fallback_numbers']
                
                if len(all_results) > 1:
                    city_text += f"Plate {plate_num}: {city}\n"
                    text_text += f"Plate {plate_num}: {text}\n"
                    if number != "Not detected":
                        english_number = arabic_to_english_digits(number)
                        # Extract numbers from the detected text
                        numbers = extract_numbers_from_text(number)
                        number_text += f"Plate {plate_num}:\n  Raw: {number}\n"
                        if numbers:
                            for num_type, num_value in numbers:
                                if num_type == 'arabic':
                                    eng_converted = arabic_to_english_digits(num_value)
                                    number_text += f"  Arabic: {num_value} → English: {eng_converted}\n"
                                elif num_type == 'english':
                                    number_text += f"  English: {num_value}\n"
                                elif num_type == 'mixed':
                                    eng_converted = arabic_to_english_digits(num_value)
                                    number_text += f"  Mixed: {num_value} → English: {eng_converted}\n"
                        number_text += "\n"
                    else:
                        number_text += f"Plate {plate_num}: {number}\n\n"
                    fallback_text += f"Plate {plate_num}: {fallback}\n"
                    if fallback_nums:
                        fallback_text += f"  Numbers found: {fallback_nums}\n"
                    fallback_text += "\n"
                else:
                    city_text = city
                    text_text = text
                    if number != "Not detected":
                        # Extract numbers from the detected text
                        numbers = extract_numbers_from_text(number)
                        number_text = f"Raw: {number}\n"
                        if numbers:
                            for num_type, num_value in numbers:
                                if num_type == 'arabic':
                                    eng_converted = arabic_to_english_digits(num_value)
                                    number_text += f"Arabic: {num_value} → English: {eng_converted}\n"
                                elif num_type == 'english':
                                    number_text += f"English: {num_value}\n"
                                elif num_type == 'mixed':
                                    eng_converted = arabic_to_english_digits(num_value)
                                    number_text += f"Mixed: {num_value} → English: {eng_converted}\n"
                        else:
                            number_text = f"Arabic: {number}\nEnglish: {arabic_to_english_digits(number)}"
                    else:
                        number_text = number
                    fallback_text = fallback
                    if fallback_nums:
                        fallback_text += f"\nNumbers found: {fallback_nums}"
            
            self.city_output.setPlainText(city_text)
            self.text_output.setPlainText(text_text)
            self.number_output.setPlainText(number_text)
            self.fallback_output.setPlainText(fallback_text)
        else:
            # No plates detected
            self.city_output.setPlainText("No plates detected")
            self.text_output.setPlainText("No plates detected")
            self.number_output.setPlainText("No plates detected")
            self.fallback_output.setPlainText("No plates detected")

        # ---- Show annotated image ----
        rgb_image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        self.image_label.setPixmap(
            pixmap.scaled(1000, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlateApp()
    window.show()
    sys.exit(app.exec())
