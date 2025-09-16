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
MODEL_PLATE = "src/models/yolov8s14/weights/best.pt"

# Second model: detects 'text', 'city', 'number'
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


class PlateApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚗 Yemeni Plate Recognition (Two-Stage)")
        self.setGeometry(200, 200, 1100, 700)

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

        # City number field (on the left)
        city_group = QGroupBox("City Number")
        city_layout = QVBoxLayout()
        self.city_output = QTextEdit()
        self.city_output.setReadOnly(True)
        city_layout.addWidget(self.city_output)
        city_group.setLayout(city_layout)

        # Text + Numbers field (on the right, stacked)
        details_group = QGroupBox("Plate Details")
        details_layout = QVBoxLayout()

        self.text_output = QTextEdit()
        self.text_output.setPlaceholderText("Arabic Text")
        self.text_output.setReadOnly(True)

        self.number_output = QTextEdit()
        self.number_output.setPlaceholderText("Plate Number (Arabic / English)")
        self.number_output.setReadOnly(True)

        details_layout.addWidget(QLabel("Arabic Text"))
        details_layout.addWidget(self.text_output)
        details_layout.addWidget(QLabel("Plate Number"))
        details_layout.addWidget(self.number_output)
        details_group.setLayout(details_layout)

        info_layout.addWidget(city_group, 1)
        info_layout.addWidget(details_group, 3)

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

        city_val = ""
        arabic_text = ""
        arabic_number = ""

        # ---- Stage 1: Detect Plate ----
        results_plate = plate_model(img)
        for r in results_plate:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                plate_crop = img[y1:y2, x1:x2]

                # ---- Stage 2: Detect details ----
                results_details = details_model(plate_crop)

                for dr in results_details:
                    for dbox in dr.boxes:
                        dx1, dy1, dx2, dy2 = map(int, dbox.xyxy[0].cpu().numpy())
                        cls_id = int(dbox.cls[0].cpu().numpy())
                        label = dr.names[cls_id]  # "text", "city", "number"

                        detail_crop = plate_crop[dy1:dy2, dx1:dx2]
                        text_detected = ""

                        if detail_crop.size > 0:
                            if label in ["text", "number", "city"]:
                                ocr_results = reader.readtext(detail_crop)
                                text_detected = " ".join([res[1] for res in ocr_results])

                                # Fix common misread
                                if label == "text":
                                    if "خصوصي" in text_detected or "نقل" in text_detected:
                                        text_detected = "خصوصي نقل اجرة"
                                    arabic_text = text_detected

                                elif label == "number":
                                    arabic_number = text_detected

                                elif label == "city":
                                    city_val = text_detected

                            # Draw rectangle
                            cv2.rectangle(plate_crop, (dx1, dy1), (dx2, dy2), (255, 0, 0), 2)
                            cv2.putText(
                                plate_crop, f"{label}: {text_detected}",
                                (dx1, dy1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 0, 0), 2
                            )

                # Replace annotated plate back
                annotated[y1:y2, x1:x2] = plate_crop

        # ---- Update GUI fields ----
        self.city_output.setPlainText(city_val if city_val else "Not detected")
        self.text_output.setPlainText(arabic_text if arabic_text else "Not detected")

        if arabic_number:
            english_number = arabic_to_english_digits(arabic_number)
            self.number_output.setPlainText(f"{arabic_number}\n{english_number}")
        else:
            self.number_output.setPlainText("Not detected")

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
