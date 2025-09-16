import sys
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import os
from pathlib import Path


from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QTextEdit
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

# ===== USER SETTINGS =====
script_dir = Path(__file__).parent
modelpath= script_dir.parent / "src" / "models" / "yolov8s14" / "weights" / "best.pt"
MODEL_PATH = modelpath.resolve()

# Load YOLO model and OCR
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en', 'ar'])

class PlateApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚗 License Plate Recognition")
        self.setGeometry(200, 200, 900, 600)

        layout = QVBoxLayout()

        # Buttons
        self.btn_upload = QPushButton("Upload Image")
        self.btn_upload.clicked.connect(self.upload_image)

        # Image display
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)

        # Text output
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)

        layout.addWidget(self.btn_upload)
        layout.addWidget(self.image_label)
        layout.addWidget(QLabel("Extracted Plate Text:"))
        layout.addWidget(self.text_output)

        self.setLayout(layout)

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select an image", "", "Images (*.jpg *.jpeg *.png)"
        )
        if not file_path:
            return

        img = cv2.imread(file_path)
        results = model(img)

        detected_texts = []
        annotated = img.copy()

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                crop = img[y1:y2, x1:x2]

                if crop.size > 0:
                    ocr_results = reader.readtext(crop)
                    text_detected = " ".join([res[1] for res in ocr_results])

                    # ---- FIX common misread word ----
                    if "خصوصي" in text_detected or "نقل" in text_detected:
                        text_detected = "خصوصي نقل اجرة"

                    detected_texts.append(text_detected)

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, text_detected, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Convert to QImage
        rgb_image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        self.image_label.setPixmap(pixmap.scaled(
            800, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

        # Show extracted text
        self.text_output.setPlainText("\n".join(detected_texts) if detected_texts else "No plate detected")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlateApp()
    window.show()
    sys.exit(app.exec())
