from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QTextEdit
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
import cv2

class ImageTab(QWidget):
    def __init__(self, plate_detector, plate_recognizer):
        super().__init__()
        self.detector = plate_detector
        self.recognizer = plate_recognizer
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()

        # Buttons
        self.btn_upload = QPushButton("Upload Image")
        self.btn_upload.clicked.connect(self.upload_image)

        # Image display
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)

        # Text output
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setMaximumHeight(100)

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
        detections = self.detector.detect(img)

        detected_texts = []
        annotated = img.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            crop = detection['crop']

            if crop.size > 0:
                text_detected = self.recognizer.recognize(crop)
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
            self.image_label.width(), 
            self.image_label.height(),
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))

        # Show extracted text
        self.text_output.setPlainText("\n".join(detected_texts) if detected_texts else "No plate detected")