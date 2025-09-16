import sys
import os
from pathlib import Path

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTabWidget

# Import our components
from components.image_tab import ImageTab
from components.live_stream_tab import LiveStreamTab
from utils.plate_detector import PlateDetector
from utils.plate_recognizer import PlateRecognizer

class PlateApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚗 License Plate Recognition")
        self.setGeometry(200, 200, 900, 700)
        
        # Initialize models
        # Go up two levels to get to the parent of GUIs folder, then to src/models
        script_dir = Path(__file__).parent
        model_path = script_dir.parent.parent/ "src" / "models" / "yolov8s14" / "weights" / "best.pt"
        
        self.detector = PlateDetector(str(model_path))
        self.recognizer = PlateRecognizer()
        
        self.init_ui()
        
    def init_ui(self):
        # Create tabs
        self.tabs = QTabWidget()
        
        # Image tab
        self.image_tab = ImageTab(self.detector, self.recognizer)
        self.tabs.addTab(self.image_tab, "Image Recognition")
        
        # Live stream tab
        self.stream_tab = LiveStreamTab(self.detector, self.recognizer)
        self.tabs.addTab(self.stream_tab, "Live Stream")
        
        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)
        
    def closeEvent(self, event):
        """Handle application close"""
        self.stream_top.stop_stream()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlateApp()
    window.show()
    sys.exit(app.exec())