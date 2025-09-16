import sys
import os
from pathlib import Path

# Add the src directory to the Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from PySide6.QtWidgets import QApplication
from src.gui.main_window import PlateApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlateApp()
    window.show()
    sys.exit(app.exec())