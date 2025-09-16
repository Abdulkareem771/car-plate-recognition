from pathlib import Path

# Get the project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Model paths - adjust these according to your actual model locations
MODEL_PLATE = PROJECT_ROOT / "models" / "yolov8s14" / "weights" / "best.pt"
MODEL_PLATE_DETAILS = PROJECT_ROOT / "models" / "yolov8s5" / "weights" / "best.pt"

# OCR settings
OCR_LANGUAGES = ['en', 'ar']

# Camera settings
DEFAULT_CAMERA_INDEX = 0
RTSP_FORMATS = [
    "rtsp://{username}:{password}@{ip}:{port}/path",
    "http://{ip}:{port}/video",
    "https://{ip}:{port}/video"
]