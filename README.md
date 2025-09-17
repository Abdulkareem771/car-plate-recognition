# Car License Plate Detection and Recognition System

## Description

This project implements an automated car license plate detection and recognition system using computer vision and deep learning techniques. The system is designed to detect license plates from vehicle images or video streams, extract the plate region, and recognize the characters for further processing.

The project focuses specifically on Yemeni license plates and supports both Arabic and English text recognition. It provides multiple user interfaces including a modern Qt-based GUI and a simple Tkinter interface for different use cases.

### Key Features

- **Real-time License Plate Detection**: Uses YOLOv8 for accurate plate detection
- **Multi-language OCR**: Supports Arabic and English text recognition using EasyOCR
- **Multiple GUI Options**: Qt-based modern interface and simple Tkinter GUI
- **Live Stream Processing**: Real-time detection from video streams
- **Batch Image Processing**: Process multiple images at once
- **Model Training Pipeline**: Complete training setup with YOLO
- **Performance Monitoring**: Comprehensive training metrics and evaluation

## Team Documentation

### Team Members

- **Ahmed Al-duais** - 202270176
- **Abulkareem Thiab** -   202270136
- **Ayman Mrwan** - 202270324

### Project Goals

1. Develop an accurate license plate detection system using YOLO
2. Implement OCR-based text recognition for Arabic and English characters
3. Create user-friendly GUI applications for real-time processing
4. Compare different model architectures and select the optimal one
5. Provide comprehensive documentation and usage examples

## Technology Stack

- **Deep Learning**: YOLOv8 (Ultralytics)
- **OCR**: EasyOCR
- **GUI Framework**: PySide6 (Qt) & Tkinter
- **Computer Vision**: OpenCV
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Python**: 3.13.5+

## Setup Instructions

### Prerequisites

- Python 3.13.5 or higher
- CUDA-compatible GPU (recommended for training)
- Windows 10/11 (tested on Windows 10.0.22631)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd car-plate-recognition
   ```

2. **Install dependencies using UV (recommended)**
   ```bash
   # Install UV if not already installed
   pip install uv
   
   # Install project dependencies
   uv sync
   ```

3. **Alternative: Install using pip**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import ultralytics, easyocr, cv2; print('All dependencies installed successfully!')"
   ```

### Model Setup

The project includes pre-trained models in the `src/models/` directory:

- `yolov8s14/weights/best.pt` - Main trained model (YOLOv8s, 50 epochs)
- `yolov8s14/weights/last.pt` - Last checkpoint
- `yolo11n.pt`, `yolo11s.pt` - YOLO11 models
- `yolov8n.pt`, `yolov8s.pt` - Base YOLOv8 models

## Usage Examples

### 1. Qt-based GUI Application (Recommended)

Launch the modern Qt-based interface with both image processing and live stream capabilities:

```bash
python GUIs/qt_with_live/main.py
```

**Features:**
- Image upload and processing
- Live video stream detection
- Real-time OCR text recognition
- Modern, responsive interface

### 2. Simple Tkinter GUI

For a lightweight interface:

```bash
python GUIs/simple_gui_tiknter/simple_gui_tiknter.py
```

**Features:**
- Simple image upload interface
- Basic detection visualization
- Easy to use and modify

### 3. Live Stream Processing

Process live video streams:

```bash
python recognition-with-live-stream/live-stream.py
```

**Configuration:**
- Update `STREAM_URL` in the script with your video stream URL
- Adjust confidence threshold as needed
- Supports HTTP, RTSP, and other streaming protocols

### 4. Command Line Usage

#### Basic Detection
```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO("src/models/yolov8s14/weights/best.pt")

# Detect plates in image
results = model("path/to/image.jpg")

# Process results
for r in results:
    for box in r.boxes:
        conf = float(box.conf[0])
        if conf >= 0.5:  # Confidence threshold
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            print(f"Plate detected with confidence: {conf:.2f}")
            print(f"Bounding box: ({x1}, {y1}, {x2}, {y2})")
```

#### OCR Recognition
```python
import easyocr

# Initialize OCR reader
reader = easyocr.Reader(['en', 'ar'])

# Read text from plate image
ocr_results = reader.readtext(plate_image)
text_detected = " ".join([res[1] for res in ocr_results])
print(f"Detected text: {text_detected}")
```

### 5. Model Training

To train your own model:

```python
from ultralytics import YOLO

# Load base model
model = YOLO("yolov8s.pt")

# Train on your dataset
results = model.train(
    data="src/data/your_dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=9,
    device=0,
    name="custom_model"
)
```

## Project Structure

```
car-plate-recognition/
├── src/
│   ├── data/                          # Training datasets
│   │   ├── plat number car yemen.v1i.yolov8/  # Primary dataset
│   │   └── yemen-plate/               # Extended dataset
│   ├── models/                        # Pre-trained models
│   │   ├── yolov8s14/                # Main trained model
│   │   └── *.pt                      # Base models
│   └── utils/                         # Utility functions
├── GUIs/                             # User interfaces
│   ├── qt_with_live/                 # Modern Qt GUI
│   │   ├── components/               # GUI components
│   │   └── utils/                    # Detection utilities
│   └── simple_gui_tiknter/           # Simple Tkinter GUI
├── runs/detect/                      # Training results
│   └── yolov8n14/                    # Latest training run
├── recognition-with-live-stream/     # Live stream processing
├── Car_Plate_Recognition_Project.ipynb  # Jupyter notebook
├── main.py                           # Training script
└── pyproject.toml                    # Project configuration
```

## Model Performance

The final YOLOv8s model achieved the following performance metrics:

- **mAP@0.5**: 0.9214 (92.14%)
- **mAP@0.5:0.95**: 0.5942 (59.42%)
- **Precision**: 0.9369 (93.69%)
- **Recall**: 0.8838 (88.38%)
- **Training Epochs**: 50
- **Training Time**: ~8.5 hours

## Dataset Information

### Primary Dataset: `plat number car yemen.v1i.yolov8`
- **Classes**: 1 (private)
- **Training Images**: 80
- **Validation Images**: 16
- **Test Images**: 8
- **Focus**: Private vehicle license plates

### Extended Dataset: `yemen-plate`
- **Classes**: 3 (city, number, text)
- **Training Images**: 52
- **Validation Images**: 52
- **Test Images**: 52
- **Focus**: Detailed plate component annotation

## Configuration

### Model Configuration
- **Architecture**: YOLOv8s (small)
- **Input Size**: 640x640
- **Batch Size**: 9
- **Optimizer**: AdamW
- **Learning Rate**: 0.01 (with cosine annealing)
- **Data Augmentation**: Enabled

### Detection Parameters
- **Confidence Threshold**: 0.5 (adjustable)
- **NMS IoU Threshold**: 0.7
- **Max Detections**: 300

## Troubleshooting

### Common Issues

1. **Model not found error**
   - Ensure model files are in the correct directory
   - Check file paths in the scripts

2. **CUDA out of memory**
   - Reduce batch size in training
   - Use CPU mode: `device='cpu'`

3. **OCR not working**
   - Install EasyOCR: `pip install easyocr`
   - Check language support: `['en', 'ar']`

4. **GUI not launching**
   - Install PySide6: `pip install PySide6`
   - Check Python version compatibility

### Performance Optimization

- Use GPU acceleration when available
- Adjust confidence thresholds based on your use case
- Consider model quantization for deployment
- Use appropriate image sizes for your hardware

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Ultralytics for the YOLO implementation
- EasyOCR team for the OCR capabilities
- The computer vision community for open-source tools and datasets

