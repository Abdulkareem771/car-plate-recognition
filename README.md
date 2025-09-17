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

- **Ahmed Al-Duais** - 202270176
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

### GUI Applications

The project includes multiple GUI applications for different use cases and complexity levels:

#### 1. Advanced Qt GUI with Live Stream (Recommended)

**File:** `GUIs/qt_with_live/main.py`

Launch the most comprehensive GUI with both image processing and live stream capabilities:

```bash
python GUIs/qt_with_live/main.py
```

**Features:**
- **Dual-tab interface**: Image processing and live stream tabs
- **Image Processing Tab**: Upload and process single images
- **Live Stream Tab**: Real-time video stream processing with configurable settings
- **Stream Configuration**: Customizable stream URL and confidence threshold
- **Real-time OCR**: Arabic and English text recognition
- **Modern Interface**: Professional Qt-based UI with responsive design

**How to Use:**
1. Launch the application
2. **For Image Processing**: Click "Upload Image" button, select an image file
3. **For Live Stream**: 
   - Enter your stream URL (supports HTTP, RTSP, RTMP, TCP)
   - Set confidence threshold (0.1-0.9)
   - Click "Start Stream" to begin real-time processing
   - Click "Stop Stream" to end processing

#### 2. Simple Qt GUI (Single Image Processing)

**File:** `GUIs/simple-gui-qt.py`

A streamlined Qt interface for basic image processing:

```bash
python GUIs/simple-gui-qt.py
```

**Features:**
- Simple image upload interface
- Direct OCR text recognition
- Arabic text correction for common misreads
- Clean, minimal interface

#### 3. Advanced Two-Stage Recognition GUI

**File:** `yemen_plate_gui.py`

Advanced GUI with two-stage detection (plate detection + detailed component recognition):

```bash
python yemen_plate_gui.py
```

**Features:**
- **Two-stage detection**: First detects plates, then analyzes components
- **Component classification**: Separates city, text, and number components
- **Arabic digit conversion**: Converts Arabic-Indic digits to English
- **Detailed results display**: Shows city number, Arabic text, and plate numbers separately

#### 4. Enhanced Two-Stage Recognition GUI (Latest)

**File:** `yemen_plate_v2_gui.py`

The most advanced GUI with enhanced Arabic OCR capabilities:

```bash
python yemen_plate_v2_gui.py
```

**Features:**
- **Enhanced Arabic OCR**: Multiple preprocessing approaches for better Arabic text recognition
- **Fallback OCR**: Direct OCR on clean plate images when class-based detection fails
- **Advanced preprocessing**: CLAHE, bilateral filtering, sharpening, and morphological operations
- **Multi-approach text extraction**: Tries multiple image preprocessing methods
- **Comprehensive results**: Shows both class-based and fallback detection results
- **Arabic text cleaning**: Removes OCR artifacts and fixes common misreads
- **Number extraction**: Separates Arabic and English digits with conversion

#### 5. Simple Tkinter GUI

**File:** `GUIs/simple_gui_tiknter/simple_gui_tiknter.py`

Lightweight interface using Tkinter:

```bash
python GUIs/simple_gui_tiknter/simple_gui_tiknter.py
```

**Features:**
- Simple image upload interface
- Basic detection visualization with bounding boxes
- Status updates and error handling
- Model path auto-detection
- Easy to modify and extend

### Command Line Tools

#### 6. Live Stream Processing

**File:** `recognition-with-live-stream/live-stream.py`

Process live video streams from command line:

```bash
python recognition-with-live-stream/live-stream.py
```

**Configuration:**
- Update `STREAM_URL` in the script with your video stream URL
- Adjust confidence threshold as needed
- Supports HTTP, RTSP, and other streaming protocols

#### 7. Batch Image Processing

**File:** `model_test.py`

Process multiple images in batch:

```bash
python model_test.py
```

**Features:**
- Processes all images in a directory
- Saves annotated results to output folder
- Automatic model loading and processing

#### 8. Plate Cropping Utility

**File:** `seperated-plates.py`

Extract and save individual license plates from images:

```bash
python seperated-plates.py
```

**Features:**
- Crops detected plates from training images
- Saves each plate as a separate image file
- Useful for creating OCR training datasets

### GUI Application Comparison

| GUI Application | Complexity | Features | Best For |
|----------------|------------|----------|----------|
| **qt_with_live/main.py** | Advanced | Live stream + Image processing | Real-time applications |
| **yemen_plate_v2_gui.py** | Advanced | Two-stage detection + Enhanced OCR | Research and detailed analysis |
| **yemen_plate_gui.py** | Intermediate | Two-stage detection | Component analysis |
| **simple-gui-qt.py** | Basic | Single image processing | Quick testing |
| **simple_gui_tiknter.py** | Basic | Simple detection | Learning and development |

### How to Launch Each GUI

1. **For Real-time Processing**: Use `GUIs/qt_with_live/main.py`
2. **For Advanced Analysis**: Use `yemen_plate_v2_gui.py`
3. **For Quick Testing**: Use `GUIs/simple-gui-qt.py`
4. **For Learning**: Use `GUIs/simple_gui_tiknter/simple_gui_tiknter.py`

### Command Line Usage

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

#### Two-Stage Detection (Advanced)
```python
from ultralytics import YOLO
import easyocr

# Load both models
plate_model = YOLO("src/models/yolov8s14/weights/best.pt")
details_model = YOLO("src/models/yolov8s5/weights/best.pt")
reader = easyocr.Reader(['en', 'ar'])

# Stage 1: Detect plates
plate_results = plate_model(image)
for r in plate_results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        plate_crop = image[y1:y2, x1:x2]
        
        # Stage 2: Detect components
        detail_results = details_model(plate_crop)
        for dr in detail_results:
            for dbox in dr.boxes:
                dx1, dy1, dx2, dy2 = map(int, dbox.xyxy[0].cpu().numpy())
                cls_id = int(dbox.cls[0].cpu().numpy())
                label = dr.names[cls_id]  # "text", "city", "number"
                
                # Extract text using OCR
                detail_crop = plate_crop[dy1:dy2, dx1:dx2]
                ocr_results = reader.readtext(detail_crop)
                text = " ".join([res[1] for res in ocr_results])
                print(f"{label}: {text}")
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
│   │   ├── plat number car yemen.v1i.yolov8/  # Primary dataset (single class)
│   │   └── yemen-plate/               # Extended dataset (3 classes)
│   ├── models/                        # Pre-trained models
│   │   ├── yolov8s14/                # Main trained model (plate detection)
│   │   ├── yolov8s5/                 # Secondary model (component detection)
│   │   └── *.pt                      # Base YOLO models
│   └── utils/                         # Utility functions and results
│       └── results/                   # Processed images and cropped plates
├── GUIs/                             # User interfaces
│   ├── qt_with_live/                 # Advanced Qt GUI with live stream
│   │   ├── components/               # GUI components (image_tab.py, live_stream_tab.py)
│   │   └── utils/                    # Detection utilities (plate_detector.py, plate_recognizer.py)
│   └── simple_gui_tiknter/           # Simple Tkinter GUI
├── runs/detect/                      # Training results and experiments
│   ├── yolov8n14/                    # Latest training run (main model)
│   ├── yolov8s5/                     # Component detection model
│   └── [multiple training runs]/     # Various training experiments
├── recognition-with-live-stream/     # Live stream processing
├── Car_Plate_Recognition_Project.ipynb  # Jupyter notebook with analysis
├── main.py                           # Training script
├── model_test.py                     # Batch image processing
├── seperated-plates.py               # Plate cropping utility
├── convert_to_yolo.py                # CSV to YOLO format converter
├── yemen car plate number model train.py  # Training script for extended dataset
├── yemen_plate_gui.py                # Two-stage recognition GUI
├── yemen_plate_v2_gui.py             # Enhanced two-stage GUI (latest)
├── GUIs/simple-gui-qt.py             # Simple Qt GUI
├── GUIs/simple_gui_tiknter.py        # Simple Tkinter GUI (root level)
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

## Additional Features

### Two-Stage Detection System

The project implements a sophisticated two-stage detection system:

1. **Stage 1 - Plate Detection**: Uses `yolov8s14` model to detect license plates in images
2. **Stage 2 - Component Analysis**: Uses `yolov8s5` model to analyze plate components (text, city, number)

### Enhanced Arabic OCR

The latest GUI (`yemen_plate_v2_gui.py`) includes advanced Arabic text recognition:

- **Multiple Preprocessing Approaches**: CLAHE, bilateral filtering, sharpening
- **Arabic Text Cleaning**: Removes OCR artifacts and fixes common misreads
- **Digit Conversion**: Converts Arabic-Indic digits (٠١٢٣٤٥٦٧٨٩) to English (0123456789)
- **Fallback OCR**: Direct text extraction when class-based detection fails

### Utility Scripts

- **`convert_to_yolo.py`**: Converts CSV annotations to YOLO format
- **`seperated-plates.py`**: Extracts individual plates from images for OCR training
- **`model_test.py`**: Batch processes multiple images for testing

### Training Scripts

- **`main.py`**: Primary training script for plate detection
- **`yemen car plate number model train.py`**: Training script for component detection
- **`Car_Plate_Recognition_Project.ipynb`**: Comprehensive analysis notebook

## Acknowledgments

- Ultralytics for the YOLO implementation
- EasyOCR team for the OCR capabilities
- The computer vision community for open-source tools and datasets

