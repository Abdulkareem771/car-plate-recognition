import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from PIL import Image
import os
from pathlib import Path
import time
import re
import threading
from queue import Queue

# ===== USER SETTINGS =====
script_dir = Path(__file__).parent
parent_path = script_dir.parent.parent
MODEL_PLATE = parent_path / "src" / "models" / "yolov8s14" / "weights" / "best.pt"
MODEL_PLATE = str(MODEL_PLATE.resolve())
MODEL_PLATE_DETAILS = parent_path / "src" / "models" / "yolov8s5" / "weights" / "best.pt"
MODEL_PLATE_DETAILS = str(MODEL_PLATE_DETAILS.resolve())

# Load models with caching
@st.cache_resource
def load_plate_model():
    try:
        return YOLO(MODEL_PLATE)
    except Exception as e:
        st.error(f"Error loading plate detection model: {str(e)}")
        return None

@st.cache_resource
def load_details_model():
    try:
        return YOLO(MODEL_PLATE_DETAILS)
    except Exception as e:
        st.error(f"Error loading details detection model: {str(e)}")
        return None

@st.cache_resource
def load_reader():
    try:
        return easyocr.Reader(['en', 'ar'])
    except Exception as e:
        st.error(f"Error loading OCR reader: {str(e)}")
        return None

plate_model = load_plate_model()
details_model = load_details_model()
reader = load_reader()

# Stop execution if models failed to load
if plate_model is None or details_model is None or reader is None:
    st.stop()

# ===== Helper functions =====
def arabic_to_english_digits(text: str) -> str:
    mapping = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    return text.translate(mapping)

def clean_arabic_text(text: str) -> str:
    if not text:
        return text
    # Remove unwanted characters but preserve Arabic text
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\s]', '', text)
    return text.strip()

def extract_city_number(text: str):
    """Extract only the city number (usually 1-2 digits at the beginning)"""
    if not text:
        return "", ""
    
    # Find all digits (Arabic and English)
    all_digits = re.findall(r'[\d٠١٢٣٤٥٦٧٨٩]', text)
    
    # City number is typically 1-2 digits at the beginning
    city_digits = all_digits[:2] if len(all_digits) >= 2 else all_digits
    
    # Convert to strings
    arabic_city = ''.join(city_digits)
    english_city = arabic_to_english_digits(arabic_city)
    
    return arabic_city, english_city

def extract_plate_number(text: str):
    """Extract the main plate number (excluding city number)"""
    if not text:
        return "", "", ""
    
    # Remove non-digit characters except Arabic/English digits
    cleaned_text = re.sub(r'[^\d٠١٢٣٤٥٦٧٨٩]', '', text)
    
    # Extract all digits
    arabic_digits = ''.join([c for c in cleaned_text if c in "٠١٢٣٤٥٦٧٨٩"])
    english_digits = ''.join([c for c in cleaned_text if c.isdigit()])
    
    # Convert Arabic digits to English
    arabic_to_eng = arabic_to_english_digits(arabic_digits) if arabic_digits else ""
    
    # Combine both English digit sources
    final_english = english_digits + arabic_to_eng
    
    return arabic_digits, english_digits, final_english

def process_frame(frame, plate_model, details_model, reader):
    """Process a single frame for plate detection"""
    annotated = frame.copy()
    city_arabic, city_english, arabic_text, arabic_number, english_number, full_plate_text = "", "", "", "", "", ""
    plate_detected = False

    # ---- Stage 1: Detect Plate ----
    results_plate = plate_model(frame)
    
    for r in results_plate:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            plate_crop = frame[y1:y2, x1:x2]
            plate_detected = True

            # ---- Stage 2: Detect details inside plate ----
            results_details = details_model(plate_crop, conf=0.1)

            for dr in results_details:
                for dbox in dr.boxes:
                    dx1, dy1, dx2, dy2 = map(int, dbox.xyxy[0].cpu().numpy())
                    cls_id = int(dbox.cls[0].cpu().numpy())
                    label = dr.names[cls_id]  # "text", "city", "number"

                    detail_crop = plate_crop[dy1:dy2, dx1:dx2]

                    if detail_crop.size > 0:
                        ocr_results = reader.readtext(detail_crop)
                        text_detected = " ".join([res[1] for res in ocr_results])

                        if label == "text":
                            arabic_text = clean_arabic_text(text_detected)
                        elif label == "number":
                            # Extract only the plate number (not city)
                            arabic_num, eng_num, final_eng = extract_plate_number(text_detected)
                            arabic_number = arabic_num
                            english_number = final_eng
                        elif label == "city":
                            # Extract only the city number
                            city_arabic, city_english = extract_city_number(text_detected)

                        cv2.rectangle(plate_crop, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
                        cv2.putText(plate_crop, f"{label}: {text_detected}",
                                    (dx1, dy1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 0), 1)

            # ---- Fallback: Full plate OCR ----
            full_ocr_results = reader.readtext(plate_crop)
            full_plate_text = " ".join([res[1] for res in full_ocr_results])

            # If we didn't detect components separately, try to extract from full text
            if not arabic_number and not english_number:
                # First extract city number from beginning
                city_arabic, city_english = extract_city_number(full_plate_text)
                
                # Then extract the remaining part as plate number
                # Remove city number from the beginning
                remaining_text = full_plate_text
                if city_arabic:
                    remaining_text = remaining_text.replace(city_arabic, '', 1)
                if city_english and city_english != city_arabic:
                    remaining_text = remaining_text.replace(city_english, '', 1)
                    
                arabic_num, eng_num, final_eng = extract_plate_number(remaining_text)
                arabic_number = arabic_num
                english_number = final_eng

            if not city_arabic and not city_english:
                city_arabic, city_english = extract_city_number(full_plate_text)

            annotated[y1:y2, x1:x2] = plate_crop

    return annotated, plate_detected, city_arabic, city_english, arabic_text, arabic_number, english_number, full_plate_text

# ===== Streamlit UI =====
st.set_page_config(page_title="🚔 Yemeni Traffic Plate Recognition", layout="wide")

st.title("🚔 Yemeni Traffic Plate Recognition System")

# Add mode selection
mode = st.radio("Select Input Mode:", ("Upload Image", "Live Stream"))

if mode == "Upload Image":
    # Add instructions
    with st.expander("ℹ️ Instructions"):
        st.markdown("""
        1. Upload an image of a vehicle with a Yemeni license plate
        2. The system will automatically detect and extract:
           - City code (1-2 digits)
           - Vehicle type (in Arabic)
           - Plate number (in Arabic and English)
        3. Results will be displayed on the right side
        """)

    uploaded_file = st.file_uploader("📷 Upload Vehicle Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Validate the uploaded file
        try:
            # Load image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            
            if img is None:
                st.error("Invalid image file. Please upload a valid image.")
                st.stop()
                
            # Show processing spinner
            with st.spinner("Processing image..."):
                annotated, plate_detected, city_arabic, city_english, arabic_text, arabic_number, english_number, full_plate_text = process_frame(
                    img, plate_model, details_model, reader
                )

            # ---- Display results ----
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            col1, col2 = st.columns([2, 1])

            with col1:
                st.image(annotated_rgb, caption="Detected Plate", use_column_width=True)

            with col2:
                st.subheader("📋 Detected Classes")
                st.text_input("City Number (Arabic)", value=city_arabic if city_arabic else "Not detected", key="city_arabic")
                st.text_input("City Number (English)", value=city_english if city_english else "Not detected", key="city_english")
                st.text_input("Vehicle Type", value=arabic_text if arabic_text else "Not detected", key="type")
                st.text_input("Arabic Number", value=arabic_number if arabic_number else "Not detected", key="arabic_num")
                st.text_input("English Number", value=english_number if english_number else "Not detected", key="english_num")

                st.subheader("🔍 Full Plate OCR (Fallback)")
                st.text_area("OCR Output", value=full_plate_text if full_plate_text else "No text detected", height=100, key="ocr_output")

                st.subheader("📝 Traffic Summary")
                if plate_detected:
                    summary_text = f"""✅ Plate Detected
    City (Arabic): {city_arabic if city_arabic else 'Unknown'}
    City (English): {city_english if city_english else 'Unknown'}
    Type: {arabic_text if arabic_text else 'Unknown'}
    Arabic: {arabic_number if arabic_number else 'Unknown'}
    English: {english_number if english_number else 'Unknown'}"""
                else:
                    summary_text = "❌ No plate detected in image"

                st.text_area("Summary", value=summary_text, height=120, key="summary")
                
                # Add export functionality
                if plate_detected:
                    if st.button("📥 Export Results"):
                        # Create a simple text report
                        report = f"""Yemeni Traffic Plate Recognition Results
                        
    Image: {uploaded_file.name}
    Processing Time: {time.strftime("%Y-%m-%d %H:%M:%S")}
                        
    Results:
    - City Number (Arabic): {city_arabic if city_arabic else 'Not detected'}
    - City Number (English): {city_english if city_english else 'Not detected'}
    - Vehicle Type: {arabic_text if arabic_text else 'Not detected'}
    - Arabic Number: {arabic_number if arabic_number else 'Not detected'}
    - English Number: {english_number if english_number else 'Not detected'}
                        
    Full OCR Text: {full_plate_text if full_plate_text else 'No text detected'}
    """
                        st.download_button(
                            label="Download Report",
                            data=report,
                            file_name=f"plate_results_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

else:  # Live Stream mode
    st.subheader("🌐 Live Stream Detection")
    
    with st.expander("ℹ️ Instructions"):
        st.markdown("""
        1. Enter the URL of a live stream (RTSP, HTTP, etc.)
        2. The system will process frames in real-time
        3. Detected plates will be displayed with results
        4. Click 'Stop Stream' to end the detection
        """)
    
    stream_url = st.text_input("Enter Stream URL:", placeholder="e.g., rtsp://username:password@ip:port/stream")
    
    if stream_url:
        # Initialize video capture
        cap = cv2.VideoCapture(stream_url)
        
        if not cap.isOpened():
            st.error("Could not open stream. Please check the URL and try again.")
        else:
            st.success("Stream connected successfully!")
            
            # Create placeholders for the video and results
            frame_placeholder = st.empty()
            results_placeholder = st.empty()
            
            stop_button = st.button("Stop Stream")
            
            # Process frames from the stream
            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("Failed to read frame from stream.")
                    break
                
                # Process the frame
                annotated, plate_detected, city_arabic, city_english, arabic_text, arabic_number, english_number, full_plate_text = process_frame(
                    frame, plate_model, details_model, reader
                )
                
                # Convert to RGB for display
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                
                # Display the frame
                frame_placeholder.image(annotated_rgb, caption="Live Stream Detection", use_column_width=True)
                
                # Display results if a plate was detected
                if plate_detected:
                    with results_placeholder.container():
                        st.subheader("📋 Latest Detection")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.text_input("City Number (Arabic)", value=city_arabic, key="live_city_arabic")
                            st.text_input("City Number (English)", value=city_english, key="live_city_english")
                            st.text_input("Vehicle Type", value=arabic_text, key="live_type")
                            
                        with col2:
                            st.text_input("Arabic Number", value=arabic_number, key="live_arabic_num")
                            st.text_input("English Number", value=english_number, key="live_english_num")
                
                # Add a small delay to prevent overwhelming the system
                time.sleep(0.1)
            
            # Release the capture when done
            cap.release()
            st.info("Stream stopped.")