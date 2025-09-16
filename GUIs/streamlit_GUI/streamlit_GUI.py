import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from PIL import Image
import os
from pathlib import Path

# ===== USER SETTINGS =====
script_dir = Path(__file__).parent
parent_path = script_dir.parent.parent #/"src"/"models"/"yolov8s14"/"weights"/"best.pt"
MODEL_PLATE =parent_path / "src" / "models" / "yolov8s14" / "weights" / "best.pt"# plate detector
MODEL_PLATE=str(MODEL_PLATE.resolve())
MODEL_PLATE_DETAILS = parent_path / "src" / "models" / "yolov8s5" / "weights" / "best.pt"   #"src/models/yolov8s5/weights/best.pt" # details detector
MODEL_PLATE_DETAILS=str(MODEL_PLATE_DETAILS.resolve())
# Load models once
plate_model = YOLO(MODEL_PLATE)
details_model = YOLO(MODEL_PLATE_DETAILS)

# OCR reader
reader = easyocr.Reader(['en', 'ar'])

# ===== Helper functions =====
def arabic_to_english_digits(text: str) -> str:
    mapping = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    return text.translate(mapping)

def clean_arabic_text(text: str) -> str:
    if not text:
        return text
    text = text.replace('?', '').replace('؟', '')
    text = text.replace('be', '').replace('4ق', 'ق').replace('nب', 'ب').replace('itق', 'ق')
    import re
    text = re.sub(r'[a-zA-Z]', '', text)
    if "خصوصي" in text or "نقل" in text:
        text = "خصوصي نقل اجرة"
    return text.strip()

def extract_clean_numbers(text: str):
    if not text:
        return "", "", ""
    arabic_digits = ''.join([c for c in text if c in "٠١٢٣٤٥٦٧٨٩"])
    english_digits = ''.join([c for c in text if c.isdigit()])
    arabic_to_eng = arabic_to_english_digits(arabic_digits) if arabic_digits else ""
    return arabic_digits, english_digits, arabic_to_eng

# ===== Streamlit UI =====
st.set_page_config(page_title="🚔 Yemeni Traffic Plate Recognition", layout="wide")

st.title("🚔 Yemeni Traffic Plate Recognition System")

uploaded_file = st.file_uploader("📷 Upload Vehicle Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    annotated = img.copy()

    # Initialize results
    city_val, arabic_text, arabic_number, english_number, full_plate_text = "", "", "", "", ""

    # ---- Stage 1: Detect Plate ----
    results_plate = plate_model(img)
    plate_detected = False

    for r in results_plate:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            plate_crop = img[y1:y2, x1:x2]
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
                            arabic_num, eng_num, arabic_to_eng = extract_clean_numbers(text_detected)
                            arabic_number = arabic_num
                            english_number = eng_num if eng_num else arabic_to_eng
                        elif label == "city":
                            city_arabic, city_eng, city_arabic_to_eng = extract_clean_numbers(text_detected)
                            city_val = city_arabic if city_arabic else (city_eng if city_eng else text_detected)

                        cv2.rectangle(plate_crop, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
                        cv2.putText(plate_crop, f"{label}: {text_detected}",
                                    (dx1, dy1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 0), 1)

            # ---- Fallback: Full plate OCR ----
            full_ocr_results = reader.readtext(plate_crop)
            full_plate_text = " ".join([res[1] for res in full_ocr_results])

            if not arabic_number and not english_number:
                arabic_num, eng_num, arabic_to_eng = extract_clean_numbers(full_plate_text)
                arabic_number = arabic_num
                english_number = eng_num if eng_num else arabic_to_eng

            if not city_val:
                city_arabic, city_eng, city_arabic_to_eng = extract_clean_numbers(full_plate_text)
                city_val = city_arabic if city_arabic else (city_eng if city_eng else "Not detected")

            annotated[y1:y2, x1:x2] = plate_crop

    # ---- Display results ----
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(annotated_rgb, caption="Detected Plate", use_column_width=True)

    with col2:
        st.subheader("📋 Detected Classes")
        st.text_input("City Number", value=city_val if city_val else "Not detected")
        st.text_input("Vehicle Type", value=arabic_text if arabic_text else "Not detected")
        st.text_input("Arabic Number", value=arabic_number if arabic_number else "Not detected")
        st.text_input("English Number", value=english_number if english_number else "Not detected")

        st.subheader("🔍 Full Plate OCR (Fallback)")
        st.text_area("OCR Output", value=full_plate_text if full_plate_text else "No text detected", height=100)

        st.subheader("📝 Traffic Summary")
        if plate_detected:
            summary_text = f"""✅ Plate Detected
City: {city_val if city_val else 'Unknown'}
Type: {arabic_text if arabic_text else 'Unknown'}
Arabic: {arabic_number if arabic_number else 'Unknown'}
English: {english_number if english_number else 'Unknown'}"""
        else:
            summary_text = "❌ No plate detected in image"

        st.text_area("Summary", value=summary_text, height=120)
