from ultralytics import YOLO
import cv2
import os

# Load your trained YOLO model
model = YOLO("runs/detect/yolov8n14/weights/best.pt")  # replace with your weights

# Input and output folders
input_folder = "cars"       # folder with your original pictures
output_folder = "yemen_plates"      # folder where cropped plates will be saved

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through images
for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)
    img = cv2.imread(img_path)

    # Run YOLO prediction
    results = model.predict(img, conf=0.5)

    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box coords
        plate_crop = img[y1:y2, x1:x2]

        # Save each plate as a separate image
        crop_name = f"{os.path.splitext(img_name)[0]}_plate{i}.jpg"
        cv2.imwrite(os.path.join(output_folder, crop_name), plate_crop)

print("✅ Plates cropped and saved in:", output_folder)
