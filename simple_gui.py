import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import cv2
from PIL import Image, ImageTk

# === USER SETTINGS ===
MODEL_PATH = "src/models/yolov8s14/weights/best.pt"

# Load YOLO model once
model = YOLO(MODEL_PATH)

# Create main window
root = tk.Tk()
root.title("YOLO Car Plate Detector")

# Label to show image
panel = tk.Label(root)
panel.pack()

def upload_and_detect():
    # Choose image file
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return
    
    # Run YOLO detection
    results = model(file_path)
    
    # Take first result and draw boxes
    for r in results:
        annotated = r.plot()  # numpy array (BGR)
    
    # Convert BGR (cv2) to RGB (PIL)
    img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Resize for display if too big
    img_pil.thumbnail((800, 600))
    
    # Convert to ImageTk
    imgtk = ImageTk.PhotoImage(img_pil)
    
    # Show in panel
    panel.config(image=imgtk)
    panel.image = imgtk  # keep reference

# Button
btn = tk.Button(root, text="Upload Image", command=upload_and_detect)
btn.pack()

root.mainloop()
