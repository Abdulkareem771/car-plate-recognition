import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
import cv2
from PIL import Image, ImageTk
import os
from pathlib import Path

# Function to find the model file automatically
def find_model_file():
    """
    Try to find the model file in common locations
    Returns the path to the model file or None if not found
    """
    # Get the current script directory
    script_dir = Path(__file__).parent
    
    # Possible model paths to check (relative to the script location)
    model_path = script_dir.parent.parent / "src" / "models" / "yolov8s14" / "weights" / "best.pt"
    return str(model_path.resolve())

# Find the model file
MODEL_PATH = find_model_file()


# Load YOLO model once
try:
    model = YOLO(MODEL_PATH)
    print(f"Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    exit()

# Create main window
root = tk.Tk()
root.title("YOLO Car Plate Detector")
root.geometry("900x700")

# Add a title label
title_label = tk.Label(root, text="Car Plate Detection", font=("Arial", 16, "bold"))
title_label.pack(pady=10)

# Label to show image
panel = tk.Label(root)
panel.pack(pady=10)

# Add a status label
status_label = tk.Label(root, text="Ready to detect", font=("Arial", 10))
status_label.pack(pady=5)

def upload_and_detect():
    # Choose image file
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        return
    
    # Update status
    status_label.config(text="Processing image...")
    root.update()
    
    try:
        # Run YOLO detection
        results = model(file_path)
        
        # Take first result and draw boxes
        for r in results:
            annotated = r.plot()  # numpy array (BGR)
        
        # Convert BGR (cv2) to RGB (PIL)
        img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Resize for display if too big (maintain aspect ratio)
        img_pil.thumbnail((800, 600))
        
        # Convert to ImageTk
        imgtk = ImageTk.PhotoImage(img_pil)
        
        # Show in panel
        panel.config(image=imgtk)
        panel.image = imgtk  # keep reference
        
        # Update status
        num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
        status_label.config(text=f"Detection complete: Found {num_detections} license plate(s)")
        
    except Exception as e:
        status_label.config(text="Error during detection")
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Button frame
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# Button
btn = tk.Button(button_frame, text="Upload Image", command=upload_and_detect, 
                font=("Arial", 12), bg="#4CAF50", fg="white", padx=20, pady=10)
btn.pack(side=tk.LEFT, padx=10)

# Add a quit button
quit_btn = tk.Button(button_frame, text="Quit", command=root.quit,
                    font=("Arial", 12), bg="#f44336", fg="white", padx=20, pady=10)
quit_btn.pack(side=tk.LEFT, padx=10)

# Add model path info
model_info = tk.Label(root, text=f"Model: {os.path.basename(MODEL_PATH)}", 
                     font=("Arial", 9), fg="gray")
model_info.pack(pady=5)

root.mainloop()