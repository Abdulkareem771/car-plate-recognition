#!/usr/bin/env python3
"""
Convert CSV annotations to YOLO format
"""

import pandas as pd
import os
from pathlib import Path

def convert_csv_to_yolo(csv_path, output_dir, class_name="License_Plate"):
    """
    Convert CSV annotations to YOLO format
    
    Args:
        csv_path: Path to the CSV annotation file
        output_dir: Directory to save the YOLO format .txt files
        class_name: Name of the class (will be converted to class_id 0)
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by filename to handle multiple annotations per image
    grouped = df.groupby('filename')
    
    converted_count = 0
    
    for filename, group in grouped:
        # Create output filename (replace .jpg with .txt)
        txt_filename = filename.replace('.jpg', '.txt')
        txt_path = os.path.join(output_dir, txt_filename)
        
        # Open the output file
        with open(txt_path, 'w') as f:
            for _, row in group.iterrows():
                # Get image dimensions
                img_width = row['width']
                img_height = row['height']
                
                # Get bounding box coordinates
                xmin = row['xmin']
                ymin = row['ymin']
                xmax = row['xmax']
                ymax = row['ymax']
                
                # Convert to YOLO format (normalized center coordinates and dimensions)
                x_center = (xmin + xmax) / 2.0 / img_width
                y_center = (ymin + ymax) / 2.0 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                # Class ID (0 for License_Plate)
                class_id = 0
                
                # Write to file in YOLO format: class_id x_center y_center width height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        converted_count += 1
        if converted_count % 100 == 0:
            print(f"Converted {converted_count} files...")
    
    print(f"Conversion complete! Converted {converted_count} files to {output_dir}")

def main():
    """Convert all CSV files to YOLO format"""
    
    # Define paths
    base_dir = "archive"
    datasets = ["train/train", "valid/valid", "test/test"]
    
    for dataset in datasets:
        csv_path = os.path.join(base_dir, dataset, "_annotations.csv")
        output_dir = os.path.join(base_dir, dataset)
        
        if os.path.exists(csv_path):
            print(f"\nConverting {dataset}...")
            convert_csv_to_yolo(csv_path, output_dir)
        else:
            print(f"CSV file not found: {csv_path}")
    
    print("\nAll conversions completed!")

if __name__ == "__main__":
    main()
