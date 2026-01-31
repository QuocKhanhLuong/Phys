"""
Create a comparison table image with patients as rows and 
InputMRI, GroundTruth, PGE-UNet as columns.
"""
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIS_OUTPUT_DIR = os.path.join(BASE_DIR, "visualization_outputs", "model_comparison", "Output")
VIS_OUTPUT_DIR_LOWER = os.path.join(BASE_DIR, "visualization_outputs", "model_comparison", "output")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Patients to include
PATIENTS = ["patient103", "patient104", "patient105", "patient112", "patient113", "patient114"]

# Columns to display
COLUMNS = ["InputMRI", "GroundTruth", "PIE-UNet"]

# Slice to use for each patient (ED slice004 as representative)
SLICE_NAME = "ED_slice004"

def get_image_path(patient, column):
    """Get the image path for a specific patient and column."""
    # patient103, 104, 105 are in Output folder
    # patient112, 113, 114 are in output folder
    if patient in ["patient103", "patient104", "patient105"]:
        base_path = VIS_OUTPUT_DIR
    else:
        base_path = VIS_OUTPUT_DIR_LOWER
    
    img_path = os.path.join(base_path, patient, column, f"{patient}_{SLICE_NAME}.png")
    return img_path

def create_comparison_table():
    """Create a grid comparison image."""
    # First, get dimensions from one image
    sample_path = get_image_path(PATIENTS[0], COLUMNS[0])
    sample_img = Image.open(sample_path)
    img_width, img_height = sample_img.size
    
    # Table settings
    header_height = 50
    row_label_width = 120
    padding = 5
    
    # Calculate total dimensions
    total_width = row_label_width + len(COLUMNS) * (img_width + padding) + padding
    total_height = header_height + len(PATIENTS) * (img_height + padding) + padding
    
    # Create canvas (white background)
    canvas = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    # Draw column headers
    for col_idx, col_name in enumerate(COLUMNS):
        x = row_label_width + padding + col_idx * (img_width + padding) + img_width // 2
        y = header_height // 2
        
        # Rename for display
        display_name = col_name
        if col_name == "InputMRI":
            display_name = "Input MRI"
        elif col_name == "GroundTruth":
            display_name = "Ground Truth"
        elif col_name == "PIE-UNet":
            display_name = "PGE-UNet (Ours)"
        
        # Center text
        bbox = draw.textbbox((0, 0), display_name, font=font)
        text_width = bbox[2] - bbox[0]
        draw.text((x - text_width // 2, y - 8), display_name, fill=(0, 0, 0), font=font)
    
    # Draw row labels and images
    for row_idx, patient in enumerate(PATIENTS):
        y_pos = header_height + padding + row_idx * (img_height + padding)
        
        # Draw row label
        label = patient.replace("patient", "Patient ")
        bbox = draw.textbbox((0, 0), label, font=small_font)
        text_height = bbox[3] - bbox[1]
        draw.text((10, y_pos + img_height // 2 - text_height // 2), label, fill=(0, 0, 0), font=small_font)
        
        # Paste images
        for col_idx, col_name in enumerate(COLUMNS):
            x_pos = row_label_width + padding + col_idx * (img_width + padding)
            
            img_path = get_image_path(patient, col_name)
            if os.path.exists(img_path):
                img = Image.open(img_path)
                canvas.paste(img, (x_pos, y_pos))
            else:
                # Draw placeholder
                draw.rectangle([x_pos, y_pos, x_pos + img_width, y_pos + img_height], 
                              outline=(200, 200, 200), width=2)
                draw.text((x_pos + 10, y_pos + img_height // 2), "Not found", fill=(150, 150, 150))
                print(f"Warning: Image not found: {img_path}")
    
    # Save the result
    os.makedirs(ASSETS_DIR, exist_ok=True)
    output_path = os.path.join(ASSETS_DIR, "patient_comparison_table.png")
    canvas.save(output_path, quality=95)
    print(f"Saved comparison table to: {output_path}")
    print(f"Dimensions: {total_width} x {total_height}")
    
    return output_path

if __name__ == "__main__":
    create_comparison_table()
