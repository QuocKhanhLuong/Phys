"""
Create a comparison grid image with patients as rows and 
InputMRI, GroundTruth, PGE-UNet as columns.
Output: Single combined image for README
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
COLUMN_LABELS = ["Input Image", "Ground Truth", "PGE-UNet (Ours)"]

# Slice to use for each patient
SLICE_NAME = "ED_slice004"

def get_image_path(patient, column):
    """Get the image path for a specific patient and column."""
    if patient in ["patient103", "patient104", "patient105"]:
        base_path = VIS_OUTPUT_DIR
    else:
        base_path = VIS_OUTPUT_DIR_LOWER
    
    img_path = os.path.join(base_path, patient, column, f"{patient}_{SLICE_NAME}.png")
    return img_path

def create_comparison_grid():
    """Create a grid comparison image like the reference."""
    # Get dimensions from sample image
    sample_path = get_image_path(PATIENTS[0], COLUMNS[0])
    sample_img = Image.open(sample_path)
    img_width, img_height = sample_img.size
    
    # Grid settings
    header_height = 40
    padding = 4
    bg_color = (0, 0, 0)  # Black background like reference
    
    # Calculate total dimensions
    num_cols = len(COLUMNS)
    num_rows = len(PATIENTS)
    total_width = num_cols * img_width + (num_cols + 1) * padding
    total_height = header_height + num_rows * img_height + (num_rows + 1) * padding + 40  # +40 for legend
    
    # Create canvas (black background)
    canvas = Image.new('RGB', (total_width, total_height), color=bg_color)
    draw = ImageDraw.Draw(canvas)
    
    # Try to load fonts
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    # Draw column headers
    for col_idx, col_label in enumerate(COLUMN_LABELS):
        x = padding + col_idx * (img_width + padding) + img_width // 2
        y = header_height // 2
        
        # Center text
        bbox = draw.textbbox((0, 0), col_label, font=font)
        text_width = bbox[2] - bbox[0]
        draw.text((x - text_width // 2, y - 7), col_label, fill=(255, 255, 255), font=font)
    
    # Paste images
    for row_idx, patient in enumerate(PATIENTS):
        y_pos = header_height + padding + row_idx * (img_height + padding)
        
        for col_idx, col_name in enumerate(COLUMNS):
            x_pos = padding + col_idx * (img_width + padding)
            
            img_path = get_image_path(patient, col_name)
            if os.path.exists(img_path):
                img = Image.open(img_path)
                canvas.paste(img, (x_pos, y_pos))
            else:
                # Draw placeholder
                draw.rectangle([x_pos, y_pos, x_pos + img_width, y_pos + img_height], 
                              outline=(100, 100, 100), width=1)
                draw.text((x_pos + 10, y_pos + img_height // 2), "Not found", fill=(150, 150, 150))
                print(f"Warning: Image not found: {img_path}")
    
    # Draw legend at bottom
    legend_y = total_height - 30
    legend_items = [
        ("Right Ventricle (RV)", (255, 0, 0)),    # Red
        ("Myocardium (MYO)", (0, 255, 0)),        # Green
        ("Left Ventricle (LV)", (0, 0, 255)),    # Blue
    ]
    
    legend_start_x = total_width // 2 - 250
    for i, (label, color) in enumerate(legend_items):
        x = legend_start_x + i * 180
        # Draw color box
        draw.rectangle([x, legend_y, x + 15, legend_y + 15], fill=color, outline=color)
        # Draw label
        draw.text((x + 20, legend_y), label, fill=(255, 255, 255), font=small_font)
    
    # Save the result
    os.makedirs(ASSETS_DIR, exist_ok=True)
    output_path = os.path.join(ASSETS_DIR, "patient_comparison_grid.png")
    canvas.save(output_path, quality=95)
    print(f"✅ Saved comparison grid to: {output_path}")
    print(f"   Dimensions: {total_width} x {total_height}")
    
    return output_path

if __name__ == "__main__":
    create_comparison_grid()
