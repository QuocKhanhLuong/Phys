"""
Create a comparison grid image with:
- Columns = Patients (103, 104, 105, 112, 113, 114)
- Rows = Image types (Input MRI, Ground Truth, PGE-UNet)
"""
import os
from PIL import Image, ImageDraw, ImageFont

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIS_OUTPUT_DIR = os.path.join(BASE_DIR, "visualization_outputs", "model_comparison", "Output")
VIS_OUTPUT_DIR_LOWER = os.path.join(BASE_DIR, "visualization_outputs", "model_comparison", "output")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Columns = Patients
PATIENTS = ["patient103", "patient104", "patient105", "patient112", "patient113", "patient114"]
PATIENT_LABELS = ["Patient 103", "Patient 104", "Patient 105", "Patient 112", "Patient 113", "Patient 114"]

# Rows = Image types
IMAGE_TYPES = ["InputMRI", "GroundTruth", "PIE-UNet"]
ROW_LABELS = ["Input MRI", "Ground Truth", "PGE-UNet (Ours)"]

# Slice to use
SLICE_NAME = "ED_slice004"

def get_image_path(patient, image_type):
    """Get the image path for a specific patient and image type."""
    if patient in ["patient103", "patient104", "patient105"]:
        base_path = VIS_OUTPUT_DIR
    else:
        base_path = VIS_OUTPUT_DIR_LOWER
    
    img_path = os.path.join(base_path, patient, image_type, f"{patient}_{SLICE_NAME}.png")
    return img_path

def create_comparison_grid():
    """Create a grid: Rows = image types, Columns = patients."""
    # Get dimensions from sample image
    sample_path = get_image_path(PATIENTS[0], IMAGE_TYPES[0])
    sample_img = Image.open(sample_path)
    img_width, img_height = sample_img.size
    
    # Grid settings - BALANCED FONTS
    header_height = 180     # Reduced from 250
    row_label_width = 500   # Reduced from 800
    padding = 20
    bg_color = (0, 0, 0)
    legend_height = 150     # Reduced from 200
    
    # Calculate total dimensions
    num_cols = len(PATIENTS)
    num_rows = len(IMAGE_TYPES)
    total_width = row_label_width + num_cols * img_width + (num_cols + 1) * padding
    total_height = header_height + num_rows * img_height + (num_rows + 1) * padding + legend_height
    
    # Create canvas
    canvas = Image.new('RGB', (total_width, total_height), color=bg_color)
    draw = ImageDraw.Draw(canvas)
    
    # Load fonts (BALANCED SIZES)
    try:
        header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 100)
        row_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 80)
        legend_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 70)
    except:
        print("Warning: Custom fonts not found, using default (might be small)")
        header_font = ImageFont.load_default()
        row_font = header_font
        legend_font = header_font
    
    # Draw column headers (Patient names)
    for col_idx, patient_label in enumerate(PATIENT_LABELS):
        x = row_label_width + padding + col_idx * (img_width + padding) + img_width // 2
        y = header_height // 2
        
        bbox = draw.textbbox((0, 0), patient_label, font=header_font)
        text_width = bbox[2] - bbox[0]
        draw.text((x - text_width // 2, y - 8), patient_label, fill=(255, 255, 255), font=header_font)
    
    # Draw row labels and images
    for row_idx, row_label in enumerate(ROW_LABELS):
        y_pos = header_height + padding + row_idx * (img_height + padding)
        
        # Draw row label (left side)
        bbox = draw.textbbox((0, 0), row_label, font=row_font)
        text_height = bbox[3] - bbox[1]
        draw.text((10, y_pos + img_height // 2 - text_height // 2), row_label, fill=(255, 255, 255), font=row_font)
        
        # Paste images for each patient
        for col_idx, patient in enumerate(PATIENTS):
            x_pos = row_label_width + padding + col_idx * (img_width + padding)
            
            img_path = get_image_path(patient, IMAGE_TYPES[row_idx])
            if os.path.exists(img_path):
                img = Image.open(img_path)
                canvas.paste(img, (x_pos, y_pos))
            else:
                draw.rectangle([x_pos, y_pos, x_pos + img_width, y_pos + img_height], 
                              outline=(100, 100, 100), width=1)
                print(f"Warning: Not found: {img_path}")
    
    # Draw legend at bottom
    legend_y = total_height - legend_height + 40
    legend_items = [
        ("Right Ventricle (RV)", (255, 0, 0)),
        ("Myocardium (MYO)", (0, 255, 0)),
        ("Left Ventricle (LV)", (0, 0, 255)),
    ]
    
    # Calculate legend total width to center it properly
    legend_item_width = 900  # Detailed width adjustment
    legend_start_x = total_width // 2 - (len(legend_items) * legend_item_width) // 2
    
    for i, (label, color) in enumerate(legend_items):
        x = legend_start_x + i * legend_item_width
        # Draw larger color box (70x70)
        draw.rectangle([x, legend_y, x + 70, legend_y + 70], fill=color, outline=color)
        # Draw label next to box
        draw.text((x + 90, legend_y - 5), label, fill=(255, 255, 255), font=legend_font)
    
    # Save result
    os.makedirs(ASSETS_DIR, exist_ok=True)
    output_path = os.path.join(ASSETS_DIR, "patient_comparison_grid.png")
    canvas.save(output_path, quality=95)
    print(f"✅ Saved: {output_path}")
    print(f"   Size: {total_width} x {total_height}")
    print(f"   Layout: {num_rows} rows (image types) x {num_cols} columns (patients)")
    
    return output_path

if __name__ == "__main__":
    create_comparison_grid()
