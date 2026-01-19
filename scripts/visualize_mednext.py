"""
MedNeXt Visualization Script (using nnUNet inference)

Uses nnUNet's predict_simple for correct preprocessing.
Generates segmentation predictions for specific patients.

Usage:
    python scripts/visualize_mednext.py
"""

import os
import sys
import shutil
import subprocess
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from datetime import datetime
from glob import glob

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_CLASSES = 4

# Patients to visualize
PATIENT_FILTER = ['patient104']

# MedNeXt work directory
MEDNEXT_BASE = PROJECT_ROOT / 'comparison' / 'MedNeXt'
WORK_DIR = MEDNEXT_BASE / 'MedNeXt_work_dir'
NNUNET_RAW = WORK_DIR / 'nnUNet_raw_data_base' / 'nnUNet_raw_data'

OUTPUT_DIR = PROJECT_ROOT / 'visualization_outputs' / 'mednext'


# ============================================================================
# INFERENCE USING NNUNET
# ============================================================================

def run_mednext_inference(input_dir, output_dir, task='001', fold=0):
    """Run MedNeXt inference using nnUNet predict_simple."""
    
    # Setup environment
    env = os.environ.copy()
    env['nnUNet_raw_data_base'] = str(WORK_DIR / 'nnUNet_raw_data_base')
    env['nnUNet_preprocessed'] = str(WORK_DIR / 'nnUNet_preprocessed')
    env['RESULTS_FOLDER'] = str(WORK_DIR / 'nnUNet_results')
    env['PYTHONPATH'] = f"{str(MEDNEXT_BASE)}:{env.get('PYTHONPATH', '')}"
    
    cmd = [
        sys.executable, "-m", "nnunet_mednext.inference.predict_simple",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-t", task,
        "-tr", "nnUNetTrainerV2_MedNeXt_S_kernel3",
        "-m", "3d_fullres",
        "-p", "nnUNetPlansv2.1_trgSp_1x1x1",
        "-f", str(fold),
        "-chk", "model_best",
        "--disable_tta"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)


# ============================================================================
# VISUALIZATION
# ============================================================================

def save_slice_figure(image_slice, pred_slice, output_path):
    """Save prediction overlay figure (no title)."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    colors_list = ['black', 'red', 'green', 'blue']
    cmap = mcolors.ListedColormap(colors_list)
    
    ax.imshow(image_slice, cmap='gray')
    pred_masked = np.ma.masked_where(pred_slice == 0, pred_slice)
    ax.imshow(pred_masked, cmap=cmap, alpha=0.6, vmin=0, vmax=NUM_CLASSES-1)
    ax.axis('off')
    
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("MEDNEXT VISUALIZATION (via nnUNet inference)")
    print("=" * 60)
    print(f"Patients: {PATIENT_FILTER}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup temp directories
    temp_input = run_dir / 'temp_input'
    temp_output = run_dir / 'temp_output'
    temp_input.mkdir(exist_ok=True)
    temp_output.mkdir(exist_ok=True)
    
    # Find test images in nnUNet format
    images_ts = NNUNET_RAW / 'Task001_ACDC' / 'imagesTs'
    
    if not images_ts.exists():
        print(f"ERROR: Test images not found at {images_ts}")
        print("Please prepare test data using MedNeXt preprocessing scripts.")
        return
    
    # Copy filtered patient files to temp input
    for f in sorted(images_ts.glob("*_0000.nii.gz")):
        case_id = f.name.replace("_0000.nii.gz", "")
        patient_id = case_id.split('_')[0]
        if patient_id in PATIENT_FILTER:
            shutil.copy(f, temp_input / f.name)
            print(f"  Copied {case_id}")
    
    if not list(temp_input.glob("*.nii.gz")):
        print("No cases found for the specified patients!")
        return
    
    # Run inference
    print("\nRunning MedNeXt inference...")
    run_mednext_inference(temp_input, temp_output)
    
    # Process results
    print("\nGenerating visualizations...")
    
    for pred_file in sorted(temp_output.glob("*.nii.gz")):
        case_id = pred_file.name.replace(".nii.gz", "")
        patient_id = case_id.split('_')[0]
        
        print(f"\nProcessing {case_id}...")
        
        # Load prediction
        pred_nii = nib.load(pred_file)
        pred_data = pred_nii.get_fdata().astype(np.int32)
        
        # Load original image
        img_file = temp_input / f"{case_id}_0000.nii.gz"
        img_nii = nib.load(img_file)
        img_data = img_nii.get_fdata()
        
        # Create patient/model folder
        patient_dir = run_dir / patient_id / "MedNeXt"
        patient_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each slice
        num_slices = img_data.shape[2]
        for s in range(num_slices):
            image_slice = img_data[:, :, s]
            pred_slice = pred_data[:, :, s]
            
            output_path = patient_dir / f"{case_id}_slice{s:03d}.png"
            save_slice_figure(image_slice, pred_slice, output_path)
        
        print(f"  Saved {num_slices} slices to {patient_dir}")
    
    # Cleanup temp
    shutil.rmtree(temp_input)
    # Keep temp_output for debugging if needed
    
    print("\n" + "=" * 60)
    print(f"VISUALIZATION COMPLETE!")
    print(f"Output directory: {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
