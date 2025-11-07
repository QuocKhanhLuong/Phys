import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config # Import config để lấy đường dẫn

def visualize_raw_patient(patient_id: str, brats_raw_dir: str):
    """
    Loads and visualizes a single slice from the raw BraTS dataset for a given patient.
    """
    patient_path = os.path.join(brats_raw_dir, patient_id)
    print(f"Loading data for patient: {patient_id} from {patient_path}")

    if not os.path.isdir(patient_path):
        print(f"Error: Patient directory not found: {patient_path}")
        return

    paths = {
        't1': os.path.join(patient_path, f"{patient_id}_t1.nii.gz"),
        't1ce': os.path.join(patient_path, f"{patient_id}_t1ce.nii.gz"),
        't2': os.path.join(patient_path, f"{patient_id}_t2.nii.gz"),
        'flair': os.path.join(patient_path, f"{patient_id}_flair.nii.gz"),
        'seg': os.path.join(patient_path, f"{patient_id}_seg.nii.gz"),
    }

    if any(not os.path.exists(p) for p in paths.values()):
        print("Error: One or more NIfTI files are missing for this patient.")
        return

    try:
        # Load all modalities and segmentation
        modalities_data = {name: nib.nifti1.load(path).get_fdata() for name, path in paths.items()}
        seg_data = modalities_data.pop('seg')
        
        # Find the slice with the largest tumor area to visualize
        if seg_data.max() > 0:
            slice_idx = np.unravel_index(np.argmax(np.sum(seg_data > 0, axis=(0, 1))), seg_data.shape[2:])[-1]
        else:
            # If no tumor, just take the middle slice
            slice_idx = seg_data.shape[2] // 2
            
        print(f"Visualizing slice: {slice_idx}")

        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Raw Data: Patient {patient_id} - Slice {slice_idx}', fontsize=16)
        
        modality_names = ['t1', 't1ce', 't2', 'flair']
        for i, mod_name in enumerate(modality_names):
            img_slice = modalities_data[mod_name][:, :, slice_idx]
            
            # Plot raw modality
            axes[0, i].imshow(np.rot90(img_slice), cmap='gray')
            axes[0, i].set_title(f'Raw {mod_name.upper()}')
            axes[0, i].axis('off')
            
            # Plot with segmentation overlay
            axes[1, i].imshow(np.rot90(img_slice), cmap='gray')
            seg_slice = seg_data[:, :, slice_idx]
            if seg_slice.max() > 0:
                masked_seg = np.ma.masked_where(seg_slice == 0, seg_slice)
                axes[1, i].imshow(np.rot90(masked_seg), cmap='jet', alpha=0.5)
            axes[1, i].set_title(f'{mod_name.upper()} + Seg Mask')
            axes[1, i].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        # ==================================
        # THÊM DÒNG NÀY ĐỂ LƯU FILE
        # ==================================
        output_filename = f"{patient_id}_slice_{slice_idx}_raw_visualization.png"
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_filename}")
        # ==================================

        # Bạn có thể giữ lại plt.show() nếu vẫn muốn thử hiển thị,
        # hoặc xóa/comment nó đi nếu chỉ cần lưu file.
        # plt.show()
        plt.close(fig) # Đóng figure sau khi lưu để giải phóng bộ nhớ

    except Exception as e:
        print(f"An error occurred while processing patient {patient_id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize raw BraTS data for a single patient.')
    parser.add_argument(
        'patient_id', 
        type=str, 
        help='The ID of the patient to visualize (e.g., BraTS2021_00000)'
    )
    args = parser.parse_args()
    
    # Use the raw data directory from your config file
    if not os.path.exists(config.BRATS_RAW_DIR):
        print(f"Error: Raw BraTS directory specified in config.py not found: {config.BRATS_RAW_DIR}")
        sys.exit(1)
        
    visualize_raw_patient(args.patient_id, config.BRATS_RAW_DIR)
