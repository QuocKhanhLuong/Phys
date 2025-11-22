"""
Tiền xử lý ATLAS: Chuyển đổi BIDS NIfTI -> .npy để tải nhanh bằng memmap.
Chạy file này MỘT LẦN trước khi train.

Usage:
    python scripts/preprocess_atlas.py
"""

import os
import sys
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import json
import cv2
from skimage.transform import resize

# Thêm thư mục gốc vào path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from src import config

def preprocess_single_patient_atlas(image_path: Path, mask_path: Path, target_size=(224, 224)):
    """
    Tải, resize 1 cặp T1w/mask của ATLAS.
    Trả về: (volume, mask, volume_id)
    """
    volume_id = image_path.parent.parent.name # Lấy tên (ví dụ: sub-001)
    
    try:
        # Tải NIfTI
        img_data = nib.nifti1.load(str(image_path)).get_fdata()
        mask_data = nib.nifti1.load(str(mask_path)).get_fdata()
        
        num_slices = img_data.shape[2]
        
        # Resize các lát cắt
        resized_img = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.float32)
        resized_mask = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.uint8)
        
        for i in range(num_slices):
            resized_img[:, :, i] = cv2.resize(
                img_data[:, :, i].astype(np.float32),
                target_size,
                interpolation=cv2.INTER_LINEAR
            )
            resized_mask[:, :, i] = cv2.resize(
                mask_data[:, :, i].astype(np.uint8),
                target_size,
                interpolation=cv2.INTER_NEAREST
            )
        
        # Chuẩn hóa volume
        max_val = resized_img.max()
        if max_val > 0:
            resized_img /= max_val
        
        # ATLAS chỉ có 2 lớp (0 và 1)
        resized_mask[resized_mask > 1] = 1 
        
        return resized_img, resized_mask, volume_id
        
    except Exception as e:
        print(f"  Lỗi xử lý {volume_id}: {e}")
        return None, None, None

def process_split(bids_split_dir: Path, output_split_dir: Path, target_size=(224, 224), skip_existing=True):
    """Xử lý một split (train hoặc test) của ATLAS BIDS."""
    
    volumes_dir = output_split_dir / 'volumes'
    masks_dir = output_split_dir / 'masks'
    os.makedirs(volumes_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    derivatives_dir = bids_split_dir / "derivatives" / "ATLAS"
    if not derivatives_dir.exists():
        print(f"LỖI: Không tìm thấy {derivatives_dir}")
        return {}

    subject_dirs = sorted([d for d in derivatives_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    
    print(f"Tìm thấy {len(subject_dirs)} bệnh nhân trong {bids_split_dir}")
    
    volume_info = {}
    processed_count = 0
    skipped_count = 0
    
    for sub_dir in tqdm(subject_dirs, desc=f"Processing {bids_split_dir.name}"):
        patient_id = sub_dir.name
        # ATLAS có thêm ses-1 trong cấu trúc
        image_path = sub_dir / "ses-1" / "anat" / f"{patient_id}_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz"
        mask_path = sub_dir / "ses-1" / "anat" / f"{patient_id}_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz"
        
        volume_save_path = volumes_dir / f'{patient_id}.npy'
        mask_save_path = masks_dir / f'{patient_id}.npy'

        if skip_existing and volume_save_path.exists() and mask_save_path.exists():
            try:
                test_mask = np.load(mask_save_path)
                volume_info[patient_id] = {'num_slices': int(test_mask.shape[2])}
                skipped_count += 1
                continue
            except:
                pass # Xử lý lại file lỗi

        if not image_path.exists() or not mask_path.exists():
            print(f"Bỏ qua {patient_id}: Thiếu file NIfTI")
            continue
            
        volume, mask, volume_id = preprocess_single_patient_atlas(image_path, mask_path, target_size)
        
        if volume is None:
            continue
            
        try:
            np.save(volume_save_path, volume)
            np.save(mask_save_path, mask)
            volume_info[patient_id] = {'num_slices': int(mask.shape[2])}
            processed_count += 1
        except Exception as e:
            print(f"  Lỗi khi lưu {volume_id}: {e}")

    print(f"Split {bids_split_dir.name}: Xử lý mới {processed_count}, Bỏ qua {skipped_count}")
    return volume_info

def main():
    bids_root = project_root / "data" / "ATLAS_BIDS"
    npy_root = config.ATLAS_PREPROCESSED_DIR
    target_size = (config.IMG_SIZE, config.IMG_SIZE)
    
    print("="*60)
    print("TIỀN XỬ LÝ ATLAS (BIDS -> NPY)")
    print(f"Nguồn BIDS: {bids_root}")
    print(f"Đích NPY: {npy_root}")
    print("="*60)

    # Xử lý tập Train
    train_info = process_split(
        bids_root / "train",
        npy_root / "train",
        target_size
    )
    
    # Xử lý tập Test
    test_info = process_split(
        bids_root / "test",
        npy_root / "test",
        target_size
    )
    
    # Lưu metadata (chủ yếu cho tập train)
    metadata = {
        'dataset': 'ATLAS',
        'target_size': list(target_size),
        'volume_info': train_info,
        'test_volume_info': test_info,
        'num_classes': 2,
        'class_names': ['Background', 'Lesion']
    }
    
    (npy_root / "train").mkdir(parents=True, exist_ok=True)
    (npy_root / "test").mkdir(parents=True, exist_ok=True)
    
    with open(npy_root / "train" / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
        
    # Copy metadata cho test
    metadata['volume_info'] = test_info
    with open(npy_root / "test" / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print("\nHoàn tất tiền xử lý ATLAS!")

if __name__ == '__main__':
    main()