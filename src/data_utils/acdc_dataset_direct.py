"""
ACDC Dataset loading directly from NIfTI files (original workflow that achieved 93.96%).
Based on commit 205caef - with ePURE augmentation inside dataset.
"""

import os
import sys
import configparser
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from skimage.transform import resize
from typing import List, Optional, Tuple


class ACDCDataset25D(Dataset):
    """
    Dataset cho ACDC 2.5D, load trực tiếp từ NIfTI files (giống BraTS21Dataset25D từ commit 205caef).
    
    NOTE: ePURE augmentation nằm TRONG dataset.__getitem__.
    Để tránh CUDA multiprocessing error, PHẢI set num_workers=0 trong DataLoader!
    """
    def __init__(self, volumes_list, masks_list, num_input_slices=5, transforms=None, 
                 noise_injector_model=None, device: str = 'cpu'):
        """
        Args:
            volumes_list (list): List of ACDC volumes, shape (H, W, num_slices)
            masks_list (list): List of ACDC masks, shape (H, W, num_slices)  
            num_input_slices (int): Số slice cho 2.5D (phải là số lẻ)
            transforms (albumentations.Compose): Albumentations transforms
            noise_injector_model (nn.Module): ePURE model for noise injection
            device (str): Device for ePURE ('cuda' or 'cpu')
        """
        if num_input_slices % 2 == 0:
            raise ValueError("num_input_slices must be odd.")
            
        self.volumes = volumes_list
        self.masks = masks_list
        self.num_input_slices = num_input_slices
        self.transforms = transforms
        self.noise_injector_model = noise_injector_model
        self.device = device
        
        # Build index map for all valid slices
        self.index_map = []
        for vol_idx, vol in enumerate(self.volumes):
            radius = (self.num_input_slices - 1) // 2
            num_slices = vol.shape[2]  # ACDC: (H, W, Z)
            for slice_idx in range(radius, num_slices - radius):
                self.index_map.append((vol_idx, slice_idx))
    
    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        """
        Giống BraTS21Dataset25D từ commit 205caef.
        Lấy 2.5D slice stack và apply transforms + ePURE noise injection.
        """
        vol_idx, center_slice_idx = self.index_map[idx]
        
        current_volume = self.volumes[vol_idx]  # Shape: (H, W, Z)
        current_mask_volume = self.masks[vol_idx]  # Shape: (H, W, Z)
        num_slices_in_vol = current_volume.shape[2]
    
        radius = (self.num_input_slices - 1) // 2
        offsets = range(-radius, radius + 1)
        
        # Extract neighboring slices (with clamping at boundaries)
        slice_indices = [np.clip(center_slice_idx + offset, 0, num_slices_in_vol - 1) 
                        for offset in offsets]
        
        # Stack slices: (H, W, num_input_slices) 
        # ACDC volumes are already single-channel, so we just stack the slices
        image_stack = np.stack(
            [current_volume[:, :, i] for i in slice_indices],
            axis=-1
        ).astype(np.float32)
        
        mask = current_mask_volume[:, :, center_slice_idx].astype(np.int64)
        
        # Apply Albumentations transforms
        if self.transforms:
            augmented = self.transforms(image=image_stack, mask=mask)
            image_tensor = augmented['image']  # Should be tensor after ToTensorV2
            mask_tensor = augmented['mask']
        else:
            # If no transforms, convert to tensor manually
            image_tensor = torch.from_numpy(image_stack).permute(2, 0, 1)  # (C, H, W)
            mask_tensor = torch.from_numpy(mask)
    
        # Apply ePURE quantum noise injection (if provided)
        # NOTE: This requires GPU access in __getitem__, so num_workers MUST be 0!
        if self.noise_injector_model is not None:
            from src.utils.helpers import adaptive_quantum_noise_injection
            with torch.no_grad():
                img_on_gpu_with_batch = image_tensor.to(self.device).unsqueeze(0)
                noise_map = self.noise_injector_model(img_on_gpu_with_batch)
                image_tensor_with_noise_gpu = adaptive_quantum_noise_injection(
                    img_on_gpu_with_batch,
                    noise_map
                )
                image_tensor = image_tensor_with_noise_gpu.squeeze(0).cpu()
                
        return image_tensor, mask_tensor.long()


def load_acdc_volumes(directory, target_size=(224, 224), max_patients=None):
    """
    Nạp các volume MRI từ thư mục ACDC dataset (training hoặc testing).
    
    Args:
        directory (str): Đường dẫn đến thư mục chứa dữ liệu ACDC 
                        (ví dụ: data/ACDC/training hoặc data/ACDC/testing).
        target_size (tuple): Kích thước mục tiêu để resize các slice.
        max_patients (int, optional): Số lượng bệnh nhân tối đa để nạp.
    
    Returns:
        volumes_list (list): Danh sách các volume ảnh 3D (shape: H, W, Z).
        masks_list (list): Danh sách các volume mask 3D tương ứng.
    """
    volumes_list = []
    masks_list = []
    
    if not os.path.exists(directory):
        print(f"Lỗi: Không tìm thấy thư mục dataset tại {directory}.", file=sys.stderr)
        return [], []

    patient_folders = sorted([d for d in os.listdir(directory) 
                             if os.path.isdir(os.path.join(directory, d)) 
                             and d.startswith('patient')])
    patient_count = 0

    for patient_folder in patient_folders:
        if max_patients and patient_count >= max_patients:
            break

        patient_path = os.path.join(directory, patient_folder)
        info_cfg_path = os.path.join(patient_path, 'Info.cfg')

        # Read ED/ES frames from Info.cfg
        ed_frame, es_frame = -1, -1
        if os.path.exists(info_cfg_path):
            parser = configparser.ConfigParser()
            try:
                with open(info_cfg_path, 'r') as f:
                    config_string = '[DEFAULT]\n' + f.read()
                parser.read_string(config_string)
                ed_frame = int(parser['DEFAULT']['ED'])
                es_frame = int(parser['DEFAULT']['ES'])
            except Exception as e:
                print(f"Warning: Cannot read Info.cfg for {patient_folder}: {e}. Skipping.", 
                      file=sys.stderr)
                continue
        else:
            print(f"Warning: Info.cfg not found for {patient_folder}. Skipping.", file=sys.stderr)
            continue
            
        # File names for ED and ES frames
        ed_img_filename = f'{patient_folder}_frame{ed_frame:02d}.nii.gz'
        es_img_filename = f'{patient_folder}_frame{es_frame:02d}.nii.gz'
        ed_mask_filename = f'{patient_folder}_frame{ed_frame:02d}_gt.nii.gz'
        es_mask_filename = f'{patient_folder}_frame{es_frame:02d}_gt.nii.gz'

        # Try .nii if .nii.gz not found
        for suffix in ['.gz', '']:
            ed_img_path = os.path.join(patient_path, ed_img_filename.replace('.gz', '') if suffix == '' else ed_img_filename)
            es_img_path = os.path.join(patient_path, es_img_filename.replace('.gz', '') if suffix == '' else es_img_filename)
            ed_mask_path = os.path.join(patient_path, ed_mask_filename.replace('.gz', '') if suffix == '' else ed_mask_filename)
            es_mask_path = os.path.join(patient_path, es_mask_filename.replace('.gz', '') if suffix == '' else es_mask_filename)
            
            if os.path.exists(ed_img_path):
                break

        # Helper function to load and process a 3D volume
        def _load_nifti_volume(img_fpath, mask_fpath, target_sz):
            try:
                if not os.path.exists(img_fpath):
                    return None, None

                img_nifti = nib.load(img_fpath)
                img_data = img_nifti.get_fdata()

                mask_data = None
                if os.path.exists(mask_fpath):
                    mask_nifti = nib.load(mask_fpath)
                    mask_data = mask_nifti.get_fdata()

                num_slices = img_data.shape[2]
                resized_img_vol = np.zeros((target_sz[0], target_sz[1], num_slices), dtype=np.float32)
                
                resized_mask_vol = None
                if mask_data is not None:
                    resized_mask_vol = np.zeros((target_sz[0], target_sz[1], num_slices), dtype=np.uint8)

                for i in range(num_slices):
                    resized_img_vol[:, :, i] = resize(
                        img_data[:, :, i], target_sz, order=1, preserve_range=True,
                        anti_aliasing=True, mode='reflect'
                    )
                    if mask_data is not None:
                        resized_mask_vol[:, :, i] = resize(
                            mask_data[:, :, i], target_sz, order=0, preserve_range=True,
                            anti_aliasing=False, mode='reflect'
                        )
                
                return resized_img_vol, resized_mask_vol
            except Exception as e:
                print(f"Error processing volume {img_fpath}: {e}", file=sys.stderr)
                return None, None

        # Load and add volumes to lists
        ed_vol, ed_mask_vol = _load_nifti_volume(ed_img_path, ed_mask_path, target_size)
        if ed_vol is not None:
            volumes_list.append(ed_vol)
            if ed_mask_vol is not None:
                masks_list.append(ed_mask_vol)

        es_vol, es_mask_vol = _load_nifti_volume(es_img_path, es_mask_path, target_size)
        if es_vol is not None:
            volumes_list.append(es_vol)
            if es_mask_vol is not None:
                masks_list.append(es_mask_vol)
        
        patient_count += 1
        
    print(f"Loaded {len(volumes_list)} volumes from {patient_count} patients.")
    return volumes_list, masks_list

