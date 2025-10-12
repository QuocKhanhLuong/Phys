"""
Dataset and data loading utilities for ACDC cardiac MRI dataset.
Supports 2.5D input with multiple slices and optional data augmentation.
"""

import os
import sys
import configparser
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from skimage.transform import resize

from utils import adaptive_quantum_noise_injection


# =============================================================================
# --- ACDC Dataset 2.5D ---
# =============================================================================

class ACDCDataset25D(Dataset):
    """
    Dataset cho ACDC, nạp dữ liệu 2.5D.
    NÂNG CẤP:
    - Tùy chỉnh số lát cắt đầu vào.
    - Tích hợp thêm nhiễu lượng tử thích nghi như một bước augmentation.
    """
    def __init__(self, volumes_list, masks_list, num_input_slices=5, transforms=None, 
                 noise_injector_model=None, device='cpu'):
        """
        Args:
            volumes_list (list): Danh sách các volume ảnh 3D.
            masks_list (list): Danh sách các volume mask 3D tương ứng.
            num_input_slices (int): Số lát cắt liên tục để xếp chồng.
            transforms (albumentations.Compose): Pipeline các phép biến đổi hình học.
            noise_injector_model (nn.Module, optional): Mô hình ePURE để tạo noise map.
            device (str): Thiết bị để chạy noise_injector_model.
        """
        if num_input_slices % 2 == 0:
            raise ValueError("num_input_slices phải là một số lẻ.")
            
        self.volumes = volumes_list
        self.masks = masks_list
        self.num_input_slices = num_input_slices
        self.transforms = transforms
        self.noise_injector_model = noise_injector_model
        self.device = device
        
        self.index_map = []
        for vol_idx, vol in enumerate(self.volumes):
            radius = (self.num_input_slices - 1) // 2
            num_slices = vol.shape[2]
            for slice_idx in range(radius, num_slices - radius):
                self.index_map.append((vol_idx, slice_idx))
    
    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        vol_idx, center_slice_idx = self.index_map[idx]
        
        current_volume = self.volumes[vol_idx]
        current_mask_volume = self.masks[vol_idx]
        num_slices_in_vol = current_volume.shape[2]
    
        radius = (self.num_input_slices - 1) // 2
        offsets = range(-radius, radius + 1)
        
        slice_indices = [np.clip(center_slice_idx + offset, 0, num_slices_in_vol - 1) for offset in offsets]
        
        image_stack = np.stack(
            [current_volume[:, :, i] for i in slice_indices],
            axis=-1
        ).astype(np.float32)
        
        mask = current_mask_volume[:, :, center_slice_idx]
        
        if self.transforms:
            augmented = self.transforms(image=image_stack, mask=mask)
            image_tensor = augmented['image']
            mask_tensor = augmented['mask']
        else:
            image_tensor = torch.from_numpy(image_stack).permute(2, 0, 1)
            mask_tensor = torch.from_numpy(mask)
    
        # --- Tăng cường dữ liệu với quantum noise injection ---
        if self.noise_injector_model is not None:
            with torch.no_grad():
                # Chuyển ảnh lên device để tạo noise map
                img_on_gpu_with_batch = image_tensor.to(self.device).unsqueeze(0)
                noise_map = self.noise_injector_model(img_on_gpu_with_batch)
                
                # Áp dụng nhiễu lượng tử
                image_tensor_with_noise_gpu = adaptive_quantum_noise_injection(
                    img_on_gpu_with_batch,
                    noise_map
                )
                
                # Chuyển kết quả cuối cùng về lại CPU và bỏ chiều batch
                image_tensor = image_tensor_with_noise_gpu.squeeze(0).cpu()
                
        return image_tensor, mask_tensor.long()


# =============================================================================
# --- Data Loading from ACDC ---
# =============================================================================

def load_acdc_volumes(directory, target_size=(224, 224), max_patients=None):
    """
    Nạp các volume MRI từ thư mục ACDC dataset.
    
    Args:
        directory (str): Đường dẫn đến thư mục chứa dữ liệu ACDC.
        target_size (tuple): Kích thước mục tiêu để resize các slice.
        max_patients (int, optional): Số lượng bệnh nhân tối đa để nạp.
    
    Returns:
        volumes_list (list): Danh sách các volume ảnh 3D.
        masks_list (list): Danh sách các volume mask 3D tương ứng.
    """
    volumes_list = []
    masks_list = []
    
    if not os.path.exists(directory):
        print(f"Lỗi: Không tìm thấy thư mục dataset tại {directory}.", file=sys.stderr)
        return [], []

    patient_folders = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    patient_count = 0

    for patient_folder in patient_folders:
        if max_patients and patient_count >= max_patients:
            break

        patient_path = os.path.join(directory, patient_folder)
        info_cfg_path = os.path.join(patient_path, 'Info.cfg')

        # --- Đọc frame ED/ES từ file Info.cfg ---
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
                print(f"Cảnh báo: Không thể đọc Info.cfg cho {patient_folder}: {e}. Bỏ qua bệnh nhân.", file=sys.stderr)
                continue
        else:
            print(f"Cảnh báo: Không tìm thấy Info.cfg cho {patient_folder}. Bỏ qua bệnh nhân.", file=sys.stderr)
            continue
            
        ed_img_filename = f'{patient_folder}_frame{ed_frame:02d}.nii'
        es_img_filename = f'{patient_folder}_frame{es_frame:02d}.nii'
        ed_mask_filename = f'{patient_folder}_frame{ed_frame:02d}_gt.nii'
        es_mask_filename = f'{patient_folder}_frame{es_frame:02d}_gt.nii'

        ed_img_path = os.path.join(patient_path, ed_img_filename)
        es_img_path = os.path.join(patient_path, es_img_filename)
        ed_mask_path = os.path.join(patient_path, ed_mask_filename)
        es_mask_path = os.path.join(patient_path, es_mask_filename)

        # --- Hàm helper để nạp và xử lý một volume 3D ---
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
                print(f"Lỗi khi xử lý volume {img_fpath}: {e}", file=sys.stderr)
                return None, None

        # --- Nạp và thêm các volume vào danh sách ---
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
        
    return volumes_list, masks_list

