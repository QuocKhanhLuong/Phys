"""
Dataset ATLAS tối ưu (Optimized) với Memory-Mapped I/O và LRU Cache.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from collections import OrderedDict
from typing import Optional, List
import json
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Thêm thư mục gốc vào path
sys.path.append(str(Path(__file__).parent.parent.parent))


class ATLASDataset25DOptimized(Dataset):
    """
    Dataset ATLAS 2.5D tối ưu (Optimized)
    (Copy logic từ ACDCDataset25DOptimized)
    """
    def __init__(
        self, 
        npy_dir: str,
        volume_ids: Optional[List[str]] = None,
        num_input_slices: int = 5,
        transforms=None,
        noise_injector_model=None,
        device: str = 'cpu',
        max_cache_size: int = 15,
        use_memmap: bool = True
    ):
        if num_input_slices % 2 == 0:
            raise ValueError("num_input_slices phải là số lẻ")
        
        self.num_input_slices = num_input_slices
        self.transforms = transforms
        self.noise_injector_model = noise_injector_model
        self.device = device
        self.max_cache_size = max_cache_size
        self.use_memmap = use_memmap
        self._volume_cache = OrderedDict()
        
        volumes_dir = os.path.join(npy_dir, 'volumes')
        masks_dir = os.path.join(npy_dir, 'masks')
        
        metadata_path = os.path.join(npy_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata.json không tìm thấy tại: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        volume_info = metadata.get('volume_info', {})
        if not volume_info:
            raise ValueError(f"metadata.json thiếu 'volume_info'")
        
        if volume_ids is not None:
            self.volume_paths = [os.path.join(volumes_dir, f'{vid}.npy') for vid in volume_ids]
            self.mask_paths = [os.path.join(masks_dir, f'{vid}.npy') for vid in volume_ids]
        else:
            import glob
            self.volume_paths = sorted(glob.glob(os.path.join(volumes_dir, '*.npy')))
            self.mask_paths = sorted(glob.glob(os.path.join(masks_dir, '*.npy')))
        
        self.index_map = []
        radius = (num_input_slices - 1) // 2
        
        for vol_idx, vol_path in enumerate(self.volume_paths):
            volume_id = os.path.basename(vol_path).replace('.npy', '')
            if volume_id not in volume_info:
                print(f"Cảnh báo: {volume_id} không có trong metadata.json. Bỏ qua.")
                continue
            
            num_slices = volume_info[volume_id]['num_slices']
            for slice_idx in range(radius, num_slices - radius):
                self.index_map.append((vol_idx, slice_idx))
        
        print(f"✓ ATLASDataset25DOptimized: {len(self.index_map)} slices từ {len(self.volume_paths)} volumes")
        print(f"  Cache: max {max_cache_size} volumes | Memmap: {use_memmap}")
    
    def _load_volume(self, vol_idx: int):
        if vol_idx in self._volume_cache:
            self._volume_cache.move_to_end(vol_idx)
            return self._volume_cache[vol_idx]
        
        if self.use_memmap:
            current_volume = np.load(self.volume_paths[vol_idx], mmap_mode='r')
            current_mask_volume = np.load(self.mask_paths[vol_idx], mmap_mode='r')
        else:
            current_volume = np.load(self.volume_paths[vol_idx])
            current_mask_volume = np.load(self.mask_paths[vol_idx])
        
        self._volume_cache[vol_idx] = (current_volume, current_mask_volume)
        if len(self._volume_cache) > self.max_cache_size:
            self._volume_cache.popitem(last=False)
        
        return current_volume, current_mask_volume
    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        vol_idx, center_slice_idx = self.index_map[idx]
        current_volume, current_mask_volume = self._load_volume(vol_idx)
        
        num_slices_in_vol = current_volume.shape[2]
        radius = (self.num_input_slices - 1) // 2
        
        slice_indices = np.clip(
            center_slice_idx + np.arange(-radius, radius + 1),
            0, num_slices_in_vol - 1
        )
        
        image_stack = current_volume[:, :, slice_indices].astype(np.float32)
        mask = current_mask_volume[:, :, center_slice_idx].astype(np.int64)
        
        if self.transforms:
            augmented = self.transforms(image=image_stack, mask=mask)
            image_tensor = augmented['image']
            mask_tensor = augmented['mask']
        else:
            image_tensor = torch.from_numpy(image_stack).permute(2, 0, 1)
            mask_tensor = torch.from_numpy(mask)
        
        if self.noise_injector_model is not None:
            from src.utils.helpers import adaptive_quantum_noise_injection
            with torch.no_grad():
                img_on_gpu_with_batch = image_tensor.to(self.device).unsqueeze(0)
                noise_map = self.noise_injector_model(img_on_gpu_with_batch)
                image_tensor_with_noise_gpu = adaptive_quantum_noise_injection(
                    img_on_gpu_with_batch, noise_map
                )
                image_tensor = image_tensor_with_noise_gpu.squeeze(0).cpu()
        
        return image_tensor, mask_tensor.long()

# --- Các hàm helper (Copy từ ACDC) ---

def get_atlas_volume_ids(npy_dir: str):
    """Lấy danh sách ID volume (patient_id) từ thư mục NPY."""
    volumes_dir = os.path.join(npy_dir, 'volumes')
    import glob
    volume_files = sorted(glob.glob(os.path.join(volumes_dir, '*.npy')))
    volume_ids = [os.path.basename(f).replace('.npy', '') for f in volume_files]
    return volume_ids

def split_atlas_by_patient(volume_ids: List[str], val_ratio: float = 0.2, random_state: int = 42):
    """
    Tách ATLAS volumes (đã là 1-1 với patient).
    """
    # ATLAS không có ED/ES, nên chỉ cần tách trực tiếp
    patient_ids = sorted(volume_ids)
    
    train_patients, val_patients = train_test_split(
        patient_ids,
        test_size=val_ratio,
        random_state=random_state
    )
    
    print(f"Tách ATLAS dataset:")
    print(f"  Train: {len(train_patients)} bệnh nhân")
    print(f"  Val:   {len(val_patients)} bệnh nhân")
    
    return train_patients, val_patients