"""
Optimized ACDC Dataset with Memory-Mapped I/O and LRU Cache.
Much faster than loading NIfTI files directly!

Usage:
    1. First run: python scripts/preprocess_acdc.py --input data/ACDC/training --output preprocessed_data/ACDC/training
    2. Then use this dataset class for training
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

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))


class ACDCDataset25DOptimized(Dataset):
    """
    Optimized 2.5D ACDC Dataset with:
    - Memory-mapped arrays (memmap) - don't load entire volumes into RAM
    - LRU cache - keep frequently used volumes in memory  
    - Vectorized slice extraction - faster than loops
    - Metadata - know num_slices without loading volumes
    
    ~10x faster I/O than loading NIfTI files directly!
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
        """
        Args:
            npy_dir: Path to preprocessed .npy files (e.g., 'preprocessed_data/ACDC/training')
            volume_ids: Optional list of volume IDs to use (for train/val split)
            num_input_slices: Number of slices for 2.5D (must be odd)
            transforms: Albumentations transforms
            noise_injector_model: ePURE model for noise injection
            device: Device for ePURE ('cuda' or 'cpu')
            max_cache_size: Max number of volumes to keep in LRU cache
            use_memmap: Use memory-mapped arrays (recommended for CPU training)
        """
        if num_input_slices % 2 == 0:
            raise ValueError("num_input_slices must be odd")
        
        self.num_input_slices = num_input_slices
        self.transforms = transforms
        self.noise_injector_model = noise_injector_model
        self.device = device
        self.max_cache_size = max_cache_size
        self.use_memmap = use_memmap
        self._volume_cache = OrderedDict()  # LRU cache
        
        # Directories
        volumes_dir = os.path.join(npy_dir, 'volumes')
        masks_dir = os.path.join(npy_dir, 'masks')
        
        # Load metadata
        metadata_path = os.path.join(npy_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"metadata.json not found: {metadata_path}\n"
                f"Run: python scripts/preprocess_acdc.py --input <acdc_dir> --output {npy_dir}"
            )
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        volume_info = metadata.get('volume_info', {})
        if not volume_info:
            raise ValueError(
                f"metadata.json missing 'volume_info'\n"
                f"Run preprocessing again: python scripts/preprocess_acdc.py"
            )
        
        # Get volume paths
        if volume_ids is not None:
            # Use specified volume IDs (for train/val split)
            self.volume_paths = [os.path.join(volumes_dir, f'{vid}.npy') for vid in volume_ids]
            self.mask_paths = [os.path.join(masks_dir, f'{vid}.npy') for vid in volume_ids]
        else:
            # Use all volumes
            import glob
            self.volume_paths = sorted(glob.glob(os.path.join(volumes_dir, '*.npy')))
            self.mask_paths = sorted(glob.glob(os.path.join(masks_dir, '*.npy')))
        
        # Build index map (vol_idx, slice_idx) for all valid slices
        self.index_map = []
        radius = (num_input_slices - 1) // 2
        
        for vol_idx, vol_path in enumerate(self.volume_paths):
            volume_id = os.path.basename(vol_path).replace('.npy', '')
            
            if volume_id not in volume_info:
                raise KeyError(
                    f"Volume {volume_id} not in metadata\n"
                    f"Run preprocessing again"
                )
            
            num_slices = volume_info[volume_id]['num_slices']
            
            # Only include valid center slices (not too close to boundaries)
            for slice_idx in range(radius, num_slices - radius):
                self.index_map.append((vol_idx, slice_idx))
        
        print(f"✓ ACDCDataset25DOptimized: {len(self.index_map)} slices from {len(self.volume_paths)} volumes")
        print(f"  Cache: max {max_cache_size} volumes | Memmap: {use_memmap}")
    
    def _load_volume(self, vol_idx: int):
        """
        Load volume with LRU cache management.
        Uses memory-mapped arrays to avoid loading entire file into RAM.
        """
        # Check cache first
        if vol_idx in self._volume_cache:
            # Move to end (most recently used)
            self._volume_cache.move_to_end(vol_idx)
            return self._volume_cache[vol_idx]
        
        # Load from disk
        if self.use_memmap:
            # Memory-mapped arrays - only load slices when accessed
            # Much faster and uses less memory!
            current_volume = np.load(self.volume_paths[vol_idx], mmap_mode='r')
            current_mask_volume = np.load(self.mask_paths[vol_idx], mmap_mode='r')
        else:
            # Load entire file into memory (slower)
            current_volume = np.load(self.volume_paths[vol_idx])
            current_mask_volume = np.load(self.mask_paths[vol_idx])
        
        # Add to cache
        self._volume_cache[vol_idx] = (current_volume, current_mask_volume)
        
        # Remove oldest if cache full (LRU eviction)
        if len(self._volume_cache) > self.max_cache_size:
            self._volume_cache.popitem(last=False)  # Remove first (oldest)
        
        return current_volume, current_mask_volume
    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        """
        Get 2.5D slice stack with optimized I/O.
        """
        vol_idx, center_slice_idx = self.index_map[idx]
        current_volume, current_mask_volume = self._load_volume(vol_idx)
        
        num_slices_in_vol = current_volume.shape[2]
        radius = (self.num_input_slices - 1) // 2
        
        # Vectorized slice extraction (faster than loop!)
        slice_indices = np.clip(
            center_slice_idx + np.arange(-radius, radius + 1),
            0, num_slices_in_vol - 1
        )
        
        # Extract all slices at once: (H, W, Z) -> (H, W, num_input_slices)
        # This is much faster than looping!
        image_stack = current_volume[:, :, slice_indices].astype(np.float32)
        mask = current_mask_volume[:, :, center_slice_idx].astype(np.int64)
        
        # Apply Albumentations transforms
        if self.transforms:
            augmented = self.transforms(image=image_stack, mask=mask)
            image_tensor = augmented['image']  # (C, H, W) after ToTensorV2
            mask_tensor = augmented['mask']
        else:
            # Manual conversion to tensor
            image_tensor = torch.from_numpy(image_stack).permute(2, 0, 1)  # (C, H, W)
            mask_tensor = torch.from_numpy(mask)
        
        # Apply ePURE quantum noise injection (if provided)
        # NOTE: Requires GPU access, so num_workers MUST be 0 in DataLoader!
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


def get_acdc_volume_ids(npy_dir: str):
    """
    Get list of all volume IDs from preprocessed directory.
    Useful for train/val split.
    
    Returns:
        List of volume IDs (e.g., ['patient001_ED', 'patient001_ES', ...])
    """
    volumes_dir = os.path.join(npy_dir, 'volumes')
    import glob
    volume_files = sorted(glob.glob(os.path.join(volumes_dir, '*.npy')))
    volume_ids = [os.path.basename(f).replace('.npy', '') for f in volume_files]
    return volume_ids


def split_acdc_by_patient(volume_ids: List[str], val_ratio: float = 0.2, random_state: int = 42):
    """
    Split ACDC volumes by patient (not by volume!).
    Each patient has 2 volumes (ED and ES) - keep them together.
    
    Args:
        volume_ids: List of volume IDs (e.g., ['patient001_ED', 'patient001_ES', ...])
        val_ratio: Validation split ratio
        random_state: Random seed for reproducibility
    
    Returns:
        train_volume_ids, val_volume_ids
    """
    # Group volumes by patient
    from collections import defaultdict
    patient_volumes = defaultdict(list)
    
    for vid in volume_ids:
        patient_id = vid.rsplit('_', 1)[0]  # 'patient001_ED' -> 'patient001'
        patient_volumes[patient_id].append(vid)
    
    # Split patients
    patient_ids = sorted(patient_volumes.keys())
    
    from sklearn.model_selection import train_test_split
    train_patients, val_patients = train_test_split(
        patient_ids,
        test_size=val_ratio,
        random_state=random_state
    )
    
    # Get volume IDs for each split
    train_volume_ids = []
    for pid in train_patients:
        train_volume_ids.extend(patient_volumes[pid])
    
    val_volume_ids = []
    for pid in val_patients:
        val_volume_ids.extend(patient_volumes[pid])
    
    print(f"Split ACDC dataset:")
    print(f"  Train: {len(train_patients)} patients, {len(train_volume_ids)} volumes")
    print(f"  Val:   {len(val_patients)} patients, {len(val_volume_ids)} volumes")
    
    return train_volume_ids, val_volume_ids


# Example usage
if __name__ == '__main__':
    """
    Example: How to use the optimized dataset
    """
    # 1. Get all volume IDs
    npy_dir = 'preprocessed_data/ACDC/training'
    all_volume_ids = get_acdc_volume_ids(npy_dir)
    
    # 2. Split by patient
    train_ids, val_ids = split_acdc_by_patient(all_volume_ids, val_ratio=0.2)
    
    # 3. Create datasets
    train_dataset = ACDCDataset25DOptimized(
        npy_dir=npy_dir,
        volume_ids=train_ids,
        num_input_slices=5,
        transforms=None,
        max_cache_size=15,
        use_memmap=True
    )
    
    val_dataset = ACDCDataset25DOptimized(
        npy_dir=npy_dir,
        volume_ids=val_ids,
        num_input_slices=5,
        transforms=None,
        max_cache_size=10,
        use_memmap=True
    )
    
    print(f"Train: {len(train_dataset)} slices")
    print(f"Val:   {len(val_dataset)} slices")

