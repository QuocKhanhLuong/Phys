import torch
from torch.utils.data import Dataset
import numpy as np
import os
import nibabel as nib
from skimage.transform import resize
import glob


class BraTS21Dataset25D(Dataset):
    def __init__(self, volumes_list, masks_list, num_input_slices=5, transforms=None, 
                 noise_injector_model=None, device: str = 'cpu'):
        if num_input_slices % 2 == 0:
            raise ValueError("num_input_slices must be odd.")
            
        self.volumes = volumes_list
        self.masks = masks_list
        self.num_input_slices = num_input_slices
        self.transforms = transforms
        self.noise_injector_model = noise_injector_model
        self.device = device
        
        self.index_map = []
        for vol_idx, vol in enumerate(self.volumes):
            radius = (self.num_input_slices - 1) // 2
            num_slices = vol.shape[3]
            for slice_idx in range(radius, num_slices - radius):
                self.index_map.append((vol_idx, slice_idx))
    
    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        vol_idx, center_slice_idx = self.index_map[idx]
        
        current_volume = self.volumes[vol_idx]
        current_mask_volume = self.masks[vol_idx]
        num_slices_in_vol = current_volume.shape[3]
    
        radius = (self.num_input_slices - 1) // 2
        offsets = range(-radius, radius + 1)
        
        slice_indices = [np.clip(center_slice_idx + offset, 0, num_slices_in_vol - 1) for offset in offsets]
        
        image_stack = np.stack(
            [current_volume[:, :, :, i].transpose(1, 2, 0) for i in slice_indices],
            axis=-1
        ).astype(np.float32)
        
        image_stack = image_stack.reshape(image_stack.shape[0], image_stack.shape[1], -1)
        
        mask = current_mask_volume[:, :, center_slice_idx].astype(np.int64)
        
        if self.transforms:
            augmented = self.transforms(image=image_stack, mask=mask)
            image_tensor = augmented['image']
            mask_tensor = augmented['mask']
        else:
            image_tensor = torch.from_numpy(image_stack.transpose(2, 0, 1))
            mask_tensor = torch.from_numpy(mask)
    
        if self.noise_injector_model is not None:
            from utils import adaptive_quantum_noise_injection
            with torch.no_grad():
                img_on_gpu_with_batch = image_tensor.to(self.device).unsqueeze(0)
                noise_map = self.noise_injector_model(img_on_gpu_with_batch)
                image_tensor_with_noise_gpu = adaptive_quantum_noise_injection(
                    img_on_gpu_with_batch,
                    noise_map
                )
                image_tensor = image_tensor_with_noise_gpu.squeeze(0).cpu()
                
        return image_tensor, mask_tensor.long()


def load_brats21_volumes(directory, target_size=(224, 224), max_patients=None):
    volumes_list = []
    masks_list = []
    
    if not os.path.exists(directory):
        print(f"Error: Dataset directory not found at {directory}.")
        return [], []

    patient_folders = sorted(glob.glob(os.path.join(directory, 'patient*')))
    if not patient_folders:
        patient_folders = sorted(glob.glob(os.path.join(directory, 'BraTS2021_*')))
    
    volume_count = 0

    for patient_path in patient_folders:
        if max_patients and volume_count >= max_patients:
            break
        
        if not os.path.isdir(patient_path):
            continue

        patient_id = os.path.basename(patient_path)
        file_4d = os.path.join(patient_path, f'{patient_id}_4d.nii')
        
        if not os.path.exists(file_4d):
            print(f"Warning: 4D file not found for {patient_id}. Skipping.")
            continue

        try:
            data_4d = nib.load(file_4d).get_fdata()
            gt_files = sorted(glob.glob(os.path.join(patient_path, f'{patient_id}_frame*_gt.nii')))
            
            if not gt_files:
                print(f"Warning: No ground truth files for {patient_id}. Skipping.")
                continue
            
            for gt_file in gt_files:
                gt_basename = os.path.basename(gt_file)
                frame_str = gt_basename.split('_')[1]
                frame_num = int(frame_str.replace('frame', ''))
                frame_idx = frame_num - 1
                
                if frame_idx >= data_4d.shape[3]:
                    print(f"Warning: Frame {frame_num} out of range for {patient_id}. Skipping.")
                    continue
                
                frame_volume = data_4d[:, :, :, frame_idx]
                volume = np.stack([frame_volume] * 4, axis=0)
                
                gt_data = nib.load(gt_file).get_fdata()
                mask = gt_data.astype(np.uint8)
                
                num_slices = volume.shape[3]
                resized_volume = np.zeros((4, target_size[0], target_size[1], num_slices), dtype=np.float32)
                resized_mask = np.zeros((target_size[0], target_size[1], num_slices), dtype=np.uint8)
                
                for i in range(num_slices):
                    for mod_idx in range(4):
                        resized_volume[mod_idx, :, :, i] = resize(
                            volume[mod_idx, :, :, i], target_size, order=1, preserve_range=True,
                            anti_aliasing=True, mode='reflect'
                        )
                    resized_mask[:, :, i] = resize(
                        mask[:, :, i], target_size, order=0, preserve_range=True,
                        anti_aliasing=False, mode='reflect'
                    )
                
                volumes_list.append(resized_volume)
                masks_list.append(resized_mask)
                volume_count += 1
                print(f"Loaded {patient_id} - {frame_str}")
            
        except Exception as e:
            print(f"Error processing {patient_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nTotal: {len(volumes_list)} volumes from {len(patient_folders)} patients")
    return volumes_list, masks_list

