import os
import json
from tqdm import tqdm
import numpy as np
import glob

from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.utility.dictionary import EnsureChannelFirstd, EnsureTyped
from monai.transforms.spatial.dictionary import Spacingd
from monai.transforms.intensity.dictionary import NormalizeIntensityd
from monai.transforms.croppad.dictionary import CropForegroundd
from monai.transforms.spatial.dictionary import Resized


def preprocess_brats21_with_monai(input_dir, output_dir, target_spacing=(1.0, 1.0, 1.0)):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}\n"
            f"Please pass the real BraTS root directory, e.g. --input_dir /data/BraTS2021_TrainingData"
        )

    os.makedirs(output_dir, exist_ok=True)
    volumes_dir = os.path.join(output_dir, 'volumes')
    masks_dir = os.path.join(output_dir, 'masks')
    os.makedirs(volumes_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    keys = ["image", "label"]
    tfm = Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Spacingd(keys=keys, pixdim=target_spacing, mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=keys, source_key="image"),
        Resized(keys=keys, spatial_size=(224, 224, -1), mode=("trilinear", "nearest")),
        EnsureTyped(keys=keys),
    ])

    # Collect patients: robust detection of BraTS folders
    candidate_dirs = [
        d for d in glob.glob(os.path.join(input_dir, '*')) if os.path.isdir(d)
    ]
    patient_folders = []
    for d in candidate_dirs:
        pid = os.path.basename(d)
        # Expect files like <pid>_t1.nii.gz ... <pid>_seg.nii.gz
        required = [
            os.path.join(d, f"{pid}_t1.nii.gz"),
            os.path.join(d, f"{pid}_t1ce.nii.gz"),
            os.path.join(d, f"{pid}_t2.nii.gz"),
            os.path.join(d, f"{pid}_flair.nii.gz"),
            os.path.join(d, f"{pid}_seg.nii.gz"),
        ]
        if all(os.path.exists(p) for p in required):
            patient_folders.append(pid)

    if not patient_folders:
        raise FileNotFoundError(
            "No valid BraTS patient folders found. Ensure input_dir contains folders like 'BraTS2021_XXXX' with *_t1.nii.gz, *_t1ce.nii.gz, *_t2.nii.gz, *_flair.nii.gz, *_seg.nii.gz"
        )
    patient_slice_counts = {}

    for patient in tqdm(patient_folders, desc="Preprocess"):
        patient_id = patient
        file_dict = {
            "image": [
                os.path.join(input_dir, patient, f"{patient_id}_t1.nii.gz"),
                os.path.join(input_dir, patient, f"{patient_id}_t1ce.nii.gz"),
                os.path.join(input_dir, patient, f"{patient_id}_t2.nii.gz"),
                os.path.join(input_dir, patient, f"{patient_id}_flair.nii.gz"),
            ],
            "label": os.path.join(input_dir, patient, f"{patient_id}_seg.nii.gz"),
        }

        try:
            processed = tfm(file_dict)
            volume = processed['image'].numpy()
            label = processed['label'].numpy().squeeze().astype(np.uint8)

            # Remap BraTS labels {1,2,4} -> {1,2,3}
            label_remapped = np.zeros_like(label, dtype=np.uint8)
            label_remapped[label == 1] = 1
            label_remapped[label == 2] = 2
            label_remapped[label == 4] = 3

            np.save(os.path.join(volumes_dir, f"{patient_id}.npy"), volume)
            np.save(os.path.join(masks_dir, f"{patient_id}.npy"), label_remapped)
            patient_slice_counts[patient_id] = int(volume.shape[-1])
        except Exception as e:
            print(f"Failed {patient_id}: {e}")

    metadata = {
        'target_spacing': list(target_spacing),
        'patient_slices': patient_slice_counts,
        'modalities': ['t1', 't1ce', 't2', 'flair'],
        'num_classes': 4,
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata for {len(patient_slice_counts)} patients at {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MONAI preprocessing for BraTS21")
    parser.add_argument('--input_dir', required=True, help='Path to raw BraTS21 root')
    parser.add_argument('--output_dir', required=True, help='Path to preprocessed NPY root')
    parser.add_argument('--spacing', nargs=3, type=float, default=(1.0, 1.0, 1.0))
    args = parser.parse_args()

    preprocess_brats21_with_monai(args.input_dir, args.output_dir, tuple(args.spacing))


