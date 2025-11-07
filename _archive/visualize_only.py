import torch
import os
from data_utils import load_brats21_volumes
from evaluate import visualize_final_results_2_5D
import config

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = config.NUM_CLASSES
NUM_SLICES = config.NUM_SLICES
IMG_SIZE = config.IMG_SIZE

print(f"Device: {DEVICE}")
print("Loading test volumes...")

test_data_path = os.path.join(config.PROJECT_ROOT, 'data', 'testing')
all_test_volumes, all_test_masks = load_brats21_volumes(
    test_data_path, target_size=(IMG_SIZE, IMG_SIZE)
)

print(f"\nLoaded {len(all_test_volumes)} test volumes.")

for i in range(len(all_test_volumes)):
    for mod_idx in range(4):
        max_val = all_test_volumes[i][mod_idx].max()
        if max_val > 0:
            all_test_volumes[i][mod_idx] /= max_val

print("\nStarting visualization...")
visualize_final_results_2_5D(
    volumes_np=all_test_volumes,
    masks_np=all_test_masks,
    num_classes=NUM_CLASSES,
    num_samples=10,
    device=DEVICE,
    num_slices=NUM_SLICES * 4
)

print("\n✓ Visualization complete! Check the 'visualizations/' folder.")

