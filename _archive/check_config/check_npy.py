import numpy as np
import matplotlib.pyplot as plt
import sys
import os

NPY_DIR = "./BraTS21_preprocessed_monai" 

PATIENT_ID = "BraTS2021_00002" 

volume_path = os.path.join(NPY_DIR, "volumes", f"{PATIENT_ID}.npy")
mask_path = os.path.join(NPY_DIR, "masks", f"{PATIENT_ID}.npy")

print(f"Đang tải: {PATIENT_ID}")

if not os.path.exists(volume_path):
    print(f"LỖI: Không tìm thấy file volume: {volume_path}")
    sys.exit()
    
print(f"\n--- Kiểm tra VOLUME: {volume_path} ---")
volume = np.load(volume_path)

print(f"Shape (Kích thước): {volume.shape}") 
print(f"Kiểu dữ liệu: {volume.dtype}")
print(f"Giá trị Min: {np.min(volume)}")
print(f"Giá trị Max: {np.max(volume)}")

has_nan = np.isnan(volume).any()
print(f"Có chứa NaN không?: {has_nan}")
if has_nan:
    print(">>> CẢNH BÁO: FILE VOLUME NÀY BỊ LỖI NaN! <<<")


if not os.path.exists(mask_path):
    print(f"LỖI: Không tìm thấy file mask: {mask_path}")
    sys.exit()
    
print(f"\n--- Kiểm tra MASK: {mask_path} ---")
mask = np.load(mask_path)

print(f"Shape (Kích thước): {mask.shape}") # (H, W, D)
print(f"Kiểu dữ liệu: {mask.dtype}")
print(f"Các giá trị nhãn duy nhất: {np.unique(mask)}")

is_empty = (np.max(mask) == 0)
print(f"Mask có rỗng không?: {is_empty}")
if is_empty:
    print(">>> CẢNH BÁO: FILE MASK NÀY RỖNG (TOÀN SỐ 0)! <<<")


if not is_empty:
    tumor_area_per_slice = np.sum(mask > 0, axis=(0, 1))
    slice_idx = np.argmax(tumor_area_per_slice)
    
    print(f"\nTìm thấy lát cắt có khối u lớn nhất tại: {slice_idx}")
else:
    slice_idx = mask.shape[2] // 2
    print(f"\nMask rỗng, chọn lát cắt giữa: {slice_idx}")

if not has_nan:
    # Lấy ra 4 modality từ volume
    t1_slice = volume[0, :, :, slice_idx]
    t1ce_slice = volume[1, :, :, slice_idx]
    t2_slice = volume[2, :, :, slice_idx]
    flair_slice = volume[3, :, :, slice_idx]
    
    # Lấy ra mask
    mask_slice = mask[:, :, slice_idx]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Bệnh nhân: {PATIENT_ID} - Lát cắt: {slice_idx}", fontsize=16)
    
    axes[0, 0].imshow(t1_slice, cmap='gray')
    axes[0, 0].set_title("T1")
    
    axes[0, 1].imshow(t1ce_slice, cmap='gray')
    axes[0, 1].set_title("T1ce")
    
    axes[0, 2].imshow(t2_slice, cmap='gray')
    axes[0, 2].set_title("T2")
    
    axes[1, 0].imshow(flair_slice, cmap='gray')
    axes[1, 0].set_title("FLAIR")
    
    axes[1, 1].imshow(mask_slice, cmap='jet')
    axes[1, 1].set_title("Mask (Nhãn)")
    
    axes[1, 2].imshow(flair_slice, cmap='gray')
    axes[1, 2].imshow(np.ma.masked_where(mask_slice == 0, mask_slice), cmap='jet', alpha=0.5)
    axes[1, 2].set_title("FLAIR + Mask")
    
    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')
            
    plt.tight_layout()
    
    # Lưu file PNG
    output_filename = f"{PATIENT_ID}_slice_{slice_idx}_preprocessed_check.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close(fig) # Đóng figure để giải phóng bộ nhớ
    print(f"\nĐã lưu ảnh kiểm tra vào: {output_filename}")

else:
    print("\nBỏ qua hiển thị ảnh do phát hiện lỗi (NaN).")