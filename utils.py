"""
Utility functions for physics-informed medical image processing.
Includes adaptive smoothing, quantum noise injection, and B1 map generation.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from scipy.ndimage import binary_fill_holes, binary_opening
from skimage.transform import resize


# =============================================================================
# --- Adaptive Smoothing ---
# =============================================================================

def adaptive_smoothing(x, noise_profile, kernel_size=5, sigma=1.0):
    """
    Áp dụng làm mịn thích nghi dựa trên noise_profile
    - x: Ảnh đầu vào hoặc feature map [B, C, H, W]
    - noise_profile: Bản đồ nhiễu [B, 1, H, W] (giá trị từ 0 đến 1)
    - kernel_size/sigma: Tham số làm mịn Gaussian
    """
    # Ensure input is float for convolution
    x_float = x.float()

    # Ensure noise_profile is float and 1 channel
    noise_profile_float = noise_profile.float()
    if noise_profile_float.size(1) != 1:
         print(f"Warning: Noise profile expected 1 channel but got {noise_profile_float.size(1)}. Using first channel.")
         noise_profile_float = noise_profile_float[:, :1, :, :]

    # Bước 1: Apply Gaussian blur channel-wise
    if isinstance(kernel_size, int):
        kernel_size_tuple = (kernel_size, kernel_size)
    else:
        kernel_size_tuple = kernel_size

    if isinstance(sigma, (int, float)):
         sigma_tuple = (float(sigma), float(sigma))
    else:
         sigma_tuple = sigma

    # Ensure sigma values are positive to avoid issues
    sigma_tuple = tuple(max(0.1, s) for s in sigma_tuple) # Add small epsilon

    smoothed = TF.gaussian_blur(x_float, kernel_size=kernel_size_tuple, sigma=sigma_tuple)

    # Bước 2: Chuẩn hóa noise_profile (sigmoid) và mở rộng cho đúng số kênh
    blending_weights = torch.sigmoid(noise_profile_float) # [B, 1, H, W]

    # Expand blending_weights to match the number of channels in x
    blending_weights = blending_weights.repeat(1, x_float.size(1), 1, 1) # [B, C, H, W]

    # Ensure dimensions match for blending
    assert blending_weights.shape == x_float.shape, f"Blending weights shape {blending_weights.shape} does not match input shape {x_float.shape}"

    # Bước 3: Trộn ảnh gốc và ảnh đã làm mịn
    weighted_sum = x_float * (1 - blending_weights) + smoothed * blending_weights

    return weighted_sum


# =============================================================================
# --- Adaptive Quantum Noise Injection ---
# =============================================================================

def adaptive_quantum_noise_injection(
    features, 
    noise_map, 
    T_min=0.5, 
    T_max=1.5, 
    pauli_prob={'X': 0.00096, 'Y': 0.00096, 'Z': 0.00096}
):
    """
    Áp dụng nhiễu lượng tử một cách THÍCH NGHI dựa trên noise_map.
    - Nơi noise_map thấp (vùng sạch), T sẽ cao -> thêm nhiều nhiễu.
    - Nơi noise_map cao (vùng nhiễu), T sẽ thấp -> thêm ít nhiễu.
    
    Args:
        features (torch.Tensor): Tensor đầu vào [B, C, H, W].
        noise_map (torch.Tensor): Bản đồ nhiễu từ ePURE [B, 1, H, W].
        T_min (float): Hệ số nhiễu tối thiểu.
        T_max (float): Hệ số nhiễu tối đa.
        pauli_prob (dict): Xác suất cơ sở của các cổng Pauli.
    """
    features_float = features.float()
    noise_map_float = noise_map.float()
    device = features_float.device

    # Bước 1: Tạo bản đồ hệ số nhiễu T (T_map) từ noise_map
    normalized_noise = torch.sigmoid(noise_map_float)
    T_map = T_max - (T_max - T_min) * normalized_noise
    T_map = T_map.repeat(1, features.size(1), 1, 1)

    # Bước 2: Tính toán xác suất Pauli theo không gian
    p_x = pauli_prob['X'] * T_map
    p_y = pauli_prob['Y'] * T_map
    p_z = pauli_prob['Z'] * T_map
    p_none = 1.0 - (p_x + p_y + p_z)
    
    # [B, C, H, W, 4] -> stack các xác suất
    probabilities_map = torch.stack([p_x, p_y, p_z, p_none], dim=-1)
    
    # Bước 3: Lấy mẫu cổng Pauli cho từng pixel
    B, C, H, W = features.shape
    prob_reshaped = probabilities_map.view(-1, 4)
    choice_indices = torch.multinomial(prob_reshaped, 1).view(B, C, H, W)
    
    # Bước 4: Áp dụng nhiễu dựa trên lựa chọn
    noisy_features = features_float.clone()
    
    # Mask cho từng cổng
    mask_x = (choice_indices == 0)
    mask_y = (choice_indices == 1)
    mask_z = (choice_indices == 2)
    
    # Áp dụng cổng Pauli
    noisy_features[mask_x] = 1.0 - noisy_features[mask_x]
    noisy_features[mask_y] = 1.0 - noisy_features[mask_y] + 0.1 * torch.randn_like(noisy_features[mask_y])
    noisy_features[mask_z] = -noisy_features[mask_z]
    
    # Đảm bảo giá trị pixel nằm trong phạm vi hợp lệ
    noisy_features = torch.clamp(noisy_features, 0.0, 1.0)
    
    return noisy_features


# =============================================================================
# --- B1 Map Generation ---
# =============================================================================

class AdvancedB1Simulator(nn.Module):
    """
    Mô phỏng B1 map dựa trên một mảng các cuộn dây bề mặt (surface coils) ngẫu nhiên.
    Cung cấp B1 map chân thực hơn mà vẫn nhẹ và hiệu quả.
    """
    def __init__(self,
                 n_coils_range: tuple = (4, 8),
                 strength_range: tuple = (0.5, 1.5),
                 radius_factor_range: tuple = (0.5, 1.5)):
        super().__init__()
        self.n_coils_range = n_coils_range
        self.strength_range = strength_range
        self.radius_factor_range = radius_factor_range

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = image_batch.shape
        device = image_batch.device

        b1_maps = []
        for i in range(batch_size):
            # 1. Ngẫu nhiên hóa các tham số của mảng coil
            n_coils = torch.randint(self.n_coils_range[0], self.n_coils_range[1] + 1, (1,)).item()
            
            centers_x = torch.randint(-width//4, width + width//4, (n_coils,), device=device)
            centers_y = torch.randint(-height//4, height + height//4, (n_coils,), device=device)
            
            strengths = torch.zeros(n_coils, device=device).uniform_(*self.strength_range)
            base_radius = (height + width) / 4
            radii = torch.zeros(n_coils, device=device).uniform_(*self.radius_factor_range) * base_radius

            # 2. Tạo bản đồ độ nhạy cho từng coil
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

            coil_maps = []
            for j in range(n_coils):
                dist_sq = (x_grid - centers_x[j])**2 + (y_grid - centers_y[j])**2
                sensitivity_map = strengths[j] / (dist_sq + radii[j]**2)
                coil_maps.append(sensitivity_map)
            
            coil_maps = torch.stack(coil_maps, dim=0)

            # 3. Kết hợp các coil map bằng phương pháp "sum of squares"
            combined_map = torch.sqrt(torch.sum(coil_maps**2, dim=0))
            
            # Chuẩn hóa để có giá trị trung bình gần 1
            combined_map = combined_map / (torch.mean(combined_map) + 1e-8)
            
            b1_maps.append(combined_map)

        b1_map_stack = torch.stack(b1_maps, dim=0).unsqueeze(1)

        # Clip về dải giá trị vật lý hợp lý
        b1_map_stack = torch.clamp(b1_map_stack, 0.4, 1.6)

        return b1_map_stack


def calculate_ultimate_common_b1_map(
    all_images: torch.Tensor,
    device: str = 'cuda',
    save_path: str = "ultimate_common_b1_map.pth"
) -> torch.Tensor:
    """
    Tính toán một B1 map chung với độ chính xác cao nhất bằng cách kết hợp:
    1. Mô phỏng coil-array (AdvancedB1Simulator).
    2. Trung bình có trọng số theo chất lượng ảnh và vùng quan tâm (ROI).
    3. Hậu xử lý làm mịn.
    """
    calc_device = torch.device(device if torch.cuda.is_available() else 'cpu')

    if os.path.exists(save_path):
        print(f"Đang tải Ultimate B1 map đã được tính toán từ '{save_path}'...")
        saved_data = torch.load(save_path, map_location=calc_device)
        return saved_data['common_b1_map']

    print("Bắt đầu tính toán Ultimate B1 map mới...")
    
    # Bước 1: Tạo các B1 map chất lượng cao
    b1_simulator = AdvancedB1Simulator().to(calc_device)
    num_images = all_images.shape[0]
    batch_size = 32
    
    all_generated_maps = []
    all_image_stats = []

    print("Tạo các B1 map ngẫu nhiên (chất lượng cao)...")
    with torch.no_grad():
        for i in range(0, num_images, batch_size):
            end_idx = min(i + batch_size, num_images)
            batch_images = all_images[i:end_idx].to(calc_device)
            
            generated_maps = b1_simulator(batch_images)
            all_generated_maps.append(generated_maps.cpu())

            for j in range(batch_images.shape[0]):
                img = batch_images[j].cpu()
                all_image_stats.append({
                    'mean': torch.mean(img).item(),
                    'std': torch.std(img).item()
                })

    all_generated_maps = torch.cat(all_generated_maps, dim=0)

    # Bước 2: Tạo các trọng số cho việc tính trung bình
    print("Tạo trọng số để tính trung bình...")
    
    # a. Trọng số theo chất lượng ảnh
    image_weights = []
    for stats in all_image_stats:
        contrast_score = stats['std'] / (stats['mean'] + 1e-8) if stats['mean'] > 0 else 0
        weight = np.clip(contrast_score, 0.5, 2.0)
        image_weights.append(weight)
    image_weights = torch.tensor(image_weights, dtype=torch.float32).view(-1, 1, 1, 1)

    # b. Trọng số theo vùng không gian
    avg_image = torch.mean(all_images, dim=0).squeeze().numpy()
    roi_mask_np = avg_image > np.mean(avg_image) * 0.5
    roi_mask_np = binary_opening(roi_mask_np, structure=np.ones((5,5)))
    roi_mask_np = binary_fill_holes(roi_mask_np)
    roi_mask = torch.from_numpy(roi_mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    
    spatial_weights = torch.ones_like(roi_mask)
    spatial_weights[roi_mask == 1] = 3.0

    # Bước 3: Tính trung bình có trọng số
    print("Tính toán trung bình có trọng số...")
    weighted_maps = all_generated_maps * image_weights * spatial_weights
    total_weights = image_weights * spatial_weights
    
    common_b1_map = torch.sum(weighted_maps, dim=0, keepdim=True) / (torch.sum(total_weights, dim=0, keepdim=True) + 1e-8)

    # Bước 4: Hậu xử lý làm mịn
    print("Hậu xử lý làm mịn B1 map...")
    common_b1_map = TF.gaussian_blur(common_b1_map, kernel_size=21, sigma=5)
    
    # Chuẩn hóa lại
    common_b1_map = common_b1_map / (torch.mean(common_b1_map) + 1e-8)
    common_b1_map = torch.clamp(common_b1_map, 0.5, 1.5)

    print(f"Lưu Ultimate B1 map vào '{save_path}'...")
    torch.save({'common_b1_map': common_b1_map}, save_path)
    
    print("Tính toán Ultimate B1 map thành công!")
    return common_b1_map.to(calc_device)

