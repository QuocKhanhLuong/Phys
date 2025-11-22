"""
Loss Functions - Exact from Notebook
From: final-application-maxwell-for-segmentation-task (3).ipynb
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict


class FocalTverskyLoss(nn.Module):
    """
    Hàm mất mát Focal Tversky Loss.
    Kết hợp Tversky Index để xử lý mất cân bằng class và Focal Loss để tập trung vào các mẫu khó.
    """
    def __init__(self, 
                 num_classes: int, 
                 alpha: float = 0.3, 
                 beta: float = 0.7, 
                 gamma: float = 4.0 / 3.0, 
                 epsilon: float = 1e-6):
        """
        Args:
            num_classes (int): Số lượng class phân vùng (bao gồm cả background).
            alpha (float): Trọng số cho False Positives (FP).
            beta (float): Trọng số cho False Negatives (FN).
            gamma (float): Tham số focal. Giá trị > 1 để tập trung vào mẫu khó.
            epsilon (float): Hằng số nhỏ để tránh chia cho 0.
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Đầu ra raw từ model, shape (B, C, H, W).
            targets (torch.Tensor): Ground truth, shape (B, H, W).

        Returns:
            torch.Tensor: Giá trị loss vô hướng.
        """
        # Áp dụng softmax để có xác suất
        probs = F.softmax(logits, dim=1)
        
        # Chuyển target sang dạng one-hot
        targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        class_losses = []
        # Bỏ qua background (class 0) vì nó thường chiếm ưu thế và dễ đoán
        for class_idx in range(1, self.num_classes):
            pred_class = probs[:, class_idx, :, :]
            target_class = targets_one_hot[:, class_idx, :, :]
            
            # Làm phẳng tensor để tính toán
            pred_flat = pred_class.contiguous().view(-1)
            target_flat = target_class.contiguous().view(-1)

            # Tính các thành phần True Positives (TP), False Positives (FP), False Negatives (FN)
            tp = torch.sum(pred_flat * target_flat)
            fp = torch.sum(pred_flat * (1 - target_flat))
            fn = torch.sum((1 - pred_flat) * target_flat)
            
            # Tính Tversky Index (TI)
            tversky_index = (tp + self.epsilon) / (tp + self.alpha * fp + self.beta * fn + self.epsilon)
            
            # Tính Focal Tversky Loss (FTL) cho class hiện tại
            # **Sử dụng công thức đã được sửa đổi và kiểm chứng: (1 - TI)^γ**
            focal_tversky_loss = torch.pow(1 - tversky_index, self.gamma)
            
            class_losses.append(focal_tversky_loss)
            
        # Lấy trung bình loss của các class foreground
        if not class_losses:
             return torch.tensor(0.0, device=logits.device) # Tránh lỗi nếu chỉ có 1 class

        total_loss = torch.mean(torch.stack(class_losses))
        
        return total_loss


class FocalLoss(nn.Module):
    """
    Hàm mất mát Focal Loss cho bài toán phân vùng đa lớp.
    Kế thừa từ https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/focal_loss.py
    """
    def __init__(self,
                 gamma: float = 2.0,
                 alpha: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        """
        Args:
            gamma (float): Tham số focal. Giá trị càng lớn, mô hình càng tập trung vào mẫu khó.
            alpha (torch.Tensor, optional): Trọng số cho mỗi class, shape (C,).
            reduction (str, optional): 'mean', 'sum' hoặc 'none'.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Đầu ra raw từ model, shape (B, C, H, W).
            targets (torch.Tensor): Ground truth, shape (B, H, W).

        Returns:
            torch.Tensor: Giá trị loss vô hướng.
        """
        # Tính CE loss gốc
        ce_loss = F.cross_entropy(logits, targets.long(), reduction='none')
        
        # Lấy xác suất của class đúng (p_t)
        # pt.shape: (B, H, W)
        pt = torch.exp(-ce_loss)
        
        # Tính Focal Loss
        # (1-pt)^gamma * ce_loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.device != focal_loss.device:
                self.alpha = self.alpha.to(focal_loss.device)
            
            # Lấy alpha tương ứng với từng pixel
            alpha_t = self.alpha.gather(0, targets.view(-1)).view_as(targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class PhysicsLoss(nn.Module):
    """
    Physics-informed loss based on Helmholtz equation residual.
    """
    def __init__(self, scale_factor=0.01):
        super().__init__()
        omega, mu_0, eps_0 = 2 * np.pi * 42.58e6, 4 * np.pi * 1e-7, 8.854187817e-12
        self.k0 = torch.tensor(omega * np.sqrt(mu_0 * eps_0), dtype=torch.float32)
        self.scale_factor = scale_factor
        
    def forward(self, b1, eps, sig):
        from src.utils.helpers import compute_helmholtz_residual
        
        eps = eps.to(b1.device)
        sig = sig.to(b1.device)
        
        residual = compute_helmholtz_residual(b1, eps, sig, self.k0)
        return torch.mean(residual) * self.scale_factor


class AnatomicalRuleLoss(nn.Module):
    """
    Tính toán loss dựa trên quy tắc giải phẫu về vị trí tương đối của các vùng tim.
    - Phạt khi Tâm thất trái (LV) không được bao quanh bởi Cơ tim (MYO).
    - Phạt khi Tâm thất phải (RV) nằm cạnh Cơ tim (MYO).
    """
    def __init__(self, class_indices: Dict[str, int]):
        """
        Args:
            class_indices (Dict[str, int]): Dictionary ánh xạ tên class sang chỉ số.
                                          Cần chứa các key: 'LV', 'MYO', 'RV'.
        """
        super().__init__()
        if not all(k in class_indices for k in ['LV', 'MYO', 'RV']):
            raise ValueError("class_indices must contain keys 'LV', 'MYO', and 'RV'.")
        self.class_indices = class_indices

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Đầu ra raw từ model, shape (B, C, H, W).

        Returns:
            torch.Tensor: Giá trị loss vô hướng.
        """
        pred_probs = torch.softmax(logits, dim=1)
        
        # Lấy bản đồ xác suất cho từng class
        lv_prob = pred_probs[:, self.class_indices['LV']]
        myo_prob = pred_probs[:, self.class_indices['MYO']]
        rv_prob = pred_probs[:, self.class_indices['RV']]

        # Mô phỏng phép giãn nở (dilation) bằng max_pool2d để tìm vùng lân cận
        dilated_lv_prob = F.max_pool2d(lv_prob.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)

        # Phạt 1: Vùng bao quanh LV (dilated_lv_prob) không phải là MYO
        loss1 = dilated_lv_prob * (1 - myo_prob)

        # Phạt 2: Phạt khi vùng bao quanh LV lại là RV
        loss2 = dilated_lv_prob * rv_prob

        # Kết hợp và lấy trung bình
        total_rule_loss = torch.mean(loss1 + loss2)
        return total_rule_loss


class DynamicLossWeighter(nn.Module):
    """
    Điều chỉnh trọng số cho nhiều thành phần loss một cách tự động,
    đảm bảo tổng các trọng số luôn bằng 1 bằng cách sử dụng Softmax.
    """
    def __init__(self, num_losses: int, tau: float = 1.0, initial_weights: Optional[List[float]] = None):
        """
        Args:
            num_losses (int): Số lượng thành phần loss cần cân bằng.
            tau (float): Hệ số nhiệt độ (temperature) cho hàm softmax.
                         - tau > 1: làm cho các trọng số "mềm" hơn (gần bằng nhau hơn).
                         - 0 < tau < 1: làm cho các trọng số "cứng" hơn (chênh lệch nhiều hơn).
                         - tau = 1: softmax tiêu chuẩn.
            initial_weights (Optional[List[float]]): Trọng số khởi tạo. Phải có tổng bằng 1.
                                                     Nếu là None, sẽ khởi tạo đều.
        """
        super().__init__()
        assert num_losses > 0, "Number of losses must be positive"
        assert tau > 0, "Temperature (tau) must be positive"
        self.num_losses = num_losses
        self.tau = tau

        if initial_weights:
            assert len(initial_weights) == num_losses, \
                f"Number of initial weights ({len(initial_weights)}) must be equal to num_losses ({num_losses})"
            initial_weights_tensor = torch.tensor(initial_weights, dtype=torch.float32)
            assert torch.isclose(initial_weights_tensor.sum(), torch.tensor(1.0)), \
                "Sum of initial weights must be 1"
            # Khởi tạo tham số logit từ log của trọng số ban đầu
            # để softmax(params) xấp xỉ initial_weights
            initial_params = torch.log(initial_weights_tensor)
        else:
            # Khởi tạo bằng 0 sẽ cho ra các trọng số đều nhau sau khi qua softmax
            initial_params = torch.zeros(num_losses, dtype=torch.float32)

        # 'params' là các logit thô mà optimizer sẽ học
        self.params = nn.Parameter(initial_params)

    def forward(self, individual_losses: torch.Tensor) -> torch.Tensor:
        """
        Tính toán tổng loss đã được cân bằng trọng số.

        Args:
            individual_losses (torch.Tensor): Một tensor 1D chứa các giá trị loss
                                              của từng thành phần.

        Returns:
            torch.Tensor: Giá trị loss tổng hợp (scalar).
        """
        if not isinstance(individual_losses, torch.Tensor):
            individual_losses = torch.stack(individual_losses)

        assert individual_losses.dim() == 1 and individual_losses.size(0) == self.num_losses, \
            f"Input individual_losses must be a 1D tensor of size {self.num_losses}"

        # 1. Tính toán các trọng số bằng cách áp dụng softmax lên các tham số có thể học
        weights = F.softmax(self.params / self.tau, dim=0)
        
        # 2. Tính loss tổng hợp bằng cách nhân các loss thành phần với trọng số tương ứng
        # Đây là phép nhân element-wise và sau đó tính tổng (dot product)
        total_loss = torch.sum(weights * individual_losses)

        return total_loss

    def get_current_weights(self) -> Dict[str, float]:
        """
        Lấy các giá trị trọng số hiện tại để theo dõi.
        Các trọng số này có tổng bằng 1.
        """
        with torch.no_grad():
            weights = F.softmax(self.params / self.tau, dim=0)
            return {f"weight_{i}": w.item() for i, w in enumerate(weights)}


class CombinedLoss(nn.Module):
    """
    Combined loss được cập nhật để sử dụng Focal Loss thay cho CE Loss.
    Exact from notebook.
    """
    def __init__(self, 
                 num_classes=4, 
                 initial_loss_weights: Optional[List[float]] = None,
                 class_indices_for_rules: Optional[Dict[str, int]] = None):
        super().__init__()
        
        # --- Initialize loss components ---
        
        # 1. FOCAL LOSS
        self.fl = FocalLoss(gamma=2.0)
        print("Initialized with Focal Loss (gamma=2.0).")
        
        # 2. FOCAL TVERSKY LOSS - Exact values from notebook
        self.ftl = FocalTverskyLoss(
            num_classes=num_classes, 
            alpha=0.2,      # ← From notebook!
            beta=0.8,       # ← From notebook!
            gamma=4.0/3.0
        )
        print("Initialized with Focal Tversky Loss (alpha=0.2, beta=0.8, gamma=4/3).")

        # 3. Physics Loss
        self.pl = PhysicsLoss(scale_factor=0.01)
        
        # 4. Anatomical Rule Loss - ABLATION: DISABLED
        # if class_indices_for_rules is None:
        #     raise ValueError("`class_indices_for_rules` must be provided.")
        # self.arl = AnatomicalRuleLoss(class_indices=class_indices_for_rules)
        self.arl = None  # ABLATION STUDY: No Anatomical Loss
        print("⚠️  ABLATION STUDY: Anatomical Loss DISABLED")
        
        # Khởi tạo bộ cân bằng trọng số cho 3 thành phần (was 4)
        self.loss_weighter = DynamicLossWeighter(num_losses=3, initial_weights=initial_loss_weights)

    def forward(self, logits, targets, b1=None, all_es=None):
        # --- Calculate individual loss components ---
        l_fl = self.fl(logits, targets)  # Tính Focal Loss
        l_ftl = self.ftl(logits, targets)  # Tính Focal Tversky Loss

        lphy = torch.tensor(0.0, device=logits.device)
        if self.pl is not None and b1 is not None and all_es:
            try:
                e1, s1 = all_es[0]
                lphy = self.pl(b1, e1, s1)
            except (IndexError, TypeError):
                print("Warning: Physics loss skipped due to unexpected `all_es` format.")
        
        # ABLATION: No Anatomical Loss
        # larl = self.arl(logits) if self.arl else torch.tensor(0.0, device=logits.device)
        
        # --- Kết hợp 3 thành phần loss (was 4) ---
        individual_losses = torch.stack([l_fl, l_ftl, lphy])
        total_loss = self.loss_weighter(individual_losses)
        
        return total_loss

    def get_current_loss_weights(self) -> Dict[str, float]:
        """Helper để theo dõi trọng số giữa các hàm loss."""
        weights = self.loss_weighter.get_current_weights()
        # Cập nhật tên cho rõ ràng (3 components for ablation study)
        return {
            "weight_FocalLoss": weights["weight_0"],
            "weight_FocalTverskyLoss": weights["weight_1"],
            "weight_Physics": weights["weight_2"]
            # "weight_Anatomical": weights["weight_3"]  # ABLATION: REMOVED
        }
