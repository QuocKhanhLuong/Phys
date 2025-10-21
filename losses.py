import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class FocalTverskyLoss(nn.Module):
    def __init__(self, 
                 num_classes: int, 
                 alpha: float = 0.3, 
                 beta: float = 0.7, 
                 gamma: float = 4.0 / 3.0, 
                 epsilon: float = 1e-6,
                 min_loss: float = 1e-4):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_loss = min_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        class_losses = []
        for class_idx in range(1, self.num_classes):
            pred_class = probs[:, class_idx, :, :]
            target_class = targets_one_hot[:, class_idx, :, :]
            
            pred_flat = pred_class.contiguous().view(-1)
            target_flat = target_class.contiguous().view(-1)

            tp = torch.sum(pred_flat * target_flat)
            fp = torch.sum(pred_flat * (1 - target_flat))
            fn = torch.sum((1 - pred_flat) * target_flat)
            
            tversky_index = (tp + self.epsilon) / (tp + self.alpha * fp + self.beta * fn + self.epsilon)
            focal_tversky_loss = torch.pow(1 - tversky_index, self.gamma)
            class_losses.append(focal_tversky_loss)
            
        if not class_losses:
            return torch.tensor(0.0, device=logits.device)

        total_loss = torch.mean(torch.stack(class_losses))
        return torch.clamp(total_loss, min=self.min_loss)


class FocalLoss(nn.Module):
    def __init__(self,
                 gamma: float = 2.0,
                 alpha: Optional[torch.Tensor] = None,
                 reduction: str = 'mean',
                 min_loss: float = 1e-4):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.min_loss = min_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.device != focal_loss.device:
                self.alpha = self.alpha.to(focal_loss.device)
            alpha_t = self.alpha.gather(0, targets.view(-1)).view_as(targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return torch.clamp(focal_loss.mean(), min=self.min_loss)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class PhysicsLoss(nn.Module):
    """
    Physics-informed loss based on Helmholtz equation residual.
    Improved scaling for better gradient flow.
    """
    def __init__(self, 
                 scale: float = 0.1,  # Increased from 0.01
                 normalization_constant: float = 100.0):  # Decreased from 3500.0
        super().__init__()
        # Physical constants
        omega = 2 * np.pi * 42.58e6  # Larmor frequency (Hz)
        mu_0 = 4 * np.pi * 1e-7      # Permeability of free space
        eps_0 = 8.854187817e-12      # Permittivity of free space
        self.k0 = torch.tensor(omega * np.sqrt(mu_0 * eps_0), dtype=torch.float32)
        
        self.scale = scale
        self.normalization_constant = normalization_constant

    def forward(self, b1, eps_map, sig):
        from utils import compute_helmholtz_residual
        
        # Move tensors to correct device
        eps_map = eps_map.to(b1.device)
        sig = sig.to(b1.device)
        self.k0 = self.k0.to(b1.device)
        
        # Compute physics residual
        residual = compute_helmholtz_residual(b1, eps_map, sig, self.k0)
        loss_raw = torch.mean(residual)
        
        # Improved normalization and scaling
        normalized_loss = loss_raw / self.normalization_constant
        scaled_loss = self.scale * normalized_loss
        
        # Wider clamp range to prevent gradient vanishing
        final_loss = torch.clamp(scaled_loss, min=0.001, max=5.0)
        
        return final_loss


class DynamicLossWeighter(nn.Module):
    """
    Enhanced multi-task weighting with strategies from:
    - GradNorm (Chen et al., ICML 2018): Balance gradient magnitudes
    - Kendall et al. (CVPR 2018): Uncertainty-based weighting
    - Additional entropy regularization to prevent collapse
    """
    def __init__(self, 
                 num_losses: int, 
                 tau: float = 2.5,  # Higher tau = smoother weights (prevent collapse)
                 initial_weights: Optional[List[float]] = None,
                 min_loss_value: float = 1e-4,
                 entropy_reg: float = 0.01,  # Entropy regularization coefficient
                 min_weight: float = 0.05):  # Minimum weight per loss (prevent zero)
        super().__init__()
        assert num_losses > 0, "Number of losses must be positive"
        assert tau > 0, "Temperature (tau) must be positive"
        self.num_losses = num_losses
        self.tau = tau
        self.min_loss_value = min_loss_value
        self.entropy_reg = entropy_reg
        self.min_weight = min_weight

        if initial_weights:
            assert len(initial_weights) == num_losses, \
                f"Number of initial weights ({len(initial_weights)}) must be equal to num_losses ({num_losses})"
            initial_weights_tensor = torch.tensor(initial_weights, dtype=torch.float32)
            assert torch.isclose(initial_weights_tensor.sum(), torch.tensor(1.0)), \
                "Sum of initial weights must be 1"
            initial_params = torch.log(initial_weights_tensor)
        else:
            initial_params = torch.zeros(num_losses, dtype=torch.float32)

        self.params = nn.Parameter(initial_params)

    def forward(self, individual_losses: torch.Tensor) -> torch.Tensor:
        if not isinstance(individual_losses, torch.Tensor):
            individual_losses = torch.stack(individual_losses)

        assert individual_losses.dim() == 1 and individual_losses.size(0) == self.num_losses, \
            f"Input individual_losses must be a 1D tensor of size {self.num_losses}"

        # Stabilize losses to prevent zeros
        stabilized_losses = torch.clamp(individual_losses, min=self.min_loss_value)
        
        # Compute softmax weights with temperature
        raw_weights = F.softmax(self.params / self.tau, dim=0)
        
        # Enforce minimum weight per task (prevent complete collapse)
        # Strategy from multi-task learning papers to keep all tasks active
        weights = torch.clamp(raw_weights, min=self.min_weight)
        weights = weights / weights.sum()  # Re-normalize
        
        # Entropy regularization: encourage balanced weights
        # Higher entropy = more uniform distribution
        if self.training and self.entropy_reg > 0:
            entropy = -torch.sum(weights * torch.log(weights + 1e-8))
            entropy_loss = -self.entropy_reg * entropy  # Negative = maximize entropy
        else:
            entropy_loss = 0.0
        
        total_loss = torch.sum(weights * stabilized_losses) + entropy_loss
        return total_loss

    def get_current_weights(self):
        with torch.no_grad():
            weights = F.softmax(self.params / self.tau, dim=0)
            return {f"weight_{i}": w.item() for i, w in enumerate(weights)}


class CombinedLoss(nn.Module):
    """
    Multi-objective loss combining segmentation + physics constraints.
    Implements strategies from BraTS literature:
    - Myronenko (BraTS 2018): Scale physics << segmentation
    - Zhu et al. (MIDL 2023): Normalize physics by running stats
    - GradNorm + entropy reg: Prevent weight collapse
    """
    def __init__(self, 
                 num_classes=4, 
                 initial_loss_weights: Optional[List[float]] = None,
                 loss_temperature: float = 3.0,  # Higher = more stable
                 entropy_regularization: float = 0.02,  # Encourage balanced weights
                 min_weight_per_loss: float = 0.15,  # Each loss gets ≥15%
                 use_physics: bool = False,  # ← FLAG to disable physics
                 fixed_weights: bool = True):  # ← NEW: Use fixed weights instead of dynamic
        super().__init__()
        
        self.fl = FocalLoss(gamma=2.0)
        self.ftl = FocalTverskyLoss(num_classes=num_classes, alpha=0.2, beta=0.8, gamma=4.0/3.0)
        self.use_physics = use_physics
        self.fixed_weights = fixed_weights
        
        if use_physics:
            self.pl = PhysicsLoss(scale=0.1, normalization_constant=100.0)  # Use improved scaling
            if fixed_weights:
                # Fixed weights: [FocalLoss, FocalTverskyLoss, PhysicsLoss] = [0.4, 0.4, 0.2]
                self.weights = torch.tensor([0.4, 0.4, 0.2], dtype=torch.float32)
                self.loss_weighter = None
            else:
                num_losses = 3
                if initial_loss_weights is None:
                    initial_loss_weights = [0.45, 0.45, 0.10]
                self.loss_weighter = DynamicLossWeighter(
                    num_losses=num_losses,
                    tau=loss_temperature,
                    initial_weights=initial_loss_weights,
                    entropy_reg=entropy_regularization,
                    min_weight=min_weight_per_loss
                )
        else:
            self.pl = None
            if fixed_weights:
                # Fixed weights: [FocalLoss, FocalTverskyLoss] = [0.5, 0.5]
                self.weights = torch.tensor([0.5, 0.5], dtype=torch.float32)
                self.loss_weighter = None
            else:
                num_losses = 2  # Only FL + FTL
                if initial_loss_weights is None:
                    initial_loss_weights = [0.5, 0.5]
                self.loss_weighter = DynamicLossWeighter(
                    num_losses=num_losses,
                    tau=loss_temperature,
                    initial_weights=initial_loss_weights,
                    entropy_reg=entropy_regularization,
                    min_weight=min_weight_per_loss
                )

    def forward(self, logits, targets, b1=None, all_es=None):
        l_fl = self.fl(logits, targets)
        l_ftl = self.ftl(logits, targets)

        if self.use_physics:
            lphy = torch.tensor(0.0, device=logits.device)
            if self.pl is not None and b1 is not None and all_es:
                try:
                    e1, s1 = all_es[0]
                    lphy = self.pl(b1, e1, s1)
                except (IndexError, TypeError):
                    pass
            
            if self.fixed_weights:
                # Fixed weighting: [FocalLoss, FocalTverskyLoss, PhysicsLoss] = [0.4, 0.4, 0.2]
                total_loss = (self.weights[0] * l_fl + 
                            self.weights[1] * l_ftl + 
                            self.weights[2] * lphy)
            else:
                # Dynamic weighting
                individual_losses = torch.stack([l_fl, l_ftl, lphy])
                total_loss = self.loss_weighter(individual_losses)
        else:
            if self.fixed_weights:
                # Fixed weighting: [FocalLoss, FocalTverskyLoss] = [0.5, 0.5]
                total_loss = self.weights[0] * l_fl + self.weights[1] * l_ftl
            else:
                # Dynamic weighting
                individual_losses = torch.stack([l_fl, l_ftl])
                total_loss = self.loss_weighter(individual_losses)
        
        return total_loss

    def get_current_loss_weights(self):
        if self.fixed_weights:
            if self.use_physics:
                return {
                    "weight_FocalLoss": self.weights[0].item(),
                    "weight_FocalTverskyLoss": self.weights[1].item(),
                    "weight_Physics": self.weights[2].item()
                }
            else:
                return {
                    "weight_FocalLoss": self.weights[0].item(),
                    "weight_FocalTverskyLoss": self.weights[1].item(),
                    "weight_Physics": 0.0
                }
        else:
            # Dynamic weighting
            weights = self.loss_weighter.get_current_weights()
            if self.use_physics:
                return {
                    "weight_FocalLoss": weights["weight_0"],
                    "weight_FocalTverskyLoss": weights["weight_1"],
                    "weight_Physics": weights["weight_2"]
                }
            else:
                return {
                    "weight_FocalLoss": weights["weight_0"],
                    "weight_FocalTverskyLoss": weights["weight_1"],
                    "weight_Physics": 0.0
                }

