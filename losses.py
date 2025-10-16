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
                 epsilon: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

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
        return total_loss


class FocalLoss(nn.Module):
    def __init__(self,
                 gamma: float = 2.0,
                 alpha: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

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
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class PhysicsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        omega, mu_0, eps_0 = 2 * np.pi * 42.58e6, 4 * np.pi * 1e-7, 8.854187817e-12
        self.k0 = torch.tensor(omega * np.sqrt(mu_0 * eps_0), dtype=torch.float32)

    def forward(self, b1, eps, sig):
        from utils import compute_helmholtz_residual
        eps = eps.to(b1.device)
        sig = sig.to(b1.device)
        residual = compute_helmholtz_residual(b1, eps, sig, self.k0)
        return torch.mean(residual)


class DynamicLossWeighter(nn.Module):
    def __init__(self, num_losses: int, tau: float = 1.0, initial_weights: Optional[List[float]] = None):
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
            initial_params = torch.log(initial_weights_tensor)
        else:
            initial_params = torch.zeros(num_losses, dtype=torch.float32)

        self.params = nn.Parameter(initial_params)

    def forward(self, individual_losses: torch.Tensor) -> torch.Tensor:
        if not isinstance(individual_losses, torch.Tensor):
            individual_losses = torch.stack(individual_losses)

        assert individual_losses.dim() == 1 and individual_losses.size(0) == self.num_losses, \
            f"Input individual_losses must be a 1D tensor of size {self.num_losses}"

        weights = F.softmax(self.params / self.tau, dim=0)
        total_loss = torch.sum(weights * individual_losses)
        return total_loss

    def get_current_weights(self):
        with torch.no_grad():
            weights = F.softmax(self.params / self.tau, dim=0)
            return {f"weight_{i}": w.item() for i, w in enumerate(weights)}


class CombinedLoss(nn.Module):
    def __init__(self, 
                 num_classes=4, 
                 initial_loss_weights: Optional[List[float]] = None):
        super().__init__()
        
        self.fl = FocalLoss(gamma=2.0)
        self.ftl = FocalTverskyLoss(num_classes=num_classes, alpha=0.2, beta=0.8, gamma=4.0/3.0)
        self.pl = PhysicsLoss()
        
        self.loss_weighter = DynamicLossWeighter(num_losses=3, initial_weights=initial_loss_weights)

    def forward(self, logits, targets, b1=None, all_es=None):
        l_fl = self.fl(logits, targets)
        l_ftl = self.ftl(logits, targets)

        lphy = torch.tensor(0.0, device=logits.device)
        if self.pl is not None and b1 is not None and all_es:
            try:
                e1, s1 = all_es[0]
                lphy = self.pl(b1, e1, s1)
            except (IndexError, TypeError):
                pass
        
        individual_losses = torch.stack([l_fl, l_ftl, lphy])
        total_loss = self.loss_weighter(individual_losses)
        return total_loss

    def get_current_loss_weights(self):
        weights = self.loss_weighter.get_current_weights()
        return {
            "weight_FocalLoss": weights["weight_0"],
            "weight_FocalTverskyLoss": weights["weight_1"],
            "weight_Physics": weights["weight_2"]
        }

