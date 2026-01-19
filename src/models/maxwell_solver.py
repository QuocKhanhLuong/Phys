"""
MaxwellSolver - Extract epsilon and sigma maps from features.
From notebook: final-application-maxwell-for-segmentation-task (3).ipynb
"""

import torch
import torch.nn as nn


class MaxwellSolver(nn.Module):
    """
    Simplified Maxwell solver that extracts epsilon and sigma maps.
    Used in decoder blocks to provide physics-informed features.
    """
    def __init__(self, in_channels, hidden_dim=32):
        super(MaxwellSolver, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1)
        )

    def forward(self, x):
        """
        Returns:
            eps_map: Epsilon map (permittivity), shape (B, 1, H, W)
            sigma_map: Sigma map (conductivity), shape (B, 1, H, W)
        """
        eps_sigma_map = self.encoder(x)
        return eps_sigma_map[:, 0:1, :, :], eps_sigma_map[:, 1:2, :, :]

