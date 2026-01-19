"""
PIE-UNet with Configurable Encoder

Same architecture as RobustMedVFL_UNet but with swappable encoder blocks.
Decoder, ASPP bottleneck, and Maxwell Solver remain unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ablation.encoder.encoder_blocks import get_encoder_block, BasicConvBlock
from src.models.maxwell_solver import MaxwellSolver


# =============================================================================
# ASPP COMPONENTS (Same as unet.py)
# =============================================================================

class ASPPConv(nn.Sequential):
    """ASPP convolution branch with atrous convolution"""
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class ASPPPooling(nn.Sequential):
    """ASPP global average pooling branch"""
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class BottleneckASPP(nn.Module):
    """ASPP bottleneck for multi-scale feature aggregation."""
    def __init__(self, in_channels, out_channels):
        super(BottleneckASPP, self).__init__()
        atrous_rates = [3, 6, 9]
        inter_channels = out_channels // (len(atrous_rates) + 2)
        
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU()
            ),
            ASPPConv(in_channels, inter_channels, atrous_rates[0]),
            ASPPConv(in_channels, inter_channels, atrous_rates[1]),
            ASPPConv(in_channels, inter_channels, atrous_rates[2]),
            ASPPPooling(in_channels, inter_channels)
        ])

        self.project = nn.Sequential(
            nn.Conv2d(inter_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        res = torch.cat(res, dim=1)
        return self.project(res)


# =============================================================================
# PIE-UNet WITH CONFIGURABLE ENCODER
# =============================================================================

class PIE_UNet_Encoder(nn.Module):
    """
    PIE-UNet with configurable encoder type.
    
    Architecture is identical to RobustMedVFL_UNet, but encoder blocks
    can be swapped for ablation study.
    
    Args:
        n_channels: Input channels (default: 5 for 2.5D)
        n_classes: Output classes (default: 4 for ACDC)
        encoder_type: One of 'standard', 'se', 'resnet', 'cbam', 'nae'
        deep_supervision: Enable deep supervision outputs
    """
    def __init__(self, n_channels=5, n_classes=4, encoder_type='nae', deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder_type = encoder_type
        
        # Feature channels at each level (same as original)
        channels = [16, 32, 64, 128, 256]

        # --- Encoder (Column j=0) with configurable encoder blocks ---
        self.conv0_0 = get_encoder_block(encoder_type, n_channels, channels[0])
        self.conv1_0 = get_encoder_block(encoder_type, channels[0], channels[1])
        self.conv2_0 = get_encoder_block(encoder_type, channels[1], channels[2])
        self.conv3_0 = get_encoder_block(encoder_type, channels[2], channels[3])
        self.conv4_0 = BottleneckASPP(channels[3], channels[4])

        # --- Pooling & Upsampling ---
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # --- Skip connections (j > 0) - Standard Conv ---
        # Column j=1
        self.conv0_1 = BasicConvBlock(channels[0] + channels[1], channels[0])
        self.conv1_1 = BasicConvBlock(channels[1] + channels[2], channels[1])
        self.conv2_1 = BasicConvBlock(channels[2] + channels[3], channels[2])
        self.conv3_1 = BasicConvBlock(channels[3] + channels[4], channels[3])

        # Column j=2
        self.conv0_2 = BasicConvBlock(channels[0]*2 + channels[1], channels[0])
        self.conv1_2 = BasicConvBlock(channels[1]*2 + channels[2], channels[1])
        self.conv2_2 = BasicConvBlock(channels[2]*2 + channels[3], channels[2])

        # Column j=3
        self.conv0_3 = BasicConvBlock(channels[0]*3 + channels[1], channels[0])
        self.conv1_3 = BasicConvBlock(channels[1]*3 + channels[2], channels[1])

        # Column j=4
        self.conv0_4 = BasicConvBlock(channels[0]*4 + channels[1], channels[0])

        # --- Maxwell Solvers for decoder paths ---
        self.maxwell_solver1 = MaxwellSolver(channels[0]*2 + channels[1])
        self.maxwell_solver2 = MaxwellSolver(channels[0]*3 + channels[1])
        self.maxwell_solver3 = MaxwellSolver(channels[0]*4 + channels[1])
        self.final_decoder_maxwell_solver = MaxwellSolver(channels[0] + channels[1])

        # --- Output layers ---
        if self.deep_supervision:
            self.final1 = nn.Conv2d(channels[0], n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(channels[0], n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(channels[0], n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(channels[0], n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(channels[0], n_classes, kernel_size=1)

    def forward(self, x):
        all_eps_sigma = []
        
        # Encoder path (Column j=0)
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Column j=1
        x0_1_cat = torch.cat([x0_0, self.up(x1_0)], dim=1)
        all_eps_sigma.append(self.final_decoder_maxwell_solver(x0_1_cat))
        x0_1 = self.conv0_1(x0_1_cat)

        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))

        # Column j=2
        x0_2_cat = torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1)
        all_eps_sigma.append(self.maxwell_solver1(x0_2_cat))
        x0_2 = self.conv0_2(x0_2_cat)

        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))

        # Column j=3
        x0_3_cat = torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1)
        all_eps_sigma.append(self.maxwell_solver2(x0_3_cat))
        x0_3 = self.conv0_3(x0_3_cat)

        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))

        # Column j=4
        x0_4_cat = torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1)
        all_eps_sigma.append(self.maxwell_solver3(x0_4_cat))
        x0_4 = self.conv0_4(x0_4_cat)

        # Outputs
        if self.deep_supervision:
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            out4 = self.final4(x0_4)
            return [out1, out2, out3, out4], all_eps_sigma
        else:
            return [self.final(x0_4)], all_eps_sigma


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing PIE_UNet_Encoder with different encoders...")
    
    x = torch.randn(2, 5, 224, 224)  # Batch=2, Channels=5
    
    for encoder_type in ['standard', 'se', 'resnet', 'cbam', 'nae']:
        model = PIE_UNet_Encoder(n_channels=5, n_classes=4, encoder_type=encoder_type)
        outputs, eps_sigma = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"Encoder: {encoder_type:10} | Output: {outputs[-1].shape} | Params: {params:,}")
    
    print("\nAll encoder variants working correctly!")
