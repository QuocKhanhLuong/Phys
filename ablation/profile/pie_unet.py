"""
PIE-UNet: Configurable version of RobustMedVFL_UNet

This model is EXACTLY the same as src/models/unet.py RobustMedVFL_UNet,
but with configurable n_channels (C_in) and depth parameters.

ONLY these 2 parameters change:
- n_channels: Number of input slices (C_in in ablation table)
- depth: Number of encoder levels (3, 4, or 5)

All other architecture components are IDENTICAL to the original.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.models.epure import ePURE
from src.models.maxwell_solver import MaxwellSolver
from src.utils.helpers import adaptive_smoothing


class BasicConvBlock(nn.Module):
    """Standard convolutional block: Conv -> BN -> ReLU (SAME as original)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                           padding=padding, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """Encoder block with ePURE noise estimation (SAME as original)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block1 = BasicConvBlock(in_channels, out_channels)
        self.conv_block2 = BasicConvBlock(out_channels, out_channels)
        self.noise_estimator = ePURE(in_channels=in_channels)

    def forward(self, x):
        noise_profile = self.noise_estimator(x)
        x_smoothed = adaptive_smoothing(x, noise_profile)
        x = self.conv_block1(x_smoothed)
        x = self.conv_block2(x)
        return x


class ASPPConv(nn.Sequential):
    """ASPP convolution branch (SAME as original)"""
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class ASPPPooling(nn.Sequential):
    """ASPP global average pooling branch (SAME as original)"""
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
    """ASPP bottleneck (SAME as original)"""
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


class PIE_UNet(nn.Module):
    """
    Configurable UNet++ - IDENTICAL to RobustMedVFL_UNet except:
    - n_channels: configurable (C_in)
    - depth: configurable (number of levels, e.g. 5 = [16,32,64,128,256])
    
    For depth=5, n_channels=5: This is EXACTLY the same as original model.
    """
    def __init__(self, n_channels=5, n_classes=4, depth=5, base_filters=16, deep_supervision=True):
        super().__init__()
        self.depth = depth
        self.deep_supervision = deep_supervision
        
        # Feature channels at each level (SAME as original: [16, 32, 64, 128, 256] for depth=5)
        # depth=5 means 5 levels: channels[0..4]
        channels = [base_filters * (2 ** i) for i in range(depth)]
        self.channels = channels
        
        # Number of encoder blocks = depth - 1 (original has 4 encoders for depth=5)
        self.num_encoders = depth - 1
        num_encoders = self.num_encoders

        # --- Encoder (Column j=0) - SAME as original ---
        self.encoders = nn.ModuleList()
        in_ch = n_channels
        for i in range(num_encoders):
            self.encoders.append(EncoderBlock(in_ch, channels[i]))
            in_ch = channels[i]
        
        # Bottleneck ASPP at deepest level (channels[depth-2] -> channels[depth-1])
        self.bottleneck = BottleneckASPP(channels[num_encoders-1], channels[num_encoders])

        # --- Pooling & Upsampling (SAME as original) ---
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # --- Skip connections (UNet++ style, SAME as original) ---
        # Build all decoder blocks based on depth
        self.decoder_blocks = nn.ModuleDict()
        
        # For depth=5: j from 1 to 4 (columns 1-4)
        for j in range(1, num_encoders + 1):  # columns 1 to num_encoders
            for i in range(num_encoders + 1 - j):  # rows 0 to (num_encoders-j)
                in_channels_dec = channels[i] * j + channels[i+1]
                key = f"conv{i}_{j}"
                self.decoder_blocks[key] = BasicConvBlock(in_channels_dec, channels[i])

        # --- Maxwell Solvers (SAME structure as original) ---
        # Original has: maxwell_solver1/2/3 for columns 2/3/4 and final_decoder for column 1
        self.maxwell_solvers = nn.ModuleDict()
        
        # final_decoder_maxwell_solver for column 1 (x0_1)
        self.maxwell_solvers["final_decoder"] = MaxwellSolver(channels[0] + channels[1])
        
        # maxwell_solver1/2/3... for columns 2, 3, 4, ... (x0_2, x0_3, x0_4, ...)
        for j in range(2, num_encoders + 1):
            in_ch = channels[0] * j + channels[1]
            self.maxwell_solvers[f"solver{j-1}"] = MaxwellSolver(in_ch)

        # --- Output layers (SAME as original) ---
        if self.deep_supervision:
            # Original has final1/2/3/4 for depth=5 (i.e., num_encoders outputs)
            self.finals = nn.ModuleList([
                nn.Conv2d(channels[0], n_classes, kernel_size=1)
                for _ in range(self.num_encoders)
            ])
        else:
            self.final = nn.Conv2d(channels[0], n_classes, kernel_size=1)

    def forward(self, x):
        num_encoders = self.num_encoders
        
        # --- Encoder Path (SAME as original) ---
        enc_outputs = {}
        
        current = x
        for i, encoder in enumerate(self.encoders):
            enc_outputs[(i, 0)] = encoder(current)
            current = self.pool(enc_outputs[(i, 0)])
        
        # Bottleneck (stored at index num_encoders, 0)
        enc_outputs[(num_encoders, 0)] = self.bottleneck(current)

        # --- Skip Path & Decoder (UNet++, SAME as original) ---
        es_list = []
        
        for j in range(1, num_encoders + 1):
            for i in range(num_encoders + 1 - j):
                # Collect previous features in this row
                row_features = [enc_outputs[(i, k)] for k in range(j)]
                # Upsample from below
                below_feature = self.up(enc_outputs[(i+1, j-1)])
                
                # Concatenate
                concat_input = torch.cat(row_features + [below_feature], dim=1)
                
                # Apply decoder block
                key = f"conv{i}_{j}"
                enc_outputs[(i, j)] = self.decoder_blocks[key](concat_input)
                
                # Maxwell solver for row 0 (SAME as original)
                if i == 0:
                    if j == 1:
                        # final_decoder_maxwell_solver
                        es = self.maxwell_solvers["final_decoder"](concat_input)
                    else:
                        # maxwell_solver1/2/3...
                        es = self.maxwell_solvers[f"solver{j-1}"](concat_input)
                    es_list.append(es)
        
        # Reorder to match original: (es1, es2, es3, es_final_decoder)
        # Original order: es1 from x0_2, es2 from x0_3, es3 from x0_4, es_final from x0_1
        # Our order: j=1 (final), j=2 (solver1), j=3 (solver2), j=4 (solver3)
        # Reorder: [1:] + [0] to get solver1, solver2, ..., final
        if len(es_list) > 1:
            all_es_tuples = tuple(es_list[1:] + [es_list[0]])
        else:
            all_es_tuples = tuple(es_list)
        
        # --- Output (SAME as original) ---
        if self.deep_supervision:
            outputs = []
            for j in range(1, num_encoders + 1):
                outputs.append(self.finals[j-1](enc_outputs[(0, j)]))
            return outputs, all_es_tuples
        else:
            final_output = self.final(enc_outputs[(0, num_encoders)])
            return final_output, all_es_tuples


def get_model_for_profile(profile_name, n_classes=4, deep_supervision=True):
    """Factory function to create a model based on profile name."""
    from ablation.profile.config import PROFILE_CONFIGS
    
    if profile_name not in PROFILE_CONFIGS:
        raise ValueError(f"Unknown profile: {profile_name}. Available: {list(PROFILE_CONFIGS.keys())}")
    
    config = PROFILE_CONFIGS[profile_name]
    
    model = PIE_UNet(
        n_channels=config["n_channels"],
        n_classes=n_classes,
        depth=config["depth"],
        base_filters=config["base_filters"],
        deep_supervision=deep_supervision
    )
    
    return model


if __name__ == "__main__":
    # Test that depth=5, n_channels=5 matches original model params
    print("Testing PIE-UNet vs Original RobustMedVFL_UNet...")
    
    from src.models.unet import RobustMedVFL_UNet
    
    # Original model
    original = RobustMedVFL_UNet(n_channels=5, n_classes=4, deep_supervision=True)
    orig_params = sum(p.numel() for p in original.parameters())
    
    # PIE_UNet with same config
    pie = PIE_UNet(n_channels=5, n_classes=4, depth=5, base_filters=16, deep_supervision=True)
    pie_params = sum(p.numel() for p in pie.parameters())
    
    print(f"Original RobustMedVFL_UNet params: {orig_params:,}")
    print(f"PIE_UNet (depth=5, n_channels=5) params: {pie_params:,}")
    print(f"Match: {orig_params == pie_params}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 5, 224, 224)
    
    orig_out, orig_es = original(dummy_input)
    pie_out, pie_es = pie(dummy_input)
    
    print(f"\nOriginal outputs: {len(orig_out)}, ES tuples: {len(orig_es)}")
    print(f"PIE_UNet outputs: {len(pie_out)}, ES tuples: {len(pie_es)}")
