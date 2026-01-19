"""
RobustMedVFL_UNet - UNet++ architecture with ePURE and Maxwell Solver integration.
From notebook: final-application-maxwell-for-segmentation-task (3).ipynb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.epure import ePURE
from src.models.maxwell_solver import MaxwellSolver
from src.utils.helpers import adaptive_smoothing


class BasicConvBlock(nn.Module):
    """Standard convolutional block: Conv -> BN -> ReLU"""
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
    """
    Encoder block with ePURE noise estimation and adaptive smoothing.
    From notebook - integrates physics-inspired preprocessing.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block1 = BasicConvBlock(in_channels, out_channels)
        self.conv_block2 = BasicConvBlock(out_channels, out_channels)
        self.noise_estimator = ePURE(in_channels=in_channels)

    def forward(self, x):
        # Estimate noise and apply adaptive smoothing
        noise_profile = self.noise_estimator(x)
        x_smoothed = adaptive_smoothing(x, noise_profile)
        x = self.conv_block1(x_smoothed)
        x = self.conv_block2(x)
        return x


class DecoderBlock(nn.Module):
    """
    Decoder block with Maxwell Solver integration.
    From notebook - extracts physics properties (epsilon, sigma).
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        concat_ch = in_channels // 2 + skip_channels
        self.maxwell_solver = MaxwellSolver(concat_ch)
        self.conv_block1 = BasicConvBlock(concat_ch, out_channels)
        self.conv_block2 = BasicConvBlock(out_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.up(x)
        # Handle size mismatch with padding
        diffY, diffX = skip_connection.size()[2]-x.size()[2], skip_connection.size()[3]-x.size()[3]
        x = F.pad(x, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x_cat = torch.cat([skip_connection, x], dim=1)
        # Extract physics properties
        es_tuple = self.maxwell_solver(x_cat)
        out = self.conv_block1(x_cat)
        out = self.conv_block2(out)
        return out, es_tuple


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
    """
    ASPP bottleneck for multi-scale feature aggregation.
    From notebook - improves receptive field at bottleneck.
    """
    def __init__(self, in_channels, out_channels):
        super(BottleneckASPP, self).__init__()
        atrous_rates = [3, 6, 9]
        inter_channels = out_channels // (len(atrous_rates) + 2)
        
        self.convs = nn.ModuleList([
            # 1x1 conv branch
            nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU()
            ),
            # Dilated conv branches
            ASPPConv(in_channels, inter_channels, atrous_rates[0]),
            ASPPConv(in_channels, inter_channels, atrous_rates[1]),
            ASPPConv(in_channels, inter_channels, atrous_rates[2]),
            # Pooling branch
            ASPPPooling(in_channels, inter_channels)
        ])

        # Projection to output channels
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


class RobustMedVFL_UNet(nn.Module):
    """
    UNet++ architecture with physics-informed components:
    - ePURE noise estimation in encoder
    - Maxwell solver in decoder
    - ASPP bottleneck for multi-scale features
    - Deep supervision support
    
    From notebook: final-application-maxwell-for-segmentation-task (3).ipynb
    This is the exact architecture that achieved 93.96% Dice on ACDC.
    """
    def __init__(self, n_channels=1, n_classes=4, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # Feature channels at each level
        channels = [16, 32, 64, 128, 256]

        # --- Encoder (Column j=0) ---
        self.conv0_0 = EncoderBlock(n_channels, channels[0])
        self.conv1_0 = EncoderBlock(channels[0], channels[1])
        self.conv2_0 = EncoderBlock(channels[1], channels[2])
        self.conv3_0 = EncoderBlock(channels[2], channels[3])
        self.conv4_0 = BottleneckASPP(channels[3], channels[4])

        # --- Pooling & Upsampling ---
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # --- Skip connections (j > 0) ---
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
        # --- Encoder Path ---
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # --- Skip Path & Decoder ---
        # Column 1
        x0_1_input = torch.cat([x0_0, self.up(x1_0)], 1)
        x0_1 = self.conv0_1(x0_1_input)
        es_final_decoder = self.final_decoder_maxwell_solver(x0_1_input)

        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))

        # Column 2
        x0_2_input = torch.cat([x0_0, x0_1, self.up(x1_1)], 1)
        x0_2 = self.conv0_2(x0_2_input)
        es1 = self.maxwell_solver1(x0_2_input)
        
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))

        # Column 3
        x0_3_input = torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1)
        x0_3 = self.conv0_3(x0_3_input)
        es2 = self.maxwell_solver2(x0_3_input)

        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        # Column 4
        x0_4_input = torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1)
        x0_4 = self.conv0_4(x0_4_input)
        es3 = self.maxwell_solver3(x0_4_input)

        # --- Collect physics results ---
        all_es_tuples = (es1, es2, es3, es_final_decoder)
        
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4], all_es_tuples
        else:
            output = self.final(x0_4)
            return output, all_es_tuples

