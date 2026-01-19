import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Một khối tích chập cơ bản: Conv -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):
    """Khối Squeeze-and-Excitation cho Channel Attention"""
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ePURE(nn.Module):
    """
    Phiên bản ePURE hoàn chỉnh nhất với:
    - Mạng sâu hơn (Deeper).
    - Lớp BatchNorm2d (Normalization).
    - Kết nối tắt (Residual Connection).
    - Cơ chế chú ý (Attention).
    """
    def __init__(self, in_channels, base_channels=32):
        super().__init__()
        # Các khối tích chập
        self.block1 = ConvBlock(in_channels, base_channels)
        self.block2 = ConvBlock(base_channels, base_channels)
        
        # THÊM MỚI: Khối Attention
        self.attention = SEBlock(channels=base_channels)
        
        self.block3 = ConvBlock(base_channels, base_channels)
        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        x_float = x.float()

        # Luồng dữ liệu qua các khối
        out_block1 = self.block1(x_float)
        out_block2 = self.block2(out_block1)
        
        # Áp dụng kết nối tắt
        residual_out = out_block2 + out_block1
        
        # Áp dụng Attention
        attention_out = self.attention(residual_out)
        
        # Đi qua khối cuối cùng
        out_block3 = self.block3(attention_out)
        
        # Tạo bản đồ nhiễu cuối cùng
        noise_map = self.final_conv(out_block3)
        
        return noise_map

