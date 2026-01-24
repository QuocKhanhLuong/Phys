"""
Encoder Blocks for Ablation Study

5 encoder variants:
1. StandardEncoderBlock - Basic double conv
2. SEEncoderBlock - Squeeze-Excitation attention
3. ResNetEncoderBlock - Residual connection
4. CBAMEncoderBlock - Channel + Spatial attention
5. NAEEncoderBlock - Original with ePURE noise estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.epure import ePURE
from src.utils.helpers import adaptive_smoothing


# =============================================================================
# BASIC BUILDING BLOCKS
# =============================================================================

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


# =============================================================================
# 1. STANDARD ENCODER BLOCK
# =============================================================================

class StandardEncoderBlock(nn.Module):
    """
    Standard encoder block (like UNet++).
    Double Conv: (Conv-BN-ReLU) × 2
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block1 = BasicConvBlock(in_channels, out_channels)
        self.conv_block2 = BasicConvBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return x


# =============================================================================
# 2. SQUEEZE-EXCITATION ENCODER BLOCK
# =============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        
    def forward(self, x):
        b, c, _, _ = x.size()
        # Global average pooling
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        # Excitation
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEEncoderBlock(nn.Module):
    """
    Encoder block with Squeeze-Excitation attention.
    Standard Conv + SE block.
    """
    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()
        self.conv_block1 = BasicConvBlock(in_channels, out_channels)
        self.conv_block2 = BasicConvBlock(out_channels, out_channels)
        self.se = SEBlock(out_channels, reduction)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.se(x)
        return x


# =============================================================================
# 3. RESNET ENCODER BLOCK
# =============================================================================

class ResNetEncoderBlock(nn.Module):
    """
    ResNet BasicBlock encoder (standard implementation).
    Conv3x3-BN-ReLU → Conv3x3-BN → Add(shortcut) → ReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # First conv with BN and ReLU
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Second conv with BN only (no ReLU before residual add)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        # 1x1 conv to match channels if needed
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        out = self.relu(out)  # ReLU after addition
        return out


# =============================================================================
# 4. CBAM ENCODER BLOCK
# =============================================================================

class ChannelAttention(nn.Module):
    """Channel attention module of CBAM."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention module of CBAM."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(concat))


class CBAMBlock(nn.Module):
    """CBAM: Convolutional Block Attention Module."""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = x * self.channel_attn(x)
        x = x * self.spatial_attn(x)
        return x


class CBAMEncoderBlock(nn.Module):
    """
    Encoder block with CBAM attention.
    Standard Conv + CBAM (Channel + Spatial attention).
    """
    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()
        self.conv_block1 = BasicConvBlock(in_channels, out_channels)
        self.conv_block2 = BasicConvBlock(out_channels, out_channels)
        self.cbam = CBAMBlock(out_channels, reduction)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.cbam(x)
        return x


# =============================================================================
# VARIANTS: ResNet-18 (2 Basic Blocks per stage)
# =============================================================================

class ResNet18EncoderBlock(nn.Module):
    """
    ResNet-18 style encoder block.
    Consists of 2 BasicBlocks per stage (standard ResNet-18/34 topology).
    This is 'heavier' than the Standard/NAE block (which effectively has 1 block of 2 convs).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # First BasicBlock: adjust channels
        self.block1 = ResNetEncoderBlock(in_channels, out_channels)
        # Second BasicBlock: maintain channels (refinement)
        self.block2 = ResNetEncoderBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


# =============================================================================
# VARIANTS: SE and CBAM without reduction (full params)
# =============================================================================

class SEEncoderBlock_NoReduction(nn.Module):
    """
    SE Encoder with NO reduction (reduction=1).
    Maximum parameters - no bottleneck in attention.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block1 = BasicConvBlock(in_channels, out_channels)
        self.conv_block2 = BasicConvBlock(out_channels, out_channels)
        self.se = SEBlock(out_channels, reduction=1)  # No reduction!

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.se(x)
        return x


class CBAMEncoderBlock_NoReduction(nn.Module):
    """
    CBAM Encoder with NO reduction (reduction=1).
    Maximum parameters - no bottleneck in channel attention.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block1 = BasicConvBlock(in_channels, out_channels)
        self.conv_block2 = BasicConvBlock(out_channels, out_channels)
        self.cbam = CBAMBlock(out_channels, reduction=1)  # No reduction!

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.cbam(x)
        return x


# =============================================================================
# 5. NAE (ORIGINAL) ENCODER BLOCK
# =============================================================================

class NAEEncoderBlock(nn.Module):
    """
    Original encoder block with ePURE noise estimation.
    From RobustMedVFL_UNet - integrates physics-inspired preprocessing.
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


# =============================================================================
# 6. RESNET50 ENCODER BLOCK (Bottleneck)
# =============================================================================

class ResNet50EncoderBlock(nn.Module):
    """
    ResNet50-style bottleneck encoder block.
    1x1 → 3x3 → 1x1 with expansion factor 4.
    """
    def __init__(self, in_channels, out_channels, expansion=4):
        super().__init__()
        mid_channels = out_channels // expansion
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return self.relu(out)


# =============================================================================
# 7. EFFICIENTNET-STYLE ENCODER BLOCK (MBConv)
# =============================================================================

class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Conv (MBConv) from EfficientNet."""
    def __init__(self, in_channels, out_channels, expand_ratio=4, se_ratio=0.25):
        super().__init__()
        mid_channels = in_channels * expand_ratio
        
        # Expansion
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True)
        ) if expand_ratio > 1 else nn.Identity()
        
        # Depthwise
        self.depthwise = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True)
        )
        
        # SE block
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels, se_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(se_channels, mid_channels, 1),
            nn.Sigmoid()
        )
        
        # Projection
        self.project = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.use_residual = (in_channels == out_channels)

    def forward(self, x):
        identity = x
        out = self.expand(x)
        out = self.depthwise(out)
        out = out * self.se(out)
        out = self.project(out)
        if self.use_residual:
            out = out + identity
        return out


class EfficientNetEncoderBlock(nn.Module):
    """EfficientNet-style encoder with 2 MBConv blocks."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block1 = MBConvBlock(in_channels, out_channels)
        self.block2 = MBConvBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


# =============================================================================
# 8. SWIN TRANSFORMER ENCODER BLOCK
# =============================================================================

class WindowAttention(nn.Module):
    """Window-based multi-head self attention (Swin Transformer)."""
    def __init__(self, dim, window_size=7, num_heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, H, W, C = x.shape
        
        # Pad to multiple of window_size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        _, Hp, Wp, _ = x.shape
        
        # Partition windows
        x = x.view(B, Hp // self.window_size, self.window_size, 
                   Wp // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, self.window_size * self.window_size, C)
        
        # Attention
        qkv = self.qkv(windows).reshape(-1, self.window_size * self.window_size, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(-1, self.window_size * self.window_size, C)
        out = self.proj(out)
        
        # Reverse windows
        out = out.view(B, Hp // self.window_size, Wp // self.window_size, 
                       self.window_size, self.window_size, C)
        out = out.permute(0, 1, 3, 2, 4, 5).contiguous()
        out = out.view(B, Hp, Wp, C)
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :H, :W, :]
        
        return out


class SwinEncoderBlock(nn.Module):
    """Swin Transformer encoder block."""
    def __init__(self, in_channels, out_channels, num_heads=4, window_size=7):
        super().__init__()
        # Project to output channels
        self.proj_in = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        self.norm1 = nn.LayerNorm(out_channels)
        self.attn = WindowAttention(out_channels, window_size, num_heads)
        self.norm2 = nn.LayerNorm(out_channels)
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.GELU(),
            nn.Linear(out_channels * 4, out_channels)
        )

    def forward(self, x):
        x = self.proj_in(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        
        # Attention block
        x = x + self.attn(self.norm1(x))
        # MLP block
        x = x + self.mlp(self.norm2(x))
        
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        return x


# =============================================================================
# 9. CONVNEXT ENCODER BLOCK
# =============================================================================

class ConvNeXtEncoderBlock(nn.Module):
    """
    ConvNeXt-style encoder block.
    Depthwise conv → LayerNorm → 1x1 → GELU → 1x1
    """
    def __init__(self, in_channels, out_channels, expansion=4):
        super().__init__()
        mid_channels = out_channels * expansion
        
        self.proj_in = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        self.dwconv = nn.Conv2d(out_channels, out_channels, 7, padding=3, groups=out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.pwconv1 = nn.Linear(out_channels, mid_channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(mid_channels, out_channels)

    def forward(self, x):
        x = self.proj_in(x)
        identity = x
        
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        
        return x + identity


# =============================================================================
# 10. DENSENET ENCODER BLOCK
# =============================================================================

class DenseLayer(nn.Module):
    """Single dense layer with bottleneck."""
    def __init__(self, in_channels, growth_rate=32):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, 3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], dim=1)


class DenseNetEncoderBlock(nn.Module):
    """DenseNet-style encoder with dense connections."""
    def __init__(self, in_channels, out_channels, num_layers=4, growth_rate=16):
        super().__init__()
        
        layers = []
        current_channels = in_channels
        for i in range(num_layers):
            layers.append(DenseLayer(current_channels, growth_rate))
            current_channels += growth_rate
        self.dense_layers = nn.Sequential(*layers)
        
        # Transition to output channels
        self.transition = nn.Sequential(
            nn.BatchNorm2d(current_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(current_channels, out_channels, 1, bias=False)
        )

    def forward(self, x):
        x = self.dense_layers(x)
        x = self.transition(x)
        return x


# =============================================================================
# 11. SAM (Segment Anything) ENCODER BLOCK
# =============================================================================

class SAMEncoderBlock(nn.Module):
    """
    SAM-inspired encoder block with ViT-style attention.
    Simplified version for per-block usage.
    """
    def __init__(self, in_channels, out_channels, num_heads=4, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, out_channels, patch_size, stride=patch_size)
        
        # Transformer block
        self.norm1 = nn.LayerNorm(out_channels)
        self.attn = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(out_channels)
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.GELU(),
            nn.Linear(out_channels * 4, out_channels)
        )
        
        # Upsample back
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # B, C, H/p, W/p
        _, _, Hp, Wp = x.shape
        
        # Reshape for attention
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        
        # Transformer
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        
        # Reshape back
        x = x.transpose(1, 2).view(B, -1, Hp, Wp)
        
        # Upsample
        x = self.upsample(x)
        
        # Adjust size if needed
        if x.shape[2] != H or x.shape[3] != W:
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        return x


# =============================================================================
# ENCODER FACTORY
# =============================================================================

def get_encoder_block(encoder_type: str, in_channels: int, out_channels: int):
    """
    Factory function to get encoder block by type.
    
    Args:
        encoder_type: One of 'standard', 'se', 'resnet', 'cbam', 'nae',
                      'resnet50', 'efficientnet', 'swin', 'convnext', 'densenet', 'sam'
        in_channels: Input channels
        out_channels: Output channels
    
    Returns:
        Appropriate encoder block module
    """
    encoder_map = {
        'standard': StandardEncoderBlock,
        'se': SEEncoderBlock,
        'se_noreduction': SEEncoderBlock_NoReduction,
        'resnet': ResNetEncoderBlock,
        'resnet18': ResNet18EncoderBlock,
        'cbam': CBAMEncoderBlock,
        'cbam_noreduction': CBAMEncoderBlock_NoReduction,
        'nae': NAEEncoderBlock,
        # New encoders
        'resnet50': ResNet50EncoderBlock,
        'efficientnet': EfficientNetEncoderBlock,
        'swin': SwinEncoderBlock,
        'convnext': ConvNeXtEncoderBlock,
        'densenet': DenseNetEncoderBlock,
        'sam': SAMEncoderBlock,
    }
    
    if encoder_type.lower() not in encoder_map:
        raise ValueError(f"Unknown encoder type: {encoder_type}. "
                        f"Available: {list(encoder_map.keys())}")
    
    return encoder_map[encoder_type.lower()](in_channels, out_channels)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing encoder blocks...")
    
    x = torch.randn(2, 5, 224, 224)  # Batch=2, Channels=5
    
    for name, EncoderClass in [
        ("Standard", StandardEncoderBlock),
        ("SE", SEEncoderBlock),
        ("ResNet", ResNetEncoderBlock),
        ("CBAM", CBAMEncoderBlock),
        ("NAE", NAEEncoderBlock),
        ("ResNet50", ResNet50EncoderBlock),
        ("EfficientNet", EfficientNetEncoderBlock),
        ("Swin", SwinEncoderBlock),
        ("ConvNeXt", ConvNeXtEncoderBlock),
        ("DenseNet", DenseNetEncoderBlock),
        ("SAM", SAMEncoderBlock),
    ]:
        encoder = EncoderClass(5, 16)
        out = encoder(x)
        params = sum(p.numel() for p in encoder.parameters())
        print(f"{name:15} | Input: {x.shape} -> Output: {out.shape} | Params: {params:,}")
    
    print("\nAll encoder blocks working correctly!")

