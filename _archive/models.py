import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import adaptive_smoothing


class ConvBlock(nn.Module):
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
    def __init__(self, in_channels, base_channels=32):
        super().__init__()
        self.block1 = ConvBlock(in_channels, base_channels)
        self.block2 = ConvBlock(base_channels, base_channels)
        self.attention = SEBlock(channels=base_channels)
        self.block3 = ConvBlock(base_channels, base_channels)
        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        x_float = x.float()
        out_block1 = self.block1(x_float)
        out_block2 = self.block2(out_block1)
        residual_out = out_block2 + out_block1
        attention_out = self.attention(residual_out)
        out_block3 = self.block3(attention_out)
        noise_map = self.final_conv(out_block3)
        return noise_map


class MaxwellSolver(nn.Module):
    def __init__(self, in_channels, hidden_dim=32):
        super(MaxwellSolver, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1)
        )

    def forward(self, x):
        eps_sigma_map = self.encoder(x)
        return eps_sigma_map[:, 0:1, :, :], eps_sigma_map[:, 1:2, :, :]


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class ASPPPooling(nn.Sequential):
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


class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
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


class PIE_UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=4, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        channels = [16, 32, 64, 128, 256]

        self.conv0_0 = EncoderBlock(n_channels, channels[0])
        self.conv1_0 = EncoderBlock(channels[0], channels[1])
        self.conv2_0 = EncoderBlock(channels[1], channels[2])
        self.conv3_0 = EncoderBlock(channels[2], channels[3])
        self.conv4_0 = BottleneckASPP(channels[3], channels[4])

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_1 = BasicConvBlock(channels[0] + channels[1], channels[0])
        self.conv1_1 = BasicConvBlock(channels[1] + channels[2], channels[1])
        self.conv2_1 = BasicConvBlock(channels[2] + channels[3], channels[2])
        self.conv3_1 = BasicConvBlock(channels[3] + channels[4], channels[3])

        self.conv0_2 = BasicConvBlock(channels[0]*2 + channels[1], channels[0])
        self.conv1_2 = BasicConvBlock(channels[1]*2 + channels[2], channels[1])
        self.conv2_2 = BasicConvBlock(channels[2]*2 + channels[3], channels[2])

        self.conv0_3 = BasicConvBlock(channels[0]*3 + channels[1], channels[0])
        self.conv1_3 = BasicConvBlock(channels[1]*3 + channels[2], channels[1])

        self.conv0_4 = BasicConvBlock(channels[0]*4 + channels[1], channels[0])

        self.maxwell_solver1 = MaxwellSolver(channels[0]*2 + channels[1])
        self.maxwell_solver2 = MaxwellSolver(channels[0]*3 + channels[1])
        self.maxwell_solver3 = MaxwellSolver(channels[0]*4 + channels[1])
        self.final_decoder_maxwell_solver = MaxwellSolver(channels[0] + channels[1]) 

        if self.deep_supervision:
            self.final1 = nn.Conv2d(channels[0], n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(channels[0], n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(channels[0], n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(channels[0], n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(channels[0], n_classes, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1_input = torch.cat([x0_0, self.up(x1_0)], 1)
        x0_1 = self.conv0_1(x0_1_input)
        es_final_decoder = self.final_decoder_maxwell_solver(x0_1_input)

        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))

        x0_2_input = torch.cat([x0_0, x0_1, self.up(x1_1)], 1)
        x0_2 = self.conv0_2(x0_2_input)
        es1 = self.maxwell_solver1(x0_2_input)
        
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))

        x0_3_input = torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1)
        x0_3 = self.conv0_3(x0_3_input)
        es2 = self.maxwell_solver2(x0_3_input)

        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        x0_4_input = torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1)
        x0_4 = self.conv0_4(x0_4_input)
        es3 = self.maxwell_solver3(x0_4_input)

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


def print_model_parameters(model):
    print("="*60)
    print("PHÂN TÍCH THAM SỐ MÔ HÌNH")
    print("="*60)
    total_params = 0
    for name, module in model.named_children():
        if list(module.parameters()):
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"- {name:<30}: {params:>12,}")
            total_params += params
    print("="*60)
    print(f"TỔNG CỘNG                      : {total_params:>12,}")
    print("="*60)
    direct_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Xác minh (tổng trực tiếp)       : {direct_total:>12,}")
    print("="*60)

