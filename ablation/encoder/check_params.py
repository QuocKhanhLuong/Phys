
import torch
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import configs securely
try:
    from ablation.encoder.config import ENCODER_CONFIGS
    from ablation.encoder.pie_unet_encoder import PIE_UNet_Encoder
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Encoders to compare
TARGET_ENCODERS = ["Standard", "ResNet", "ResNet18", "ConvNeXt"]

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\n" + "=" * 80)
print(f"{'Encoder Key':<20} | {'Type':<15} | {'Total Params':<15} | {'vs ResNet (1-Blk)':<20}")
print("=" * 80)

# 1. Calculate Baseline (ResNet 1-Block)
try:
    lbl_resnet = "ResNet"
    type_resnet = ENCODER_CONFIGS[lbl_resnet]["type"]
    model_resnet = PIE_UNet_Encoder(n_channels=5, n_classes=4, encoder_type=type_resnet, deep_supervision=True)
    params_resnet = count_params(model_resnet)
except Exception as e:
    print(f"Error creating baseline ResNet: {e}")
    sys.exit(1)

# 2. Iterate and Compare
for key in TARGET_ENCODERS:
    if key not in ENCODER_CONFIGS:
        print(f"Warning: {key} not found in ENCODER_CONFIGS")
        continue
        
    enc_conf = ENCODER_CONFIGS[key]
    enc_type = enc_conf["type"]
    enc_name = enc_conf["name"]
    
    try:
        model = PIE_UNet_Encoder(n_channels=5, n_classes=4, encoder_type=enc_type, deep_supervision=True)
        total = count_params(model)
        diff = total - params_resnet
        
        diff_str = f"{diff:+15,}" if diff != 0 else "Baseline"
        print(f"{key:<20} | {enc_type:<15} | {total:<15,} | {diff_str:<20}")
        
    except Exception as e:
        print(f"Error checking {key}: {e}")

print("-" * 80)
print("Done.")
