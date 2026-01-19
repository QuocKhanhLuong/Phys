import torch
import sys
import os
from thop import profile, clever_format

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.unet import RobustMedVFL_UNet

def measure_flops():
    NUM_SLICES = 5
    NUM_CLASSES = 4
    IMG_SIZE = 224
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {DEVICE}")
    print(f"Input configuration: {NUM_SLICES} slices, image size {IMG_SIZE}x{IMG_SIZE}")

    # Initialize model
    model = RobustMedVFL_UNet(n_channels=NUM_SLICES, n_classes=NUM_CLASSES, deep_supervision=True).to(DEVICE)
    model.eval()

    # Create dummy input
    # Shape: (Batch_Size, Channels, Height, Width)
    input_tensor = torch.randn(1, NUM_SLICES, IMG_SIZE, IMG_SIZE).to(DEVICE)

    print("Measuring GFLOPs and Params...")
    
    # Measure
    # custom_ops may be needed for some specific layers, but usually thop handles standard UNet layers well
    macs, params = profile(model, inputs=(input_tensor, ), verbose=False)

    # Convert to GFLOPs (1 MAC = 2 FLOPs typically)
    # thop returns MACs on the scale of raw numbers
    
    formatted_macs, formatted_params = clever_format([macs, params], "%.3f")
    
    g_macs = macs / 1e9
    g_flops = g_macs * 2
    
    print("\n" + "="*40)
    print(f"Model: RobustMedVFL_UNet")
    print(f"{'='*40}")
    print(f"Params:     {formatted_params}")
    print(f"MACs:       {formatted_macs} (Multiply-Accumulate Operations)")
    print(f"G-MACs:     {g_macs:.3f} G")
    print(f"GFLOPs:     {g_flops:.3f} G (assuming 1 MAC = 2 FLOPs)")
    print(f"{'='*40}")

if __name__ == "__main__":
    measure_flops()
