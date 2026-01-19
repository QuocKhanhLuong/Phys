import torch
from thop import profile
import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.unet import RobustMedVFL_UNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slices', type=int, default=5, help='Number of input slices (channels)')
    parser.add_argument('--classes', type=int, default=4, help='Number of classes')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Instantiate Our Model
    print(f"Instantiating RobustMedVFL_UNet with n_channels={args.slices}, n_classes={args.classes}...")
    model = RobustMedVFL_UNet(n_channels=args.slices, n_classes=args.classes, deep_supervision=True).to(device)
    model.eval()

    # Create dummy input
    # Shape: (Batch, Channels/Slices, Height, Width)
    input_tensor = torch.randn(1, args.slices, args.img_size, args.img_size).to(device)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Measure
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)

    print("="*40)
    print(f"RobustMedVFL_UNet Stats (Input {args.slices}x{args.img_size}x{args.img_size})")
    print(f"FLOPs: {flops / 1e9:.4f} G")
    print(f"Params: {params / 1e6:.4f} M")
    print("="*40)

if __name__ == "__main__":
    main()
