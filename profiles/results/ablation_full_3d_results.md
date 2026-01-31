# PIE-UNet Ablation - FULL Evaluation Results

Generated: 2026-01-17 09:24:04

## Computational Metrics + 3D Volumetric Accuracy

| Profile | C_in | Depth | Params | G-MACs | GFLOPs | CPU Latency | Peak GPU | Dice (3D) | HD95 (3D) |
|---------|------|-------|--------|--------|--------|-------------|----------|-----------|----------|
| PIE-UNet-T | 3 | 4 | 466,687 | 7.4201 | 14.8402 | 50.20±1.34ms | 944MB | 0.9080±0.0556 | 1.2508±1.5183 |
| PIE-UNet-M | 5 | 5 | 1,597,706 | 11.2069 | 22.4138 | 71.08±0.46ms | 1119MB | 0.9126±0.0488 | 1.1289±0.4345 |
| PIE-UNet-XL | 7 | 6 | 5,953,743 | 16.0160 | 32.0320 | 97.11±0.44ms | 1313MB | 0.9029±0.0732 | 1.1203±0.8622 |
