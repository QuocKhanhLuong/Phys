# PIE-UNet Full Ablation Study Results

Generated: 2026-01-05 17:10:19

Training epochs: 250 (with early stopping)

## Results (Evaluated on TEST SET)

| Profile | C_in | Depth | #Params | G-MACs | GFLOPs | Peak GPU | Test DICE | Test HD95 |
|---------|------|-------|---------|--------|--------|----------|-----------|----------|
| PIE-UNet-T | 3 | 4 | 0.47M | 7.420 | 14.840 | 4685MB | 0.9153 | 2.71 |
| PIE-UNet-S | 5 | 4 | 0.47M | 7.463 | 14.927 | 4743MB | 0.9136 | 2.60 |
| PIE-UNet-M | 5 | 5 | 1.60M | 11.207 | 22.414 | 5691MB | 0.9171 | 2.33 |
| PIE-UNet-L | 7 | 5 | 1.60M | 11.250 | 22.501 | 5598MB | 0.9189 | 2.37 |
| PIE-UNet-XL | 7 | 6 | 5.95M | 16.016 | 32.032 | 8861MB | 0.9120 | 2.55 |
