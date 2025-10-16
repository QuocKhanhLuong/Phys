import sys
import torch

print("Testing imports...")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    from models import ePURE, MaxwellSolver, RobustMedVFL_UNet, print_model_parameters
    print("✓ models.py imported successfully")
except Exception as e:
    print(f"✗ models.py import failed: {e}")
    sys.exit(1)

try:
    from losses import FocalLoss, FocalTverskyLoss, PhysicsLoss, CombinedLoss
    print("✓ losses.py imported successfully")
except Exception as e:
    print(f"✗ losses.py import failed: {e}")
    sys.exit(1)

try:
    from utils import adaptive_smoothing, adaptive_quantum_noise_injection, AdvancedB1Simulator, calculate_ultimate_common_b1_map
    print("✓ utils.py imported successfully")
except Exception as e:
    print(f"✗ utils.py import failed: {e}")
    sys.exit(1)

try:
    from data_utils import BraTS21Dataset25D, load_brats21_volumes
    print("✓ data_utils.py imported successfully")
except Exception as e:
    print(f"✗ data_utils.py import failed: {e}")
    sys.exit(1)

try:
    from evaluate import evaluate_metrics, evaluate_metrics_with_tta, run_and_print_test_evaluation, visualize_final_results_2_5D
    print("✓ evaluate.py imported successfully")
except Exception as e:
    print(f"✗ evaluate.py import failed: {e}")
    sys.exit(1)

try:
    import config
    print("✓ config.py imported successfully")
except Exception as e:
    print(f"✗ config.py import failed: {e}")
    sys.exit(1)

print("\n--- Testing Model Initialization ---")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RobustMedVFL_UNet(n_channels=20, n_classes=4)
    model.eval()  # Set to eval mode to avoid BatchNorm issues with small tensors
    print(f"✓ Model initialized successfully")
    print(f"  Input: 20 channels (4 modalities × 5 slices)")
    print(f"  Output: 4 classes")
    
    dummy_input = torch.randn(1, 20, 224, 224)
    with torch.no_grad():
        outputs, physics_outputs = model(dummy_input)
    print(f"✓ Forward pass successful")
    print(f"  Deep supervision outputs: {len(outputs)}")
    print(f"  Physics outputs: {len(physics_outputs)}")
    
except Exception as e:
    print(f"✗ Model test failed: {e}")
    sys.exit(1)

print("\n--- Testing Loss Functions ---")
try:
    criterion = CombinedLoss(num_classes=4, initial_loss_weights=[0.5, 0.4, 0.1])
    print(f"✓ CombinedLoss initialized")
    print(f"  Components: FocalLoss, FocalTverskyLoss, PhysicsLoss")
    
    dummy_logits = torch.randn(1, 4, 224, 224)
    dummy_targets = torch.randint(0, 4, (1, 224, 224))
    dummy_b1 = torch.randn(1, 1, 224, 224)
    
    loss = criterion(dummy_logits, dummy_targets, dummy_b1, physics_outputs)
    print(f"✓ Loss computation successful: {loss.item():.4f}")
    
except Exception as e:
    print(f"✗ Loss test failed: {e}")
    sys.exit(1)

print("\n✓ All tests passed!")
print("\nProject is ready for training.")
print("Next steps:")
print("  1. Extract dataset: ./extract_dataset.sh")
print("  2. Train model: python train.py")

