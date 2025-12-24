"""Quick test of train_resnet_improved.py to verify it runs without errors"""
import sys
import argparse

# Import the main function
sys.path.insert(0, '/root/Perceptual-IQA-CS3324')
from train_resnet_improved import main

# Create minimal config for testing
parser = argparse.ArgumentParser()
parser.add_argument('--use_multiscale', action='store_true', default=True)
parser.add_argument('--use_attention', action='store_true', default=True)
parser.add_argument('--dropout_rate', type=float, default=0.3)
parser.add_argument('--dataset', type=str, default='koniq-10k')
parser.add_argument('--data_path', type=str, default='./koniq-10k')
parser.add_argument('--epochs', type=int, default=1)  # Just 1 epoch for testing
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=96)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--patch_size', type=int, default=224)
parser.add_argument('--train_patch_num', type=int, default=5)  # Small number for quick test
parser.add_argument('--test_patch_num', type=int, default=5)
parser.add_argument('--no_color_jitter', action='store_true', default=True)
parser.add_argument('--test_random_crop', action='store_true', default=True)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--log_dir', type=str, default='./logs_test')
parser.add_argument('--exp_name', type=str, default='quick_test')
parser.add_argument('--save_model', action='store_true', default=False)
parser.add_argument('--preload_images', action='store_true', default=False)  # No preload for quick test

config = parser.parse_args([])
config.use_color_jitter = not config.no_color_jitter

print("=" * 80)
print("QUICK TEST: Running train_resnet_improved.py for 1 epoch")
print("=" * 80)

try:
    main(config)
    print("\n" + "=" * 80)
    print("✓ TEST PASSED: Training completed without errors!")
    print("=" * 80)
except Exception as e:
    print("\n" + "=" * 80)
    print(f"✗ TEST FAILED: {str(e)}")
    print("=" * 80)
    import traceback
    traceback.print_exc()
    sys.exit(1)

