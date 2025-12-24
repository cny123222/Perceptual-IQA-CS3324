"""Test script to verify data loader output shapes"""
import torch
import data_loader
import numpy as np

print("Testing data loader...")

# Minimal config
dataset = 'koniq-10k'
path = './koniq-10k'
patch_size = 224
train_patch_num = 5  # Small number for quick test
test_patch_num = 5

# Get a small subset of indices for testing
train_idx = np.array([0, 1, 2])
test_idx = np.array([0, 1])

print("\nCreating train loader (NO preload)...")
train_loader_wrapper = data_loader.DataLoader(
    dataset=dataset,
    path=path,
    img_indx=train_idx,
    patch_size=patch_size,
    patch_num=train_patch_num,
    batch_size=1,
    istrain=True,
    use_color_jitter=False,
    preload=False  # Test without preload first
)
train_loader = train_loader_wrapper.get_data()

print("\nTesting train loader output...")
for i, (patches, label) in enumerate(train_loader):
    print(f"  Batch {i}:")
    print(f"    patches shape: {patches.shape}")
    print(f"    label shape: {label.shape}")
    print(f"    patches dtype: {patches.dtype}")
    print(f"    patches min/max: {patches.min():.3f} / {patches.max():.3f}")
    
    # Check individual patches
    patches_squeezed = patches.squeeze(0)  # Remove batch dim
    print(f"    After squeeze(0): {patches_squeezed.shape}")
    
    for j, patch in enumerate(patches_squeezed[:2]):  # Check first 2 patches
        print(f"      Patch {j} shape: {patch.shape}")
        if patch.shape[0] != 3:
            print(f"        ❌ ERROR: Expected 3 channels, got {patch.shape[0]}")
        else:
            print(f"        ✓ OK: 3 channels")
    
    if i >= 2:  # Only test 3 batches
        break

print("\n✓ Test complete!")

