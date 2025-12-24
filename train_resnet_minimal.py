"""Minimal ResNet+improvements training script - simplified for debugging"""
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import stats
import time

# Import model
from models_resnet_improved import ResNetImproved

# Import existing function
from train_resnet_improved import get_koniq_train_test_indices as get_koniq_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_multiscale', action='store_true')
    parser.add_argument('--use_attention', action='store_true')
    config = parser.parse_args()
    
    print("=" * 80)
    print("Minimal ResNet Training Test")
    print("=" * 80)
    
    # Get indices
    print("\n1. Loading indices...")
    train_idx, test_idx = get_koniq_indices()
    print(f"   Train: {len(train_idx)}, Test: {len(test_idx)}")
    
    # Create model
    print("\n2. Creating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetImproved(
        lda_out_channels=128,
        hyper_in_channels=2048,
        target_in_size=7 * 7,
        target_fc1_size=112,
        target_fc2_size=56,
        target_fc3_size=28,
        target_fc4_size=14,
        feature_size=7 * 7,
        use_multiscale=config.use_multiscale,
        use_attention=config.use_attention,
        dropout_rate=0.3
    ).to(device)
    print(f"   Model created: Multi-scale={config.use_multiscale}, Attention={config.use_attention}")
    
    # Create optimizer
    print("\n3. Creating optimizer...")
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    criterion = nn.L1Loss()
    print("   Optimizer and loss function ready")
    
    # Import data loader last (this is where it might hang)
    print("\n4. Creating data loaders (THIS IS WHERE IT MIGHT HANG)...")
    sys.stdout.flush()  # Force output before potential hang
    
    import data_loader
    
    print("   4a. Creating TRAIN loader...")
    sys.stdout.flush()
    train_loader_wrapper = data_loader.DataLoader(
        dataset='koniq-10k',
        path='./koniq-10k',
        img_indx=train_idx,
        patch_size=224,
        patch_num=25,
        batch_size=1,
        istrain=True,
        use_color_jitter=False,
        preload=False  # Disable preload for minimal test
    )
    print("   4b. Getting TRAIN DataLoader object...")
    sys.stdout.flush()
    train_loader = train_loader_wrapper.get_data()
    print(f"   ✓ Train loader ready: {len(train_loader)} batches")
    
    print("   4c. Creating TEST loader...")
    sys.stdout.flush()
    test_loader_wrapper = data_loader.DataLoader(
        dataset='koniq-10k',
        path='./koniq-10k',
        img_indx=test_idx,
        patch_size=224,
        patch_num=25,
        istrain=False,
        test_random_crop=True,
        preload=False  # Disable preload for minimal test
    )
    print("   4d. Getting TEST DataLoader object...")
    sys.stdout.flush()
    test_loader = test_loader_wrapper.get_data()
    print(f"   ✓ Test loader ready: {len(test_loader)} batches")
    
    print("\n5. ALL INITIALIZATION COMPLETE!")
    print("=" * 80)
    print("\nStarting training loop...\n")
    
    # Simple training loop (just 1 batch to test)
    model.train()
    print("Epoch 1/1")
    for i, (patches, label) in enumerate(train_loader):
        patches = patches.to(device)
        label = label.to(device).float()
        
        # Forward
        out = model(patches)
        pred = out['score']
        
        # Loss
        loss = criterion(pred, label)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  Batch {i+1}/{len(train_loader)}: Loss={loss.item():.4f}")
        
        if i >= 2:  # Only do 3 batches for testing
            break
    
    print("\n✓ Training test completed successfully!")

if __name__ == '__main__':
    main()

