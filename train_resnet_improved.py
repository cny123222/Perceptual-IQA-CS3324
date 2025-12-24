"""
Training script for ResNet50 + Improvements
验证Multi-scale和Attention是否对ResNet50也有效
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from scipy import stats
import time
import os
import argparse
from models_resnet_improved import ResNetImproved
import data_loader


class ResNetImprovedSolver:
    """Solver for training and testing ResNetImproved"""
    
    def __init__(self, config, train_idx, test_idx):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        print(f"\n{'='*80}")
        print(f"Creating ResNetImproved model")
        print(f"  use_multiscale: {config.use_multiscale}")
        print(f"  use_attention: {config.use_attention}")
        print(f"{'='*80}\n")
        
        self.model = ResNetImproved(
            use_multiscale=config.use_multiscale,
            use_attention=config.use_attention,
            dropout_rate=config.dropout_rate
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params / 1e6:.2f}M\n")
        
        # Data loaders with preloading
        print("Initializing data loaders...")
        if config.preload_images:
            print("⚡ Image preloading ENABLED - loading images into memory...")
        else:
            print("💾 Image preloading DISABLED - loading on-the-fly")
        
        # Our data_loader.DataLoader returns a PyTorch DataLoader via .get_data()
        train_data_wrapper = data_loader.DataLoader(
            config.dataset, 
            config.data_path, 
            train_idx, 
            config.patch_size, 
            config.train_patch_num,
            batch_size=config.batch_size,
            istrain=True,
            use_color_jitter=config.use_color_jitter,
            preload=config.preload_images
        )
        self.train_loader = train_data_wrapper.get_data()
        
        test_data_wrapper = data_loader.DataLoader(
            config.dataset,
            config.data_path,
            test_idx,
            config.patch_size,
            config.test_patch_num,
            istrain=False,
            test_random_crop=config.test_random_crop,
            preload=config.preload_images
        )
        self.test_loader = test_data_wrapper.get_data()
        
        if config.preload_images:
            print("✓ All images loaded into memory!\n")
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # Loss function
        self.criterion = nn.L1Loss()
        
        # Training settings
        self.epochs = config.epochs
        self.best_srcc = 0
        self.best_plcc = 0
        
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        losses = []
        
        for i, (patches, label) in enumerate(self.train_loader):
            # patches: (1, N, 3, H, W), label: (1,)
            patches = patches.squeeze(0).to(self.device)  # (N, 3, H, W)
            label = label.to(self.device).float()
            
            # Forward pass for all patches
            outputs = []
            for patch in patches:
                out = self.model(patch.unsqueeze(0))
                outputs.append(out['score'])
            
            # Average prediction across patches
            pred = torch.mean(torch.stack(outputs))
            
            # Compute loss
            loss = self.criterion(pred, label)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            
            if (i + 1) % 100 == 0:
                print(f"  Batch [{i+1}/{len(self.train_loader)}], Loss: {loss.item():.4f}")
        
        avg_loss = np.mean(losses)
        return avg_loss
    
    def test(self):
        """Test on test set"""
        self.model.eval()
        predictions = []
        labels = []
        
        with torch.no_grad():
            for patches, label in self.test_loader:
                patches = patches.squeeze(0).to(self.device)
                
                # Forward pass for all patches
                outputs = []
                for patch in patches:
                    out = self.model(patch.unsqueeze(0))
                    outputs.append(out['score'])
                
                # Average prediction
                pred = torch.mean(torch.stack(outputs))
                
                predictions.append(pred.cpu().item())
                labels.append(label.item())
        
        # Compute SRCC and PLCC
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        srcc = stats.spearmanr(predictions, labels)[0]
        plcc = stats.pearsonr(predictions, labels)[0]
        
        return srcc, plcc
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*80}")
        print(f"Starting Training")
        print(f"{'='*80}")
        print(f"Configuration:")
        print(f"  Dataset: {self.config.dataset}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Learning Rate: {self.config.lr}")
        print(f"  Batch Size: {self.config.batch_size}")
        print(f"  Train Patches: {self.config.train_patch_num}")
        print(f"  Test Patches: {self.config.test_patch_num}")
        print(f"  ColorJitter: {self.config.use_color_jitter}")
        print(f"  Test Random Crop: {self.config.test_random_crop}")
        print(f"  Dropout: {self.config.dropout_rate}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print("-" * 80)
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Test
            test_srcc, test_plcc = self.test()
            
            # Update best
            if test_srcc > self.best_srcc:
                self.best_srcc = test_srcc
                self.best_plcc = test_plcc
                
                # Save model
                if self.config.save_model:
                    save_path = f"checkpoints/resnet_improved_{'ms' if self.config.use_multiscale else 'ss'}_{'att' if self.config.use_attention else 'noatt'}_best.pth"
                    os.makedirs("checkpoints", exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'srcc': test_srcc,
                        'plcc': test_plcc,
                    }, save_path)
                    print(f"✓ Model saved to {save_path}")
            
            epoch_time = time.time() - epoch_start
            
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Test SRCC: {test_srcc:.4f}")
            print(f"  Test PLCC: {test_plcc:.4f}")
            print(f"  Best SRCC: {self.best_srcc:.4f}")
            print(f"  Best PLCC: {self.best_plcc:.4f}")
            print(f"  Time: {epoch_time:.1f}s")
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"{'='*80}")
        print(f"Best Test SRCC: {self.best_srcc:.4f}")
        print(f"Best Test PLCC: {self.best_plcc:.4f}")
        print(f"Total Time: {total_time / 3600:.2f} hours")
        print(f"{'='*80}\n")
        
        return self.best_srcc, self.best_plcc


def get_koniq_train_test_indices(root_path):
    """Get train and test indices for KonIQ-10k based on official split"""
    import csv
    import json
    
    csv_file = os.path.join(root_path, 'koniq10k_scores_and_distributions.csv')
    train_json = os.path.join(root_path, 'koniq_train.json')
    test_json = os.path.join(root_path, 'koniq_test.json')
    
    # Read CSV to get all image names
    csv_images = []
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_images.append(row['image_name'])
    
    # Read official train/test split from JSON
    train_images = set()
    test_images = set()
    
    if os.path.exists(train_json):
        with open(train_json) as f:
            train_data = json.load(f)
            for item in train_data:
                train_images.add(os.path.basename(item['image']))
    
    if os.path.exists(test_json):
        with open(test_json) as f:
            test_data = json.load(f)
            for item in test_data:
                test_images.add(os.path.basename(item['image']))
    
    # Get indices for train and test images from CSV
    train_indices = []
    test_indices = []
    
    for idx, img_name in enumerate(csv_images):
        if img_name in train_images:
            train_indices.append(idx)
        elif img_name in test_images:
            test_indices.append(idx)
    
    return train_indices, test_indices


def main(config):
    """Main function"""
    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Load dataset split
    print(f"Loading dataset: {config.dataset}")
    
    if config.dataset == 'koniq-10k':
        # Try different path locations
        possible_paths = [
            config.data_path,
            'koniq-10k',
            './koniq-10k',
            '../koniq-10k',
            '/root/Perceptual-IQA-CS3324/koniq-10k'
        ]
        
        dataset_path = None
        for path in possible_paths:
            test_file = os.path.join(path, 'koniq10k_scores_and_distributions.csv')
            if os.path.exists(test_file):
                dataset_path = path
                config.data_path = path  # Update config with correct path
                print(f"✓ Found dataset at: {path}")
                break
        
        if dataset_path is None:
            raise FileNotFoundError(
                f"Could not find koniq-10k dataset in any of:\n" +
                "\n".join([f"  - {p}" for p in possible_paths])
            )
        
        train_idx, test_idx = get_koniq_train_test_indices(dataset_path)
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        print(f"Train images: {len(train_idx)}")
        print(f"Test images: {len(test_idx)}")
    else:
        raise NotImplementedError(f"Dataset {config.dataset} not supported")
    
    # Create solver and train
    solver = ResNetImprovedSolver(config, train_idx, test_idx)
    best_srcc, best_plcc = solver.train()
    
    return best_srcc, best_plcc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet50 + Improvements Training')
    
    # Model settings
    parser.add_argument('--use_multiscale', action='store_true', 
                       help='Use multi-scale feature fusion')
    parser.add_argument('--use_attention', action='store_true',
                       help='Use channel attention mechanism')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate (default: 0.3)')
    
    # Training settings
    parser.add_argument('--dataset', type=str, default='koniq-10k',
                       help='Dataset name')
    parser.add_argument('--data_path', type=str, default='./koniq-10k',
                       help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4 for ResNet)')
    parser.add_argument('--batch_size', type=int, default=96,
                       help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    
    # Data settings
    parser.add_argument('--patch_size', type=int, default=224,
                       help='Patch size')
    parser.add_argument('--train_patch_num', type=int, default=25,
                       help='Number of patches per image during training')
    parser.add_argument('--test_patch_num', type=int, default=25,
                       help='Number of patches per image during testing')
    parser.add_argument('--no_color_jitter', dest='use_color_jitter', 
                       action='store_false',
                       help='Disable color jitter augmentation')
    parser.add_argument('--test_random_crop', action='store_true',
                       help='Use random crop for testing')
    
    # Other settings
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_model', action='store_true',
                       help='Save best model')
    parser.add_argument('--preload_images', action='store_true',
                       help='Preload all images into memory (faster training, requires ~10GB RAM)')
    
    config = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"ResNet50 + Improvements Experiment")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Multi-scale: {config.use_multiscale}")
    print(f"  Attention: {config.use_attention}")
    print(f"  Learning Rate: {config.lr}")
    print(f"  Epochs: {config.epochs}")
    print(f"{'='*80}\n")
    
    main(config)

