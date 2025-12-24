"""
Simplified ResNet+improvements training - minimal dependencies
"""
import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from PIL import Image
import torchvision.transforms as transforms
import json

# Import model
from models_resnet_simple import create_model


class SimpleKoniqDataset(Dataset):
    """Simple KonIQ dataset without complex caching"""
    def __init__(self, root, indices, transform, patch_num=25):
        self.root = root
        self.transform = transform
        self.patch_num = patch_num
        
        # Load MOS scores
        import csv
        csv_file = os.path.join(root, 'koniq10k_scores_and_distributions.csv')
        imgnames = []
        mos_scores = []
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgnames.append(row['image_name'])
                mos_scores.append(float(row['MOS_zscore']))
        
        # Load train/test split
        train_json = os.path.join(root, 'koniq_train.json')
        test_json = os.path.join(root, 'koniq_test.json')
        
        with open(train_json) as f:
            train_data = json.load(f)
            train_imgs = set([os.path.basename(item['image']) for item in train_data])
        
        with open(test_json) as f:
            test_data = json.load(f)
            test_imgs = set([os.path.basename(item['image']) for item in test_data])
        
        # Build sample list
        self.samples = []
        for idx in indices:
            img_name = imgnames[idx]
            mos = mos_scores[idx]
            
            # Find image path
            img_path = None
            if img_name in train_imgs:
                img_path = os.path.join(root, 'train', img_name)
            elif img_name in test_imgs:
                img_path = os.path.join(root, 'test', img_name)
            
            if img_path and os.path.exists(img_path):
                # Create patch_num copies of this image
                for _ in range(patch_num):
                    self.samples.append((img_path, mos))
        
        print(f"Dataset: {len(self.samples)} samples from {len(indices)} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, mos = self.samples[idx]
        
        # Load and transform image
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        return img, torch.FloatTensor([mos])


def get_indices(root):
    """Get train/test indices"""
    train_json = os.path.join(root, 'koniq_train.json')
    test_json = os.path.join(root, 'koniq_test.json')
    
    # Load CSV to get image names
    import csv
    csv_file = os.path.join(root, 'koniq10k_scores_and_distributions.csv')
    imgnames = []
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            imgnames.append(row['image_name'])
    
    # Load JSON splits
    with open(train_json) as f:
        train_data = json.load(f)
        train_imgs = [os.path.basename(item['image']) for item in train_data]
    
    with open(test_json) as f:
        test_data = json.load(f)
        test_imgs = [os.path.basename(item['image']) for item in test_data]
    
    # Map to indices
    train_idx = [i for i, name in enumerate(imgnames) if name in train_imgs]
    test_idx = [i for i, name in enumerate(imgnames) if name in test_imgs]
    
    return np.array(train_idx), np.array(test_idx)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    losses = []
    
    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward
        outputs = model(images)
        preds = outputs['score']
        
        # Loss
        loss = criterion(preds.squeeze(), labels.squeeze())
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (i + 1) % 100 == 0:
            print(f"  Batch {i+1}/{len(loader)}, Loss: {loss.item():.4f}")
    
    return np.mean(losses)


def test(model, loader, device):
    """Test the model"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            
            # Forward
            outputs = model(images)
            preds = outputs['score']
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    srcc = stats.spearmanr(all_preds, all_labels)[0]
    plcc = stats.pearsonr(all_preds, all_labels)[0]
    
    return srcc, plcc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_multiscale', action='store_true', default=False)
    parser.add_argument('--use_attention', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--patch_num', type=int, default=25)
    config = parser.parse_args()
    
    print("=" * 80)
    print(f"ResNet + AFA + Attention Experiment")
    print(f"Multi-scale: {config.use_multiscale}, Attention: {config.use_attention}")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get data
    root = './koniq-10k'
    train_idx, test_idx = get_indices(root)
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((512, 384)),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((512, 384)),
        transforms.RandomCrop(224),  # Use random crop for testing too
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    print("\nCreating datasets...")
    train_dataset = SimpleKoniqDataset(root, train_idx, train_transform, config.patch_num)
    test_dataset = SimpleKoniqDataset(root, test_idx, test_transform, config.patch_num)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                            shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Model
    print("\nCreating model...")
    model = create_model(
        use_multiscale=config.use_multiscale,
        use_attention=config.use_attention,
        dropout_rate=0.3
    ).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    criterion = nn.L1Loss()
    
    # Training
    print("\nStarting training...")
    start_time = time.time()
    best_srcc = 0
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Test
        test_srcc, test_plcc = test(model, test_loader, device)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test SRCC: {test_srcc:.4f}, PLCC: {test_plcc:.4f}")
        
        if test_srcc > best_srcc:
            best_srcc = test_srcc
            print(f"  ✓ New best SRCC: {best_srcc:.4f}")
    
    total_time = time.time() - start_time
    print(f"\n✓ Training completed in {total_time/60:.1f} minutes")
    print(f"✓ Best SRCC: {best_srcc:.4f}")


if __name__ == '__main__':
    main()

