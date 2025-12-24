"""
Clean ResNet ablation study training script
Tests: Baseline, +AFA, +AFA+Attention
"""
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from scipy import stats
import csv
import json


class ResNetIQA(nn.Module):
    """Simple ResNet-based IQA model"""
    
    def __init__(self, use_afa=False, use_attention=False):
        super(ResNetIQA, self).__init__()
        self.use_afa = use_afa
        self.use_attention = use_attention
        
        # Load pretrained ResNet50
        from torchvision.models import resnet50
        resnet = resnet50(pretrained=True)
        
        # Extract feature extractor (remove final FC layer)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels
        
        # Pooling for multi-scale
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Determine feature dimension
        if use_afa:
            # Multi-scale: concatenate all 4 layers
            feat_dim = 256 + 512 + 1024 + 2048  # 3840
            
            if use_attention:
                # Channel attention
                self.channel_attention = nn.Sequential(
                    nn.Linear(feat_dim, feat_dim // 16),
                    nn.ReLU(),
                    nn.Linear(feat_dim // 16, feat_dim),
                    nn.Sigmoid()
                )
        else:
            # Baseline: only use layer4
            feat_dim = 2048
        
        # Quality regression head
        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        # Extract features
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        f1 = self.layer1(x)   # (B, 256, H/4, W/4)
        f2 = self.layer2(f1)  # (B, 512, H/8, W/8)
        f3 = self.layer3(f2)  # (B, 1024, H/16, W/16)
        f4 = self.layer4(f3)  # (B, 2048, H/32, W/32)
        
        if self.use_afa:
            # Adaptive Feature Aggregation: pool all to same size
            f1_pooled = self.pool(f1)  # (B, 256, 7, 7)
            f2_pooled = self.pool(f2)  # (B, 512, 7, 7)
            f3_pooled = self.pool(f3)  # (B, 1024, 7, 7)
            f4_pooled = self.pool(f4)  # (B, 2048, 7, 7)
            
            # Global average pooling
            f1_gap = f1_pooled.mean(dim=[2, 3])  # (B, 256)
            f2_gap = f2_pooled.mean(dim=[2, 3])  # (B, 512)
            f3_gap = f3_pooled.mean(dim=[2, 3])  # (B, 1024)
            f4_gap = f4_pooled.mean(dim=[2, 3])  # (B, 2048)
            
            # Concatenate
            feat = torch.cat([f1_gap, f2_gap, f3_gap, f4_gap], dim=1)  # (B, 3840)
            
            if self.use_attention:
                # Apply channel attention
                attn = self.channel_attention(feat)  # (B, 3840)
                feat = feat * attn
        else:
            # Baseline: only use layer4
            feat = f4.mean(dim=[2, 3])  # (B, 2048)
        
        # Predict quality score
        score = self.regressor(feat)  # (B, 1)
        
        return score


class KoniqDataset(Dataset):
    """KonIQ-10k dataset"""
    
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform
        
        # Load MOS scores
        csv_file = os.path.join(root, 'koniq10k_scores_and_distributions.csv')
        img_to_mos = {}
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_to_mos[row['image_name']] = float(row['MOS_zscore'])
        
        # Load split
        if split == 'train':
            json_file = os.path.join(root, 'koniq_train.json')
            img_dir = os.path.join(root, 'train')
        else:
            json_file = os.path.join(root, 'koniq_test.json')
            img_dir = os.path.join(root, 'test')
        
        with open(json_file) as f:
            data = json.load(f)
        
        # Build sample list
        self.samples = []
        for item in data:
            img_name = os.path.basename(item['image'])
            img_path = os.path.join(img_dir, img_name)
            
            if os.path.exists(img_path) and img_name in img_to_mos:
                self.samples.append({
                    'path': img_path,
                    'mos': img_to_mos[img_name]
                })
        
        print(f"  {split}: {len(self.samples)} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        mos = torch.FloatTensor([sample['mos']])
        
        return img, mos


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    losses = []
    
    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward
        preds = model(images)
        loss = criterion(preds, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (i + 1) % 50 == 0:
            print(f"    Batch {i+1}/{len(loader)}, Loss: {loss.item():.4f}", flush=True)
    
    return np.mean(losses)


def evaluate(model, loader, device):
    """Evaluate model"""
    model.eval()
    preds_all = []
    labels_all = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            
            preds = model(images)
            
            preds_all.extend(preds.cpu().numpy().flatten())
            labels_all.extend(labels.numpy().flatten())
    
    preds_all = np.array(preds_all)
    labels_all = np.array(labels_all)
    
    srcc = stats.spearmanr(preds_all, labels_all)[0]
    plcc = stats.pearsonr(preds_all, labels_all)[0]
    
    return srcc, plcc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_afa', action='store_true', help='Use AFA (multi-scale)')
    parser.add_argument('--use_attention', action='store_true', help='Use channel attention')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Config name
    if args.use_afa and args.use_attention:
        config_name = "ResNet+AFA+Attention"
    elif args.use_afa:
        config_name = "ResNet+AFA"
    else:
        config_name = "ResNet-Baseline"
    
    print("=" * 80)
    print(f"Training: {config_name}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((512, 384)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((512, 384)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    print("Loading datasets...")
    train_dataset = KoniqDataset('./koniq-10k', split='train', transform=train_transform)
    test_dataset = KoniqDataset('./koniq-10k', split='test', transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"  Train batches: {len(train_loader)}, Test batches: {len(test_loader)}\n")
    
    # Model
    print("Creating model...")
    model = ResNetIQA(use_afa=args.use_afa, use_attention=args.use_attention).to(device)
    print(f"  Model created\n")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.L1Loss()
    
    # Training loop
    print("Starting training...\n")
    best_srcc = 0
    best_epoch = 0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        train_srcc, train_plcc = evaluate(model, train_loader, device)
        test_srcc, test_plcc = evaluate(model, test_loader, device)
        
        epoch_time = time.time() - epoch_start
        
        print(f"  Train Loss: {train_loss:.4f}, SRCC: {train_srcc:.4f}, PLCC: {train_plcc:.4f}")
        print(f"  Test  SRCC: {test_srcc:.4f}, PLCC: {test_plcc:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        if test_srcc > best_srcc:
            best_srcc = test_srcc
            best_epoch = epoch + 1
            print(f"  ✓ New best SRCC: {best_srcc:.4f}")
        
        print()
    
    total_time = time.time() - start_time
    print("=" * 80)
    print(f"Training completed in {total_time/60:.1f} minutes")
    print(f"Best Test SRCC: {best_srcc:.4f} (Epoch {best_epoch})")
    print("=" * 80)


if __name__ == '__main__':
    main()

