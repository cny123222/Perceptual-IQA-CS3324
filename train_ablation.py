"""
Ablation study training script
Based on original HyperIQA, adding AFA and Attention
"""
import os
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from scipy import stats

# Import custom modules
import data_loader
from models_ablation import HyperNet, TargetNet


class AblationSolver(object):
    """Solver for ablation study experiments"""
    
    def __init__(self, config, train_idx, test_idx, use_afa=False, use_attention=False):
        
        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num
        self.use_afa = use_afa
        self.use_attention = use_attention
        
        self.model_hyper = HyperNet(
            lda_out_channels=128,
            hyper_in_channels=112,
            target_in_size=128,
            target_fc1_size=112,
            target_fc2_size=56,
            target_fc3_size=28,
            target_fc4_size=14,
            feature_size=7,
            use_afa=use_afa,
            use_attention=use_attention
        ).cuda()
        
        self.model_hyper.train(True)
        
        # Loss and optimizer
        self.l1_loss = torch.nn.L1Loss().cuda()
        
        backbone_params = []
        hyper_params = []
        for name, param in self.model_hyper.named_parameters():
            if 'res.' in name:
                backbone_params.append(param)
            else:
                hyper_params.append(param)
        
        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay
        paras = [{'params': hyper_params, 'lr': self.lr * self.lrratio},
                {'params': backbone_params, 'lr': self.lr}]
        self.solver = optim.Adam(paras, weight_decay=self.weight_decay)
        
        # Data (pass dataset root, not subdirectories for koniq-10k)
        dataset_root = './koniq-10k' if config.dataset == 'koniq-10k' else config.train_image_dir
        
        train_loader = data_loader.DataLoader(
            config.dataset,
            dataset_root,
            train_idx,
            config.patch_size,
            config.train_patch_num,
            batch_size=config.batch_size,
            istrain=True,
            preload=config.preload_images
        )
        
        test_loader = data_loader.DataLoader(
            config.dataset,
            dataset_root,
            test_idx,
            config.patch_size,
            config.test_patch_num,
            istrain=False,
            preload=config.preload_images
        )
        
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()
    
    def train(self):
        """Training"""
        print('\nStarting training...\n')
        best_srcc = 0.0
        best_plcc = 0.0
        best_epoch = 0
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # Train
            self.model_hyper.train(True)
            loss_epoch = self.train_epoch(epoch)
            
            # Evaluate on train and test
            train_srcc, train_plcc = self.test(self.train_data)
            test_srcc, test_plcc = self.test(self.test_data)
            
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"  Train Loss: {loss_epoch:.4f}, SRCC: {train_srcc:.4f}, PLCC: {train_plcc:.4f}")
            print(f"  Test  SRCC: {test_srcc:.4f}, PLCC: {test_plcc:.4f}")
            print(f"  Time: {epoch_time:.1f}s")
            
            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
                best_epoch = epoch + 1
                print(f"  ✓ New best: SRCC={best_srcc:.4f}, PLCC={best_plcc:.4f}")
            
            print()
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time/60:.1f} minutes")
        print(f"Best: SRCC={best_srcc:.4f}, PLCC={best_plcc:.4f} (Epoch {best_epoch})")
        
        return best_srcc, best_plcc
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        losses = []
        
        for i, (patches, label) in enumerate(self.train_data):
            patches = patches.cuda()
            label = label.cuda()
            
            # Generate weights for target network
            self.solver.zero_grad()
            
            paras = self.model_hyper(patches)  # 'paras' contains the network weights conveyed to target network
            
            # Building target network
            model_target = TargetNet(paras).cuda()
            for param in model_target.parameters():
                param.requires_grad = False
            
            # Quality prediction
            pred = model_target(paras['target_in_vec'])  # while 'paras['target_in_vec']' is the input to target net
            pred_scores = pred.mean(dim=(2, 3))
            
            # Compute loss
            loss = self.l1_loss(pred_scores.squeeze(), label.float().detach())
            losses.append(loss.item())
            
            loss.backward()
            self.solver.step()
            
            if (i + 1) % 50 == 0:
                print(f"  Batch {i+1}/{len(self.train_data)}, Loss: {loss.item():.4f}", flush=True)
        
        return np.mean(losses)
    
    def test(self, data):
        """Test"""
        self.model_hyper.train(False)
        pred_scores = []
        gt_scores = []

        for patches, label in data:
            patches = patches.cuda()
            label = label.cpu().data.numpy()
            
            with torch.no_grad():
                paras = self.model_hyper(patches)
                model_target = TargetNet(paras).cuda()
                model_target.train(False)
                pred = model_target(paras['target_in_vec'])
            
            pred_scores.append(float(pred.mean()))
            gt_scores.append(float(label))

        pred_scores = np.array(pred_scores)
        gt_scores = np.array(gt_scores)
        
        srcc = stats.spearmanr(pred_scores, gt_scores)[0]
        plcc = stats.pearsonr(pred_scores, gt_scores)[0]
        
        self.model_hyper.train(True)
        
        return srcc, plcc


def main(config):
    # Get data indices
    import data_loader
    train_idx, test_idx = data_loader.get_koniq_splits('./koniq-10k')
    
    print("=" * 80)
    if config.use_afa and config.use_attention:
        exp_name = "ResNet50 + AFA + Attention"
    elif config.use_afa:
        exp_name = "ResNet50 + AFA"
    else:
        exp_name = "ResNet50 Baseline"
    
    print(f"Experiment: {exp_name}")
    print(f"Dataset: {config.dataset}")
    print(f"Epochs: {config.epochs}, Batch size: {config.batch_size}")
    print(f"Learning rate: {config.lr}, LR ratio: {config.lr_ratio}")
    print(f"Train patches: {config.train_patch_num}, Test patches: {config.test_patch_num}")
    print(f"Preload images: {config.preload_images}")
    print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
    print("=" * 80)
    print()
    
    # Train
    solver = AblationSolver(config, train_idx, test_idx, 
                           use_afa=config.use_afa, 
                           use_attention=config.use_attention)
    
    srcc, plcc = solver.train()
    
    print("\n" + "=" * 80)
    print(f"Final Results for {exp_name}:")
    print(f"  SRCC: {srcc:.4f}")
    print(f"  PLCC: {plcc:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='koniq-10k', help='Dataset name')
    parser.add_argument('--train_image_dir', dest='train_image_dir', type=str, default='./koniq-10k/train', help='Training images directory')
    parser.add_argument('--test_image_dir', dest='test_image_dir', type=str, default='./koniq-10k/test', help='Test images directory')
    parser.add_argument('--epochs', dest='epochs', type=int, default=10, help='Epochs for training')
    parser.add_argument('--lr', dest='lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=float, default=10, help='Learning rate ratio for hyper/target')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=96, help='Batch size')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224, help='Crop size for training & testing image patches')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=25, help='Number of patches per training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=25, help='Number of patches per test image')
    
    # Ablation flags
    parser.add_argument('--use_afa', action='store_true', help='Use Adaptive Feature Aggregation (multi-scale)')
    parser.add_argument('--use_attention', action='store_true', help='Use channel attention')
    
    # Preload
    parser.add_argument('--preload_images', action='store_true', help='Preload images into memory')
    
    config = parser.parse_args()
    
    main(config)

