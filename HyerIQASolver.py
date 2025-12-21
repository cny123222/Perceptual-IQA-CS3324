import torch
from scipy import stats
import numpy as np
import models
import data_loader
from tqdm import tqdm
import os
from datetime import datetime

# 自动检测可用设备：优先使用 CUDA，然后是 MPS (macOS GPU)，最后使用 CPU
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

class HyperIQASolver(object):
    """Solver for training and testing hyperIQA"""
    def __init__(self, config, path, train_idx, test_idx):

        self.device = get_device()
        print(f'Using device: {self.device}')

        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num
        self.dataset = config.dataset
        
        # Early stopping parameters
        self.patience = getattr(config, 'patience', 5)  # Default: stop after 5 epochs with no improvement
        self.early_stopping_enabled = getattr(config, 'early_stopping', True)  # Enable by default
        
        # Learning rate scheduler parameters
        self.use_lr_scheduler = getattr(config, 'use_lr_scheduler', True)  # Enable by default
        self.lr_scheduler_type = getattr(config, 'lr_scheduler_type', 'step')  # 'step' (original) or 'cosine'
        
        # Test crop method
        self.test_random_crop = getattr(config, 'test_random_crop', False)  # Default: CenterCrop for reproducibility
        
        # SPAQ cross-dataset testing
        self.test_spaq = getattr(config, 'test_spaq', True)  # Default: enable SPAQ testing
        
        # 创建模型保存目录（带时间戳防止覆盖）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir_name = f"{self.dataset}-resnet_{timestamp}"
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', save_dir_name)
        os.makedirs(self.save_dir, exist_ok=True)
        print(f'Model checkpoints will be saved to: {self.save_dir}')

        self.model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(self.device)
        self.model_hyper.train(True)

        self.l1_loss = torch.nn.L1Loss().to(self.device)

        backbone_params = list(map(id, self.model_hyper.res.parameters()))
        self.hypernet_params = filter(lambda p: id(p) not in backbone_params, self.model_hyper.parameters())
        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay
        paras = [{'params': self.hypernet_params, 'lr': self.lr * self.lrratio},
                 {'params': self.model_hyper.res.parameters(), 'lr': self.lr}
                 ]
        self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        train_loader = data_loader.DataLoader(config.dataset, path, train_idx, config.patch_size, config.train_patch_num, batch_size=config.batch_size, istrain=True)
        test_loader = data_loader.DataLoader(config.dataset, path, test_idx, config.patch_size, config.test_patch_num, istrain=False, test_random_crop=self.test_random_crop)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()
        
        # 初始化SPAQ数据集用于跨数据集测试（如果存在）
        self.spaq_path = None
        self.spaq_loader = None
        base_dir = os.path.dirname(os.path.abspath(__file__))
        spaq_base_path = os.path.join(base_dir, 'spaq-test')
        spaq_json_path = os.path.join(spaq_base_path, 'spaq_test.json') if os.path.exists(spaq_base_path) else None
        
        if self.test_spaq:
            if spaq_json_path and os.path.exists(spaq_json_path):
                self.spaq_path = spaq_base_path
                print(f'SPAQ test dataset found at: {self.spaq_path}')
                # 在初始化时加载SPAQ数据集，避免每个epoch重复加载
                self._init_spaq_dataset()
            else:
                print('SPAQ dataset not found. SPAQ testing will be skipped.')
        else:
            print('SPAQ cross-dataset testing: DISABLED (use --test_spaq to enable)')

    def _count_parameters(self):
        """Count total trainable parameters in millions"""
        return sum(p.numel() for p in self.model_hyper.parameters() if p.requires_grad) / 1e6

    def train(self):
        """Training"""
        # Print detailed training configuration
        print("\n" + "=" * 80)
        print("TRAINING CONFIGURATION SUMMARY (ResNet-50 Baseline)")
        print("=" * 80)
        print(f"Model Architecture:")
        print(f"  Backbone:                 ResNet-50")
        print(f"  Parameters:               ~{self._count_parameters():.1f}M")
        print("-" * 80)
        print(f"Loss Function:")
        print(f"  Type:                     L1 Loss Only")
        print("-" * 80)
        print(f"Regularization:")
        print(f"  Weight Decay:             {self.weight_decay}")
        print(f"  Early Stopping:           {'ENABLED' if self.early_stopping_enabled else 'DISABLED'}")
        if self.early_stopping_enabled:
            print(f"  Patience:                 {self.patience} epochs")
        print("-" * 80)
        print(f"Optimization:")
        print(f"  Learning Rate (HyperNet): {self.lr * self.lrratio}")
        print(f"  Learning Rate (Backbone): {self.lr}")
        print(f"  LR Scheduler:             {self.lr_scheduler_type if self.use_lr_scheduler else 'None'}")
        print(f"  Optimizer:                Adam")
        print("-" * 80)
        print(f"Data:")
        print(f"  Test Crop Method:         {'RandomCrop' if self.test_random_crop else 'CenterCrop'}")
        print(f"  Test Patch Num:           {self.test_patch_num}")
        print(f"  Cross-Dataset Test:       {'SPAQ' if self.spaq_path else 'None'}")
        print("-" * 80)
        print(f"Training:")
        print(f"  Epochs:                   {self.epochs}")
        print(f"  Checkpoint Directory:     {self.save_dir}")
        print("=" * 80 + "\n")
        
        best_srcc = 0.0
        best_plcc = 0.0
        
        # Early stopping variables
        epochs_no_improve = 0
        best_model_path = None
        
        # Learning rate scheduler
        if self.use_lr_scheduler:
            if self.lr_scheduler_type == 'cosine':
                from torch.optim.lr_scheduler import CosineAnnealingLR
                self.scheduler = CosineAnnealingLR(self.solver, T_max=self.epochs, eta_min=1e-6)
                print(f'Learning rate scheduler: CosineAnnealingLR (T_max={self.epochs}, eta_min=1e-6)')
            elif self.lr_scheduler_type == 'step':
                print('Learning rate scheduler: Step decay (original, divide by 10 every 6 epochs)')
                self.scheduler = None  # Will use manual step decay
            else:
                print(f'Unknown scheduler type: {self.lr_scheduler_type}, using step decay')
                self.scheduler = None
        else:
            print('Learning rate scheduler: DISABLED (constant LR)')
            self.scheduler = None
        
        if self.early_stopping_enabled:
            print(f'Early stopping enabled with patience={self.patience}')
        
        if self.spaq_path is not None:
            print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC\tSPAQ_SRCC\tSPAQ_PLCC')
        else:
            print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')
        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            for batch_idx, (img, label) in enumerate(self.train_data):
                img = img.to(self.device)
                label = label.float().to(self.device)

                self.solver.zero_grad()

                # Generate weights for target network
                paras = self.model_hyper(img)

                # Building target network
                model_target = models.TargetNet(paras).to(self.device)
                for param in model_target.parameters():
                    param.requires_grad = False

                # Quality prediction
                pred = model_target(paras['target_in_vec'])
                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

                loss = self.l1_loss(pred.squeeze(), label.float().detach())
                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()
                
                # Update progress bar with current loss
                if batch_idx % 10 == 0:
                    avg_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0.0
                    train_loader_with_progress.set_postfix({'loss': f'{avg_loss:.4f}'})

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)

            test_srcc, test_plcc = self.test(self.test_data)
            
            # Check if this is the best model so far
            improved = False
            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
                improved = True
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            # 在SPAQ数据集上测试
            spaq_srcc, spaq_plcc = None, None
            if self.spaq_path is not None:
                spaq_srcc, spaq_plcc = self.test_spaq()
            
            # Print epoch results
            if self.spaq_path is not None and spaq_srcc is not None:
                print('%d\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                      (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc, spaq_srcc, spaq_plcc))
            else:
                print('%d\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                      (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc))

            # Only save best model (not every epoch to save disk space)
            if improved:
                if self.spaq_path is not None and spaq_srcc is not None:
                    best_model_path = os.path.join(self.save_dir, f'best_model_srcc_{best_srcc:.4f}_plcc_{best_plcc:.4f}_spaq_srcc_{spaq_srcc:.4f}_plcc_{spaq_plcc:.4f}.pkl')
                else:
                    best_model_path = os.path.join(self.save_dir, f'best_model_srcc_{best_srcc:.4f}_plcc_{best_plcc:.4f}.pkl')
                torch.save(self.model_hyper.state_dict(), best_model_path)
                print(f'  ⭐ New best model saved! SRCC: {best_srcc:.4f}, PLCC: {best_plcc:.4f}')
                print(f'     Path: {best_model_path}')
            
            # Early stopping check
            if self.early_stopping_enabled and epochs_no_improve >= self.patience:
                print(f'\n🛑 Early stopping triggered!')
                print(f'   No improvement for {self.patience} consecutive epochs.')
                print(f'   Best SRCC: {best_srcc:.4f}, Best PLCC: {best_plcc:.4f}')
                print(f'   Best model saved at: {best_model_path}')
                break

            # Learning rate update
            if self.use_lr_scheduler and self.scheduler is not None:
                # Use PyTorch scheduler (e.g., CosineAnnealingLR)
                self.scheduler.step()
                current_lr_hypernet = self.solver.param_groups[0]['lr']
                current_lr_backbone = self.solver.param_groups[1]['lr']
                print(f'  Learning rates: HyperNet={current_lr_hypernet:.6f}, Backbone={current_lr_backbone:.6f}')
            elif self.use_lr_scheduler and self.lr_scheduler_type == 'step':
                # Original step decay: recreate optimizer each epoch
                hypernet_lr = self.lr * self.lrratio / pow(10, (t // 6))
                backbone_lr = self.lr / pow(10, (t // 6))  # Backbone LR also decays (fixed bug)
                
                self.paras = [{'params': self.hypernet_params, 'lr': hypernet_lr},
                              {'params': self.model_hyper.res.parameters(), 'lr': backbone_lr}]
                self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)
                print(f'  Learning rates: HyperNet={hypernet_lr:.6f}, Backbone={backbone_lr:.6f}')
            # else: constant LR, no update needed

        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))

        return best_srcc, best_plcc

    def test(self, data):
        """Testing"""
        self.model_hyper.train(False)
        pred_scores = []
        gt_scores = []

        with torch.no_grad():  # Disable gradient computation for faster inference
            for img, label in data:
                # DataLoader returns tensors, so use .to() directly to avoid warning
                img = img.to(self.device)
                label = label.float().to(self.device)  # MPS/CUDA 需要 float32

                paras = self.model_hyper(img)
                model_target = models.TargetNet(paras).to(self.device)
                model_target.train(False)
                pred = model_target(paras['target_in_vec'])

                pred_scores.append(float(pred.item()))
                gt_scores = gt_scores + label.cpu().tolist()

            pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
            gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
            test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
            test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        self.model_hyper.train(True)
        return test_srcc, test_plcc

    def _init_spaq_dataset(self):
        """Initialize SPAQ dataset in __init__ to avoid reloading every epoch"""
        import json
        from PIL import Image
        import torchvision
        
        json_path = os.path.join(self.spaq_path, 'spaq_test.json')
        if not os.path.exists(json_path):
            self.spaq_loader = None
            return
        
        # Load SPAQ test data
        with open(json_path) as f:
            spaq_data = json.load(f)
        
        def pil_loader(path):
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        
        # Create dataset similar to JSONTestFolder in test_pretrained.py
        # Generate samples: each image has test_patch_num patches
        samples = []
        for item in spaq_data:
            img_path = os.path.join(self.spaq_path, os.path.basename(item['image']))
            if not os.path.exists(img_path):
                continue
            score = float(item['score'])
            for _ in range(self.test_patch_num):
                samples.append((img_path, score))
        
        if len(samples) == 0:
            self.spaq_loader = None
            return
        
        # Create a dataset class with optimized caching: pre-resize images
        # SPAQ images are much larger (13MP vs 0.8MP), so pre-resizing saves significant time
        class SPAQDataset(torch.utils.data.Dataset):
            def __init__(self, samples, transform):
                self.samples = samples
                # Split transform: Resize is expensive for large images, cache it
                self.resize_transform = torchvision.transforms.Resize((512, 384))
                # Use CenterCrop for testing (reproducible results)
                self.crop_transform = torchvision.transforms.Compose([
                    torchvision.transforms.CenterCrop(size=224),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
                
                # Cache resized images (key: path, value: resized PIL Image)
                self._resized_cache = {}
                # Pre-load and resize unique images
                unique_paths = list(set([s[0] for s in samples]))
                print(f'  Pre-loading {len(unique_paths)} SPAQ images into cache...')
                for path in tqdm(unique_paths, desc='  Loading SPAQ images', unit='img'):
                    if os.path.exists(path):
                        try:
                            img = pil_loader(path)
                            self._resized_cache[path] = self.resize_transform(img)
                        except Exception as e:
                            print(f'  Warning: Failed to load {path}: {e}')
                            continue
                print(f'  Cached {len(self._resized_cache)} SPAQ images successfully')
            
            def __getitem__(self, index):
                path, target = self.samples[index]
                # Get pre-resized image from cache
                resized_img = self._resized_cache.get(path)
                if resized_img is None:
                    # Fallback
                    img = pil_loader(path)
                    resized_img = self.resize_transform(img)
                    self._resized_cache[path] = resized_img
                
                # Only do CenterCrop + ToTensor + Normalize (fast, deterministic)
                sample = self.crop_transform(resized_img)
                return sample, target
            
            def __len__(self):
                return len(self.samples)
        
        # Create DataLoader (same as KonIQ test)
        spaq_dataset = SPAQDataset(samples, None)  # transform not used, handled in class
        self.spaq_loader = torch.utils.data.DataLoader(
            spaq_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
        self.spaq_num_images = len(spaq_data)
        print(f'  SPAQ dataset initialized: {self.spaq_num_images} images, {len(samples)} patches')
    
    def test_spaq(self):
        """Test on SPAQ dataset for cross-dataset evaluation"""
        if self.spaq_loader is None:
            return None, None
        
        self.model_hyper.train(False)
        pred_scores = []
        gt_scores = []
        
        print(f'  Testing on SPAQ dataset ({self.spaq_num_images} images)...')
        
        with torch.no_grad():  # Disable gradient computation for faster inference
            for img, label in self.spaq_loader:
                img = img.to(self.device)
                label = label.float().to(self.device)
                
                paras = self.model_hyper(img)
                model_target = models.TargetNet(paras).to(self.device)
                model_target.train(False)
                pred = model_target(paras['target_in_vec'])
                
                pred_scores.append(float(pred.item()))
                gt_scores = gt_scores + label.cpu().tolist()
        
        if len(pred_scores) == 0:
            self.model_hyper.train(True)
            return None, None
        
        # Reshape and average patches per image (same as KonIQ test)
        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
        
        spaq_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        spaq_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
        
        self.model_hyper.train(True)
        return spaq_srcc, spaq_plcc