import torch
from scipy import stats
import numpy as np
import models
import data_loader
from tqdm import tqdm
import os

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
        
        # 创建模型保存目录
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', self.dataset)
        os.makedirs(self.save_dir, exist_ok=True)
        print(f'Model checkpoints will be saved to: {self.save_dir}')

        self.model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(self.device)
        self.model_hyper.train(True)

        self.l1_loss = torch.nn.L1Loss().to(self.device)

        backbone_params = list(map(id, self.model_hyper.res.parameters()))
        self.hypernet_params = list(filter(lambda p: id(p) not in backbone_params, self.model_hyper.parameters()))
        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay
        paras = [{'params': self.hypernet_params, 'lr': self.lr * self.lrratio},
                 {'params': self.model_hyper.res.parameters(), 'lr': self.lr}
                 ]
        self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        train_loader = data_loader.DataLoader(config.dataset, path, train_idx, config.patch_size, config.train_patch_num, batch_size=config.batch_size, istrain=True)
        test_loader = data_loader.DataLoader(config.dataset, path, test_idx, config.patch_size, config.test_patch_num, istrain=False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

    def train(self):
        """Training"""
        best_srcc = 0.0
        best_plcc = 0.0
        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')
        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            # Use tqdm for progress bar with total length
            total_batches = len(self.train_data)
            print(f'  Total batches: {total_batches}')
            print(f'  Loading first batch (this may take a moment)...')
            train_loader_with_progress = tqdm(
                self.train_data, 
                desc=f'Epoch {t+1}/{self.epochs}',
                total=total_batches,
                unit='batch',
                mininterval=0.5,
                maxinterval=2.0,
                smoothing=0.1,
                initial=0
            )
            for batch_idx, (img, label) in enumerate(train_loader_with_progress):
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
            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
            print('%d\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc))

            # 每两个epoch保存一次模型
            if (t + 1) % 2 == 0:
                model_path = os.path.join(self.save_dir, f'checkpoint_epoch_{t+1}_srcc_{test_srcc:.4f}_plcc_{test_plcc:.4f}.pkl')
                torch.save(self.model_hyper.state_dict(), model_path)
                print(f'  Model saved to: {model_path}')

            # Update optimizer
            lr = self.lr / pow(10, (t // 6))
            if t > 8:
                self.lrratio = 1
            self.paras = [{'params': self.hypernet_params, 'lr': lr * self.lrratio},
                          {'params': self.model_hyper.res.parameters(), 'lr': self.lr}
                          ]
            self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)

        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))

        return best_srcc, best_plcc

    def test(self, data):
        """Testing"""
        self.model_hyper.train(False)
        pred_scores = []
        gt_scores = []

        # Use tqdm for progress bar during testing
        total_test_batches = len(data)
        test_loader_with_progress = tqdm(
            data, 
            desc='Testing',
            total=total_test_batches,
            unit='batch',
            mininterval=1.0
        )
        for img, label in test_loader_with_progress:
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