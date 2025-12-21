"""
Test pretrained HyperIQA model on KonIQ-10k test set
"""
import torch
from scipy import stats
import numpy as np
import models
import data_loader
import os
import csv
import json

def get_koniq_test_indices(root_path):
    """Get test indices for KonIQ-10k based on official split"""
    test_json = os.path.join(root_path, 'koniq_test.json')
    
    with open(test_json) as f:
        test_data = json.load(f)
        test_images = [item['image'].split('/')[-1] for item in test_data]  # Extract filename from path
    
    # Convert to indices
    test_indices = [int(img.replace('.jpg', '')) for img in test_images]
    return test_indices

# Configuration
model_path = '/root/Perceptual-IQA-CS3324/pretrained/koniq_pretrained.pkl'
dataset_path = '/root/Perceptual-IQA-CS3324/koniq-10k/'

# Set random seed
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load model
print(f'Loading pretrained model: {model_path}')
model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(device)
model_hyper.load_state_dict(torch.load(model_path, map_location=device))
model_hyper.eval()
print('Model loaded successfully\n')

# Get test indices
test_indices = get_koniq_test_indices(dataset_path)
print(f'KonIQ-10k test set: {len(test_indices)} images\n')

# Load test data
print('Loading test data...')
test_data = data_loader.DataLoader(
    'koniq-10k',
    dataset_path,
    test_indices,
    224,  # patch_size
    20,   # patch_num (test_patch_num)
    1,    # batch_size
    istrain=False
)
print('Test data loaded\n')

# Test
print('Testing...')
pred_scores = []
gt_scores = []

with torch.no_grad():
    for img, label in test_data:
        img = img.to(device)
        label = label.float().to(device)
        
        # Generate weights for target network
        paras = model_hyper(img)
        
        # Building target network
        model_target = models.TargetNet(paras).to(device)
        for param in model_target.parameters():
            param.requires_grad = False
        
        # Quality prediction
        pred = model_target(paras['target_in_vec'])
        pred_scores.append(pred.item())
        gt_scores.append(label.item())

# Calculate metrics
pred_scores = np.array(pred_scores)
gt_scores = np.array(gt_scores)

test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

print('\n' + '='*80)
print('PRETRAINED MODEL TEST RESULTS')
print('='*80)
print(f'Model:           {model_path}')
print(f'Dataset:         KonIQ-10k (test set)')
print(f'Test samples:    {len(test_indices)}')
print('-'*80)
print(f'SRCC:            {test_srcc:.6f}')
print(f'PLCC:            {test_plcc:.6f}')
print('='*80)
print(f'\nExpected (from paper): SRCC ~0.9009')
print(f'Difference:            {test_srcc - 0.9009:.6f} ({(test_srcc - 0.9009) * 100:.2f}%)')
print('='*80)

