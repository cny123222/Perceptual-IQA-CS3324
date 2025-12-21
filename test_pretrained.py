import torch
import numpy as np
from scipy import stats
import models
import data_loader
from argparse import ArgumentParser
import os

def test_model(model_path, dataset='koniq-10k'):
    """Test pretrained model on specified dataset"""
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print(f'Loading pretrained model from: {model_path}')
    model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(device)
    model_hyper.load_state_dict(torch.load(model_path, map_location=device))
    model_hyper.eval()
    print('Model loaded successfully')
    
    # Dataset paths
    folder_path = {
        'live': '/home/ssl/Database/databaserelease2/',
        'csiq': '/home/ssl/Database/CSIQ/',
        'tid2013': '/home/ssl/Database/TID2013/',
        'livec': '/home/ssl/Database/ChallengeDB_release/',
        'koniq-10k': '/root/Perceptual-IQA-CS3324/koniq-10k/',
        'bid': '/home/ssl/Database/BID/',
    }
    
    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),
        'livec': list(range(0, 1162)),
        'koniq-10k': list(range(0, 10073)),
        'bid': list(range(0, 586)),
    }
    
    print(f'\nTesting on {dataset} dataset...')
    
    if dataset == 'koniq-10k':
        # Use official test split for KonIQ-10k
        import csv as csv_module
        import json
        csv_file = os.path.join(folder_path[dataset], 'koniq10k_scores_and_distributions.csv')
        train_json = os.path.join(folder_path[dataset], 'koniq_train.json')
        test_json = os.path.join(folder_path[dataset], 'koniq_test.json')
        
        # Read official train/test split from JSON
        with open(train_json) as f:
            train_data = json.load(f)
            train_images = set([item['image_name'] for item in train_data])
        
        with open(test_json) as f:
            test_data = json.load(f)
            test_images = set([item['image_name'] for item in test_data])
        
        # Convert to indices
        train_indices = [int(img.replace('.jpg', '')) for img in train_images]
        test_indices = [int(img.replace('.jpg', '')) for img in test_images]
        print(f'KonIQ-10k: {len(train_indices)} train images, {len(test_indices)} test images')
        
        # Load test data
        test_loader = data_loader.DataLoader(
            dataset,
            folder_path[dataset],
            test_indices,
            patch_size=224,
            test_patch_num=20,
            batch_size=1,  # Use batch_size=1 for testing
            istrain=False,
            pin_memory=True
        )
    else:
        # For other datasets, use all images
        sel_num = img_num[dataset]
        test_loader = data_loader.DataLoader(
            dataset,
            folder_path[dataset],
            sel_num,
            patch_size=224,
            test_patch_num=20,
            batch_size=1,
            istrain=False
        )
    
    print(f'Testing...')
    pred_scores = []
    gt_scores = []
    
    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(device)
            label = label.float().to(device)
            
            # Generate weights for target network
            paras = model_hyper(img)
            
            # Building target network
            model_target = models.TargetNet(paras).to(device)
            
            # Quality prediction
            pred = model_target(paras['target_in_vec'])
            pred_scores.append(pred.item())
            gt_scores.append(label.item())
    
    # Calculate metrics
    pred_scores = np.array(pred_scores)
    gt_scores = np.array(gt_scores)
    
    test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
    test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
    
    print(f'\n{"="*60}')
    print(f'Results on {dataset}:')
    print(f'{"="*60}')
    print(f'SRCC: {test_srcc:.6f}')
    print(f'PLCC: {test_plcc:.6f}')
    print(f'{"="*60}\n')
    
    return test_srcc, test_plcc

if __name__ == '__main__':
    parser = ArgumentParser(description='Test pretrained HyperIQA model')
    parser.add_argument('--model_path', type=str, 
                        default='/root/Perceptual-IQA-CS3324/pretrained/koniq_pretrained.pkl',
                        help='Path to pretrained model')
    parser.add_argument('--dataset', type=str, default='koniq-10k',
                        choices=['live', 'csiq', 'tid2013', 'livec', 'koniq-10k', 'bid'],
                        help='Dataset to test on')
    parser.add_argument('--all', action='store_true',
                        help='Test on all available datasets')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if args.all:
        print('Testing on all datasets...\n')
        results = {}
        for dataset in ['koniq-10k', 'live', 'csiq', 'livec']:
            try:
                srcc, plcc = test_model(args.model_path, dataset)
                results[dataset] = {'srcc': srcc, 'plcc': plcc}
            except Exception as e:
                print(f'Error testing {dataset}: {e}\n')
                results[dataset] = {'srcc': None, 'plcc': None}
        
        # Print summary
        print('\n' + '='*60)
        print('SUMMARY OF ALL DATASETS')
        print('='*60)
        print(f'{"Dataset":<15} {"SRCC":>10} {"PLCC":>10}')
        print('-'*60)
        for dataset, metrics in results.items():
            if metrics['srcc'] is not None:
                print(f'{dataset:<15} {metrics["srcc"]:>10.6f} {metrics["plcc"]:>10.6f}')
            else:
                print(f'{dataset:<15} {"N/A":>10} {"N/A":>10}')
        print('='*60)
    else:
        test_model(args.model_path, args.dataset)

