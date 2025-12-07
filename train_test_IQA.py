import os
import argparse
import random
import numpy as np
import csv
import json
from HyerIQASolver import HyperIQASolver

# 设置 CUDA 设备（如果使用 CUDA，取消注释下面一行并指定 GPU ID）
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第一个 GPU，可以改为 '0,1' 使用多个 GPU


def get_koniq_train_test_indices(root_path):
    """Get train and test indices for KonIQ-10k based on official split"""
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

    # 获取当前脚本所在目录的绝对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    folder_path = {
        'live': '/home/ssl/Database/databaserelease2/',
        'csiq': '/home/ssl/Database/CSIQ/',
        'tid2013': '/home/ssl/Database/TID2013/',
        'livec': '/home/ssl/Database/ChallengeDB_release/ChallengeDB_release/',
        'koniq-10k': os.path.join(base_dir, 'koniq-10k') + '/',
        'bid': '/home/ssl/Database/BID/',
    }

    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),
        'livec': list(range(0, 1162)),
        'koniq-10k': None,  # Will be handled separately
        'bid': list(range(0, 586)),
    }
    
    srcc_all = np.zeros(config.train_test_num, dtype=np.float64)
    plcc_all = np.zeros(config.train_test_num, dtype=np.float64)

    print('Training and testing on %s dataset for %d rounds...' % (config.dataset, config.train_test_num))
    
    # Special handling for koniq-10k to use official train/test split
    if config.dataset == 'koniq-10k':
        # Get indices for train and test sets based on official split
        train_indices_all, test_indices_all = get_koniq_train_test_indices(folder_path[config.dataset])
        print(f'KonIQ-10k: {len(train_indices_all)} train images, {len(test_indices_all)} test images')
        
        for i in range(config.train_test_num):
            print('Round %d' % (i+1))
            # Use all training images (no validation split, following original paper)
            train_index = train_indices_all
            test_index = test_indices_all
            
            print(f'  Train: {len(train_index)} images, Test: {len(test_index)} images')
            
            solver = HyperIQASolver(config, folder_path[config.dataset], train_index, test_index)
            srcc_all[i], plcc_all[i] = solver.train()
    else:
        # Original logic for other datasets
        sel_num = img_num[config.dataset]
        
        for i in range(config.train_test_num):
            print('Round %d' % (i+1))
            # Randomly select 80% images for training and the rest for testing
            random.shuffle(sel_num)
            train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
            test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]

            solver = HyperIQASolver(config, folder_path[config.dataset], train_index, test_index)
            srcc_all[i], plcc_all[i] = solver.train()

    # print(srcc_all)
    # print(plcc_all)
    srcc_med = np.median(srcc_all)
    plcc_med = np.median(plcc_all)

    print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc_med, plcc_med))

    # return srcc_med, plcc_med


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='livec', help='Support datasets: livec|koniq-10k|bid|live|csiq|tid2013')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=25, help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=25, help='Number of sample patches from testing image')
    parser.add_argument('--lr', dest='lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10, help='Learning rate ratio for hyper network')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=96, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=16, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224, help='Crop size for training & testing image patches')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=10, help='Train-test times')

    config = parser.parse_args()
    main(config)

