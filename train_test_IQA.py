import os
import argparse
import random
import numpy as np
from HyerIQASolver import HyperIQASolver

# macOS 不支持 CUDA，移除 CUDA_VISIBLE_DEVICES 设置
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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
        # koniq-10k: CSV has 10073 records, but only 9056 images are available
        # (7046 train + 2010 test). The code will skip missing images automatically.
        'koniq-10k': list(range(0, 10073)),  # Use CSV count, missing images will be skipped
        'bid': list(range(0, 586)),
    }
    sel_num = img_num[config.dataset]

    srcc_all = np.zeros(config.train_test_num, dtype=np.float64)
    plcc_all = np.zeros(config.train_test_num, dtype=np.float64)

    print('Training and testing on %s dataset for %d rounds...' % (config.dataset, config.train_test_num))
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

