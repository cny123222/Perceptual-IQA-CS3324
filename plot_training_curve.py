#!/usr/bin/env python3
"""
绘制训练曲线脚本
Plot training curves from log file
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def extract_training_data(log_file, round_num=1):
    """从日志文件提取训练数据"""
    epochs = []
    train_losses = []
    train_srccs = []
    test_srccs = []
    test_plccs = []
    
    current_round = 0
    in_target_round = False
    
    with open(log_file, 'r') as f:
        for line in f:
            # 检测Round标记
            if re.match(r'^Round \d+', line):
                current_round += 1
                if current_round == round_num:
                    in_target_round = True
                elif current_round > round_num:
                    break  # 已经读取完目标轮次，退出
            
            # 只在目标轮次内提取数据
            if in_target_round:
                # 匹配格式: "1	15.8931		0.4873		0.9169		0.9285"
                # Epoch	Train_Loss	Train_SRCC	Test_SRCC	Test_PLCC
                match = re.match(r'^(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
                if match:
                    epoch = int(match.group(1))
                    train_loss = float(match.group(2))
                    train_srcc = float(match.group(3))
                    test_srcc = float(match.group(4))
                    test_plcc = float(match.group(5))
                    
                    epochs.append(epoch)
                    train_losses.append(train_loss)
                    train_srccs.append(train_srcc)
                    test_srccs.append(test_srcc)
                    test_plccs.append(test_plcc)
    
    return {
        'epochs': epochs,
        'train_loss': train_losses,
        'train_srcc': train_srccs,
        'test_srcc': test_srccs,
        'test_plcc': test_plccs
    }

def plot_training_curves(data, output_path='training_curves.png', title='Training Curves'):
    """绘制训练曲线"""
    epochs = data['epochs']
    
    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Train Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, data['train_loss'], 'b-o', linewidth=2, markersize=6, label='Train Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # 标注最小值
    min_idx = np.argmin(data['train_loss'])
    ax1.annotate(f'Min: {data["train_loss"][min_idx]:.4f}\n(Epoch {epochs[min_idx]})',
                xy=(epochs[min_idx], data['train_loss'][min_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 2. Test SRCC
    ax2 = axes[0, 1]
    ax2.plot(epochs, data['test_srcc'], 'r-s', linewidth=2, markersize=6, label='Test SRCC')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('SRCC', fontsize=12)
    ax2.set_title('Test SRCC', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # 标注最大值
    max_idx = np.argmax(data['test_srcc'])
    ax2.annotate(f'Best: {data["test_srcc"][max_idx]:.4f}\n(Epoch {epochs[max_idx]})',
                xy=(epochs[max_idx], data['test_srcc'][max_idx]),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 3. Test PLCC
    ax3 = axes[1, 0]
    ax3.plot(epochs, data['test_plcc'], 'g-^', linewidth=2, markersize=6, label='Test PLCC')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('PLCC', fontsize=12)
    ax3.set_title('Test PLCC', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    
    # 标注最大值
    max_idx = np.argmax(data['test_plcc'])
    ax3.annotate(f'Best: {data["test_plcc"][max_idx]:.4f}\n(Epoch {epochs[max_idx]})',
                xy=(epochs[max_idx], data['test_plcc'][max_idx]),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 4. Train SRCC vs Test SRCC (观察过拟合)
    ax4 = axes[1, 1]
    ax4.plot(epochs, data['train_srcc'], 'm-o', linewidth=2, markersize=6, label='Train SRCC', alpha=0.7)
    ax4.plot(epochs, data['test_srcc'], 'c-s', linewidth=2, markersize=6, label='Test SRCC', alpha=0.7)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('SRCC', fontsize=12)
    ax4.set_title('Train vs Test SRCC (Overfitting Check)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    
    # 计算gap
    gap = np.array(data['train_srcc']) - np.array(data['test_srcc'])
    final_gap = gap[-1]
    ax4.text(0.5, 0.05, f'Final Gap: {final_gap:.4f}', 
             transform=ax4.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training curves saved to: {output_path}")
    
    # 打印统计信息
    print("\n" + "="*60)
    print("📊 TRAINING STATISTICS")
    print("="*60)
    print(f"Total Epochs: {len(epochs)}")
    print(f"\n🔵 Train Loss:")
    print(f"  Initial: {data['train_loss'][0]:.4f}")
    print(f"  Final:   {data['train_loss'][-1]:.4f}")
    print(f"  Min:     {min(data['train_loss']):.4f} (Epoch {epochs[np.argmin(data['train_loss'])]})")
    
    print(f"\n🔴 Test SRCC:")
    print(f"  Initial: {data['test_srcc'][0]:.4f}")
    print(f"  Final:   {data['test_srcc'][-1]:.4f}")
    print(f"  Best:    {max(data['test_srcc']):.4f} (Epoch {epochs[np.argmax(data['test_srcc'])]})")
    
    print(f"\n🟢 Test PLCC:")
    print(f"  Initial: {data['test_plcc'][0]:.4f}")
    print(f"  Final:   {data['test_plcc'][-1]:.4f}")
    print(f"  Best:    {max(data['test_plcc']):.4f} (Epoch {epochs[np.argmax(data['test_plcc'])]})")
    
    print(f"\n🟣 Overfitting Analysis:")
    print(f"  Train-Test Gap (Initial): {data['train_srcc'][0] - data['test_srcc'][0]:.4f}")
    print(f"  Train-Test Gap (Final):   {data['train_srcc'][-1] - data['test_srcc'][-1]:.4f}")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Plot training curves from log file')
    parser.add_argument('--log_file', type=str, 
                        default='logs/batch1_gpu1_lr5e7_20251223_002208.log',
                        help='Path to log file')
    parser.add_argument('--output', type=str, default='training_curves_best_model.png',
                        help='Output image path')
    parser.add_argument('--title', type=str, default='Best Model Training Curves (LR=5e-7, 10 Epochs)',
                        help='Plot title')
    parser.add_argument('--round', type=int, default=1, dest='round_num',
                        help='Which training round to plot (default: 1)')
    
    args = parser.parse_args()
    
    # 检查日志文件是否存在
    if not Path(args.log_file).exists():
        print(f"❌ Error: Log file not found: {args.log_file}")
        return
    
    print(f"📖 Reading log file: {args.log_file}")
    print(f"📍 Extracting Round {args.round_num} data...")
    data = extract_training_data(args.log_file, round_num=args.round_num)
    
    if not data['epochs']:
        print(f"❌ Error: No training data found for Round {args.round_num} in log file!")
        return
    
    print(f"✅ Found {len(data['epochs'])} epochs of training data")
    
    # 绘制曲线
    plot_training_curves(data, args.output, args.title)

if __name__ == '__main__':
    main()

