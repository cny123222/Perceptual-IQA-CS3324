#!/usr/bin/env python3
"""
使用真实训练数据生成训练曲线图
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# 使用Times字体（serif系列）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'  # STIX fonts similar to Times

output_dir = 'paper_figures'
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("使用真实训练数据生成训练曲线")
print("=" * 80)

# 读取真实数据
df = pd.read_csv('training_data_real.csv')

# 只保留前10个epoch（最后一个是重复的）
df = df.head(10)

epochs = df['epoch'].values
train_losses = df['train_loss'].values
train_srccs = df['train_srcc'].values
test_srccs = df['test_srcc'].values
test_plccs = df['test_plcc'].values

print(f"\n✓ 读取了 {len(epochs)} 个epoch的数据")
print(f"✓ Epoch范围: {epochs[0]} - {epochs[-1]}")
print(f"✓ 最佳Test SRCC: {max(test_srccs):.4f} (Epoch {epochs[np.argmax(test_srccs)]})")
print(f"✓ 最佳Test PLCC: {max(test_plccs):.4f} (Epoch {epochs[np.argmax(test_plccs)]})")

# 生成训练曲线图（三子图）
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 子图1: Loss
ax1 = axes[0]
ax1.plot(epochs, train_losses, 'o-', linewidth=2.5, markersize=8, 
        color='#4ECDC4', markeredgecolor='black', markeredgewidth=1.5)
ax1.set_xlabel('Epoch', fontsize=13, weight='bold')
ax1.set_ylabel('Loss (L1)', fontsize=13, weight='bold')
ax1.set_title('Training Loss', fontsize=14, weight='bold')
# No legend - as per user request
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xticks(epochs)

# 子图2: SRCC
ax2 = axes[1]
ax2.plot(epochs, test_srccs, 'D-', linewidth=2.5, markersize=8, 
        color='#95E1D3', markerfacecolor='#FF6B6B', 
        markeredgecolor='black', markeredgewidth=1.5)
ax2.set_xlabel('Epoch', fontsize=13, weight='bold')
ax2.set_ylabel('SRCC', fontsize=13, weight='bold')
ax2.set_title('Validation SRCC', fontsize=14, weight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xticks(epochs)

# 标注最佳值
best_idx = np.argmax(test_srccs)
ax2.scatter([epochs[best_idx]], [test_srccs[best_idx]], s=500, c='gold', 
           marker='*', edgecolors='black', linewidths=2, zorder=5)
ax2.annotate(f'Best: {test_srccs[best_idx]:.4f}\nEpoch {epochs[best_idx]}', 
            xy=(epochs[best_idx], test_srccs[best_idx]), 
            xytext=(epochs[best_idx] + 0.8, test_srccs[best_idx] - 0.008),
            fontsize=10, weight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', lw=2, color='red'))

# 子图3: PLCC
ax3 = axes[2]
ax3.plot(epochs, test_plccs, 'o-', linewidth=2.5, markersize=8, 
        color='#F38181', markeredgecolor='black', markeredgewidth=1.5)
ax3.set_xlabel('Epoch', fontsize=13, weight='bold')
ax3.set_ylabel('PLCC', fontsize=13, weight='bold')
ax3.set_title('Validation PLCC', fontsize=14, weight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_xticks(epochs)

# 标注最佳值
best_plcc_idx = np.argmax(test_plccs)
ax3.scatter([epochs[best_plcc_idx]], [test_plccs[best_plcc_idx]], s=500, c='gold', 
           marker='*', edgecolors='black', linewidths=2, zorder=5)

plt.tight_layout()
plt.savefig(f'{output_dir}/main_training_curves_real.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/main_training_curves_real.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ 已保存: {output_dir}/main_training_curves_real.pdf/.png")

# 生成详细的4子图版本
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图1: Train Loss
ax = axes[0, 0]
ax.plot(epochs, train_losses, 'o-', linewidth=2.5, markersize=10, 
       color='#4ECDC4', markeredgecolor='black', markeredgewidth=1.5)
ax.set_xlabel('Epoch', fontsize=12, weight='bold')
ax.set_ylabel('Train Loss (L1)', fontsize=12, weight='bold')
ax.set_title('Training Loss Convergence', fontsize=13, weight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(epochs)
for i, (e, loss) in enumerate(zip(epochs, train_losses)):
    if i % 2 == 0:  # 每2个epoch标注一次
        ax.text(e, loss + 0.3, f'{loss:.2f}', ha='center', fontsize=9, weight='bold')

# 子图2: Test SRCC
ax = axes[0, 1]
ax.plot(epochs, test_srccs, 'D-', linewidth=2.5, markersize=10, 
       color='#95E1D3', markerfacecolor='#FF6B6B', 
       markeredgecolor='black', markeredgewidth=1.5)
ax.set_xlabel('Epoch', fontsize=12, weight='bold')
ax.set_ylabel('Test SRCC', fontsize=12, weight='bold')
ax.set_title('Validation SRCC Progression', fontsize=13, weight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(epochs)
best_idx = np.argmax(test_srccs)
ax.scatter([epochs[best_idx]], [test_srccs[best_idx]], s=600, c='gold', 
          marker='*', edgecolors='black', linewidths=3, zorder=10)

# 子图3: Test PLCC
ax = axes[1, 0]
ax.plot(epochs, test_plccs, 'o-', linewidth=2.5, markersize=10, 
       color='#F38181', markeredgecolor='black', markeredgewidth=1.5)
ax.set_xlabel('Epoch', fontsize=12, weight='bold')
ax.set_ylabel('Test PLCC', fontsize=12, weight='bold')
ax.set_title('Validation PLCC Progression', fontsize=13, weight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(epochs)
best_plcc_idx = np.argmax(test_plccs)
ax.scatter([epochs[best_plcc_idx]], [test_plccs[best_plcc_idx]], s=600, c='gold', 
          marker='*', edgecolors='black', linewidths=3, zorder=10)

# 子图4: Train vs Test SRCC
ax = axes[1, 1]
ax.plot(epochs, train_srccs, 's-', linewidth=2.5, markersize=8, 
       color='#4ECDC4', markeredgecolor='black', markeredgewidth=1.5)
ax.plot(epochs, test_srccs, 'D-', linewidth=2.5, markersize=8, 
       color='#FF6B6B', markeredgecolor='black', markeredgewidth=1.5)
ax.set_xlabel('Epoch', fontsize=12, weight='bold')
ax.set_ylabel('SRCC', fontsize=12, weight='bold')
ax.set_title('Train vs Test SRCC', fontsize=13, weight='bold')
# No legend - as per user request
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(epochs)

plt.tight_layout()
plt.savefig(f'{output_dir}/training_curves_detailed_real.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/training_curves_detailed_real.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ 已保存: {output_dir}/training_curves_detailed_real.pdf/.png")

print("\n" + "=" * 80)
print("✅ 真实训练曲线生成完成！")
print("=" * 80)
print(f"\n生成的文件:")
print(f"  1. main_training_curves_real.pdf/.png     - 主图（3子图）论文使用")
print(f"  2. training_curves_detailed_real.pdf/.png - 详细版（4子图）")
print("\n关键数据:")
print(f"  最佳Epoch: {epochs[best_idx]}")
print(f"  Test SRCC: {test_srccs[best_idx]:.4f}")
print(f"  Test PLCC: {test_plccs[best_idx]:.4f}")
print(f"  Train Loss: {train_losses[best_idx]:.4f}")
print("=" * 80)

