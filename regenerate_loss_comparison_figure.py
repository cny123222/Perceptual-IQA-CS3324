#!/usr/bin/env python3
"""
重新生成Loss对比图 - Times New Roman字体，无图例
"""

import matplotlib.pyplot as plt
import numpy as np

# 设置Times New Roman字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.5

# Loss function数据
loss_functions = ['L1\n(MAE)', 'L2\n(MSE)', 'Pairwise\nFidelity', 'SRCC\nLoss', 'Pairwise\nRanking']
srcc_values = [0.9375, 0.9373, 0.9315, 0.9313, 0.9292]
plcc_values = [0.9488, 0.9469, 0.9373, 0.9416, 0.9249]

# 颜色方案
colors = ['#2E7D32', '#1976D2', '#F57C00', '#7B1FA2', '#C62828']

fig = plt.figure(figsize=(14, 5))

# ===== 子图1：SRCC对比 =====
ax1 = plt.subplot(1, 3, 1)

bars1 = ax1.bar(range(len(loss_functions)), srcc_values, 
               color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)

# 标注数值
for i, (bar, val) in enumerate(zip(bars1, srcc_values)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=10, weight='bold',
            fontfamily='Times New Roman')

ax1.set_ylabel('SRCC', fontsize=13, weight='bold', fontfamily='Times New Roman')
ax1.set_title('Loss Function Comparison (SRCC)', fontsize=13, weight='bold', 
             fontfamily='Times New Roman')
ax1.set_xticks(range(len(loss_functions)))
ax1.set_xticklabels(loss_functions, fontsize=10, fontfamily='Times New Roman')
ax1.set_ylim([0.925, 0.950])
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# 添加最佳标注
ax1.text(0, 0.9495, '✓ Best', ha='center', fontsize=10, weight='bold',
        color='#2E7D32', fontfamily='Times New Roman',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#A5D6A7', alpha=0.7))

# ===== 子图2：PLCC对比 =====
ax2 = plt.subplot(1, 3, 2)

bars2 = ax2.bar(range(len(loss_functions)), plcc_values, 
               color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)

# 标注数值
for i, (bar, val) in enumerate(zip(bars2, plcc_values)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=10, weight='bold',
            fontfamily='Times New Roman')

ax2.set_ylabel('PLCC', fontsize=13, weight='bold', fontfamily='Times New Roman')
ax2.set_title('Loss Function Comparison (PLCC)', fontsize=13, weight='bold',
             fontfamily='Times New Roman')
ax2.set_xticks(range(len(loss_functions)))
ax2.set_xticklabels(loss_functions, fontsize=10, fontfamily='Times New Roman')
ax2.set_ylim([0.920, 0.952])
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# 添加最佳标注
ax2.text(0, 0.9505, '✓ Best', ha='center', fontsize=10, weight='bold',
        color='#2E7D32', fontfamily='Times New Roman',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#A5D6A7', alpha=0.7))

# ===== 子图3：SRCC vs PLCC散点图 =====
ax3 = plt.subplot(1, 3, 3)

for i, (loss, srcc, plcc, color) in enumerate(zip(loss_functions, srcc_values, plcc_values, colors)):
    ax3.scatter(srcc, plcc, s=300, c=color, edgecolors='black', 
               linewidth=2, alpha=0.85, zorder=10)
    
    # 标注loss function名称
    label = loss.replace('\n', ' ')
    offset_x = [0.0002, -0.0002, 0.0002, -0.0002, 0.0002][i]
    offset_y = [0.001, -0.002, 0.001, -0.002, 0.001][i]
    ax3.annotate(label, xy=(srcc, plcc), 
                xytext=(srcc + offset_x, plcc + offset_y),
                fontsize=9, ha='center', weight='bold',
                fontfamily='Times New Roman',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         alpha=0.8, edgecolor=color, linewidth=1.5))

ax3.set_xlabel('SRCC', fontsize=13, weight='bold', fontfamily='Times New Roman')
ax3.set_ylabel('PLCC', fontsize=13, weight='bold', fontfamily='Times New Roman')
ax3.set_title('SRCC vs PLCC Scatter Plot', fontsize=13, weight='bold',
             fontfamily='Times New Roman')
ax3.grid(alpha=0.3, linestyle='--')
ax3.set_xlim([0.9285, 0.9382])
ax3.set_ylim([0.924, 0.950])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# 添加对角线参考
ax3.plot([0.92, 0.95], [0.92, 0.95], 'k--', alpha=0.3, linewidth=1, zorder=1)

plt.tight_layout()

# 保存
plt.savefig('paper_figures/loss_function_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper_figures/loss_function_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Loss对比图已生成（Times New Roman字体，无图例）: paper_figures/loss_function_comparison.pdf")

plt.close()

