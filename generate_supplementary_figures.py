#!/usr/bin/env python3
"""
生成补充图表
"""

import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

output_dir = 'paper_figures'
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("生成补充图表")
print("=" * 80)

# ============================================================================
# 图1: 学习率敏感度
# ============================================================================
print("\n[1/3] 生成学习率敏感度图...")

lr_data = {
    '1e-4': {'srcc': 0.7520, 'plcc': 0.7820, 'epochs': 8, 'status': 'Unstable'},
    '1e-5': {'srcc': 0.8950, 'plcc': 0.9050, 'epochs': 9, 'status': 'Too Fast'},
    '1e-6': {'srcc': 0.9310, 'plcc': 0.9425, 'epochs': 10, 'status': 'Good'},
    '5e-7': {'srcc': 0.9378, 'plcc': 0.9485, 'epochs': 10, 'status': 'Optimal'},
    '1e-7': {'srcc': 0.9350, 'plcc': 0.9455, 'epochs': 12, 'status': 'Too Slow'},
}

lrs = ['1e-4', '1e-5', '1e-6', '5e-7', '1e-7']
lrs_numeric = [1e-4, 1e-5, 1e-6, 5e-7, 1e-7]
srccs = [lr_data[lr]['srcc'] for lr in lrs]
plccs = [lr_data[lr]['plcc'] for lr in lrs]
epochs_to_converge = [lr_data[lr]['epochs'] for lr in lrs]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 左图: SRCC vs LR
colors_lr = ['red' if lr_data[lr]['status'] == 'Unstable' else 
             'orange' if lr_data[lr]['status'] in ['Too Fast', 'Too Slow'] else 
             'lightgreen' if lr_data[lr]['status'] == 'Good' else 'gold' 
             for lr in lrs]

ax1.plot(range(len(lrs)), srccs, 'o-', linewidth=2.5, markersize=10, 
        color='#4ECDC4', markeredgecolor='black', markeredgewidth=1.5)

for i, (lr, srcc_val, color) in enumerate(zip(lrs, srccs, colors_lr)):
    ax1.scatter([i], [srcc_val], s=300, c=color, edgecolors='black', 
               linewidths=2, zorder=5)

# 标注最佳值
best_idx = 3  # 5e-7
ax1.scatter([best_idx], [srccs[best_idx]], s=800, c='gold', marker='*', 
           edgecolors='black', linewidths=3, zorder=10)
ax1.annotate(f'Optimal\n{srccs[best_idx]:.4f}', 
            xy=(best_idx, srccs[best_idx]), 
            xytext=(best_idx, srccs[best_idx] + 0.06),
            fontsize=11, weight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='red'))

ax1.set_xticks(range(len(lrs)))
ax1.set_xticklabels(lrs, fontsize=11, weight='bold')
ax1.set_xlabel('Learning Rate', fontsize=13, weight='bold')
ax1.set_ylabel('SRCC on KonIQ-10k', fontsize=13, weight='bold')
ax1.set_title('Learning Rate Sensitivity: SRCC', fontsize=14, weight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim([0.73, 0.95])

# 右图: Training Efficiency
ax2.bar(range(len(lrs)), epochs_to_converge, color=colors_lr, 
       edgecolor='black', linewidth=2, alpha=0.8, width=0.6)

ax2.set_xticks(range(len(lrs)))
ax2.set_xticklabels(lrs, fontsize=11, weight='bold')
ax2.set_xlabel('Learning Rate', fontsize=13, weight='bold')
ax2.set_ylabel('Epochs to Converge', fontsize=13, weight='bold')
ax2.set_title('Training Efficiency', fontsize=14, weight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

for i, (lr, epochs) in enumerate(zip(lrs, epochs_to_converge)):
    ax2.text(i, epochs + 0.3, f'{epochs}', ha='center', 
            fontsize=11, weight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/lr_sensitivity_final.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/lr_sensitivity_final.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ 已保存: {output_dir}/lr_sensitivity_final.pdf/.png")

# ============================================================================
# 图2: 损失函数对比
# ============================================================================
print("\n[2/3] 生成损失函数对比图...")

loss_data = {
    'L1\n(MAE)': {'srcc': 0.9375, 'plcc': 0.9488, 'color': '#FFD93D'},
    'L2\n(MSE)': {'srcc': 0.9373, 'plcc': 0.9469, 'color': '#4ECDC4'},
    'Pairwise\nFidelity': {'srcc': 0.9315, 'plcc': 0.9373, 'color': '#95E1D3'},
    'SRCC\nLoss': {'srcc': 0.9313, 'plcc': 0.9416, 'color': '#F38181'},
    'Pairwise\nRanking': {'srcc': 0.9292, 'plcc': 0.9249, 'color': '#FF6B6B'},
}

loss_names = list(loss_data.keys())
loss_srccs = [loss_data[name]['srcc'] for name in loss_names]
loss_plccs = [loss_data[name]['plcc'] for name in loss_names]
colors_loss = [loss_data[name]['color'] for name in loss_names]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 左图: SRCC对比
bars1 = ax1.bar(range(len(loss_names)), loss_srccs, color=colors_loss,
               edgecolor='black', linewidth=2, alpha=0.8, width=0.6)

ax1.set_xticks(range(len(loss_names)))
ax1.set_xticklabels(loss_names, fontsize=10, weight='bold')
ax1.set_ylabel('SRCC on KonIQ-10k', fontsize=13, weight='bold')
ax1.set_title('Loss Function Comparison: SRCC', fontsize=14, weight='bold')
ax1.set_ylim([0.925, 0.940])
ax1.grid(axis='y', alpha=0.3, linestyle='--')

for i, (name, srcc_val) in enumerate(zip(loss_names, loss_srccs)):
    ax1.text(i, srcc_val + 0.0008, f'{srcc_val:.4f}', ha='center', 
            fontsize=9, weight='bold')

# 标注最佳
ax1.scatter([0], [loss_srccs[0]], s=800, c='gold', marker='*', 
           edgecolors='black', linewidths=3, zorder=10)

# 右图: SRCC vs PLCC scatter
ax2.scatter(loss_srccs, loss_plccs, s=400, c=colors_loss, 
           edgecolors='black', linewidths=2.5, alpha=0.8, zorder=5)

for i, name in enumerate(loss_names):
    name_clean = name.replace('\n', ' ')
    ax2.annotate(name_clean, 
                xy=(loss_srccs[i], loss_plccs[i]), 
                xytext=(loss_srccs[i] + 0.0015, loss_plccs[i] + 0.004),
                fontsize=9, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors_loss[i], alpha=0.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

ax2.set_xlabel('SRCC', fontsize=13, weight='bold')
ax2.set_ylabel('PLCC', fontsize=13, weight='bold')
ax2.set_title('SRCC vs PLCC Consistency', fontsize=14, weight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(f'{output_dir}/loss_function_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/loss_function_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ 已保存: {output_dir}/loss_function_comparison.pdf/.png")

# ============================================================================
# 图3: 模型大小对比
# ============================================================================
print("\n[3/3] 生成模型大小对比图...")

model_data = {
    'HyperIQA\n(ResNet50)': {'params': 25, 'srcc': 0.9070, 'plcc': 0.9180, 'color': '#FF6B6B'},
    'SMART-IQA\nTiny': {'params': 28, 'srcc': 0.9249, 'plcc': 0.9360, 'color': '#4ECDC4'},
    'SMART-IQA\nSmall': {'params': 50, 'srcc': 0.9338, 'plcc': 0.9455, 'color': '#95E1D3'},
    'SMART-IQA\nBase': {'params': 88, 'srcc': 0.9378, 'plcc': 0.9485, 'color': '#FFD93D'},
}

model_names = list(model_data.keys())
model_params = [model_data[name]['params'] for name in model_names]
model_srccs = [model_data[name]['srcc'] for name in model_names]
model_plccs = [model_data[name]['plcc'] for name in model_names]
colors_model = [model_data[name]['color'] for name in model_names]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 左图: SRCC对比
bars1 = ax1.bar(range(len(model_names)), model_srccs, color=colors_model,
               edgecolor='black', linewidth=2, alpha=0.8, width=0.6)

ax1.set_xticks(range(len(model_names)))
ax1.set_xticklabels(model_names, fontsize=10, weight='bold')
ax1.set_ylabel('SRCC on KonIQ-10k', fontsize=13, weight='bold')
ax1.set_title('Model Size Comparison: Performance', fontsize=14, weight='bold')
ax1.set_ylim([0.900, 0.945])
ax1.grid(axis='y', alpha=0.3, linestyle='--')

for i, (name, srcc_val) in enumerate(zip(model_names, model_srccs)):
    ax1.text(i, srcc_val + 0.002, f'{srcc_val:.4f}', ha='center', 
            fontsize=10, weight='bold')

ax1.axhline(y=model_srccs[0], color='red', linestyle='--', linewidth=2, alpha=0.5,
           label='HyperIQA Baseline')
ax1.legend(fontsize=10)

# 右图: Parameters vs SRCC scatter
ax2.scatter(model_params, model_srccs, s=500, c=colors_model, 
           edgecolors='black', linewidths=2.5, alpha=0.8, zorder=5)

# 连接演化路径
ax2.plot(model_params[1:], model_srccs[1:], 'k--', linewidth=2, alpha=0.3, zorder=1)

for i, name in enumerate(model_names):
    name_clean = name.replace('\n', ' ')
    offset_x = 5 if i < 2 else -12
    offset_y = -0.004 if i == 0 else 0.004
    ax2.annotate(name_clean, 
                xy=(model_params[i], model_srccs[i]), 
                xytext=(model_params[i] + offset_x, model_srccs[i] + offset_y),
                fontsize=9, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors_model[i], alpha=0.7),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

ax2.set_xlabel('Parameters (M)', fontsize=13, weight='bold')
ax2.set_ylabel('SRCC', fontsize=13, weight='bold')
ax2.set_title('Performance vs Model Size', fontsize=14, weight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(f'{output_dir}/model_size_final.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/model_size_final.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ 已保存: {output_dir}/model_size_final.pdf/.png")

print("\n" + "=" * 80)
print("✅ 补充图表生成完成！")
print("=" * 80)

