#!/usr/bin/env python3
"""
生成SMART-IQA论文的额外图表
包括: 训练曲线、学习率敏感度、模型大小对比、消融实验随epoch变化
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置样式
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

output_dir = 'paper_figures'
os.makedirs(output_dir, exist_ok=True)

def parse_log_file(log_path):
    """解析日志文件，提取epoch、loss、SRCC、PLCC"""
    epochs = []
    train_losses = []
    val_losses = []
    srccs = []
    plccs = []
    
    with open(log_path, 'r') as f:
        content = f.read()
        
        # 匹配epoch信息
        # 格式: Epoch [X/Y] - Train Loss: X.XXXX | Val Loss: X.XXXX | SRCC: X.XXXX | PLCC: X.XXXX
        pattern = r'Epoch \[(\d+)/\d+\].*?Train Loss: ([\d.]+).*?Val Loss: ([\d.]+).*?SRCC: ([\d.]+).*?PLCC: ([\d.]+)'
        matches = re.findall(pattern, content)
        
        for match in matches:
            epoch, train_loss, val_loss, srcc, plcc = match
            epochs.append(int(epoch))
            train_losses.append(float(train_loss))
            val_losses.append(float(val_loss))
            srccs.append(float(srcc))
            plccs.append(float(plcc))
    
    return {
        'epochs': np.array(epochs),
        'train_losses': np.array(train_losses),
        'val_losses': np.array(val_losses),
        'srccs': np.array(srccs),
        'plccs': np.array(plccs)
    }

print("=" * 80)
print("开始生成额外的论文图表")
print("=" * 80)

# ============================================================================
# 图1: 主实验（最佳模型）的训练曲线
# ============================================================================
print("\n[1/5] 生成主实验训练曲线...")

try:
    # 最佳模型的日志
    best_model_log = 'logs/batch1_gpu0_lr5e7_20251223_002208.log'
    
    if os.path.exists(best_model_log):
        data = parse_log_file(best_model_log)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 子图1: Training and Validation Loss
        ax1 = axes[0]
        ax1.plot(data['epochs'], data['train_losses'], 'o-', linewidth=2.5, 
                markersize=8, color='#4ECDC4', label='Train Loss')
        ax1.plot(data['epochs'], data['val_losses'], 's-', linewidth=2.5, 
                markersize=8, color='#FF6B6B', label='Val Loss')
        ax1.set_xlabel('Epoch', fontsize=13, weight='bold')
        ax1.set_ylabel('Loss (L1)', fontsize=13, weight='bold')
        ax1.set_title('Training and Validation Loss', fontsize=14, weight='bold')
        ax1.legend(fontsize=11, frameon=True, shadow=True)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 子图2: SRCC
        ax2 = axes[1]
        ax2.plot(data['epochs'], data['srccs'], 'D-', linewidth=2.5, 
                markersize=8, color='#95E1D3', markerfacecolor='#FF6B6B', 
                markeredgecolor='black', markeredgewidth=1.5)
        ax2.set_xlabel('Epoch', fontsize=13, weight='bold')
        ax2.set_ylabel('SRCC', fontsize=13, weight='bold')
        ax2.set_title('Validation SRCC', fontsize=14, weight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim([0.90, 0.94])
        
        # 标注最佳值
        best_idx = np.argmax(data['srccs'])
        best_srcc = data['srccs'][best_idx]
        best_epoch = data['epochs'][best_idx]
        ax2.scatter([best_epoch], [best_srcc], s=500, c='gold', marker='*', 
                   edgecolors='black', linewidths=2, zorder=5)
        ax2.annotate(f'Best: {best_srcc:.4f}\nEpoch {best_epoch}', 
                    xy=(best_epoch, best_srcc), 
                    xytext=(best_epoch + 0.5, best_srcc - 0.01),
                    fontsize=11, weight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        
        # 子图3: PLCC
        ax3 = axes[2]
        ax3.plot(data['epochs'], data['plccs'], 'o-', linewidth=2.5, 
                markersize=8, color='#F38181', markeredgecolor='black', 
                markeredgewidth=1.5)
        ax3.set_xlabel('Epoch', fontsize=13, weight='bold')
        ax3.set_ylabel('PLCC', fontsize=13, weight='bold')
        ax3.set_title('Validation PLCC', fontsize=14, weight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_ylim([0.90, 0.95])
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/main_training_curves.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/main_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 已保存: {output_dir}/main_training_curves.pdf/.png")
    else:
        print(f"  ⚠ 警告: 找不到日志文件 {best_model_log}")
except Exception as e:
    print(f"  ✗ 错误: {e}")

# ============================================================================
# 图2: 学习率敏感度（ylim范围更大）
# ============================================================================
print("\n[2/5] 生成学习率敏感度曲线...")

try:
    # 学习率数据（从EXPERIMENTS_LOG_TRACKER.md）
    lr_data = {
        '5e-6': {'srcc': 0.9354, 'plcc': 0.9448, 'epochs': 5},
        '3e-6': {'srcc': 0.9364, 'plcc': 0.9464, 'epochs': 5},
        '1e-6': {'srcc': 0.9374, 'plcc': 0.9485, 'epochs': 10},
        '5e-7': {'srcc': 0.9378, 'plcc': 0.9485, 'epochs': 10},  # Best
        '1e-7': {'srcc': 0.9375, 'plcc': 0.9488, 'epochs': 14},
    }
    
    lr_values = [5e-6, 3e-6, 1e-6, 5e-7, 1e-7]
    srcc_values = [lr_data[str(lr)]['srcc'] for lr in lr_values]
    plcc_values = [lr_data[str(lr)]['plcc'] for lr in lr_values]
    epochs_values = [lr_data[str(lr)]['epochs'] for lr in lr_values]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 图1: SRCC vs Learning Rate (扩大y轴范围)
    ax1.plot(lr_values, srcc_values, 'o-', linewidth=2.5, markersize=10, 
             color='#4ECDC4', markerfacecolor='#FF6B6B', 
             markeredgecolor='black', markeredgewidth=2)
    
    # 标记最优点
    best_idx = srcc_values.index(max(srcc_values))
    ax1.scatter([lr_values[best_idx]], [srcc_values[best_idx]], 
                s=500, c='gold', marker='*', edgecolors='black', 
                linewidths=2, zorder=5)
    
    # 标注数值
    for i, (lr, srcc) in enumerate(zip(lr_values, srcc_values)):
        offset_y = 0.001 if i != best_idx else 0.0015
        ax1.annotate(f'{srcc:.4f}', (lr, srcc), 
                     xytext=(0, offset_y), textcoords='offset points',
                     ha='center', fontsize=10, weight='bold')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Learning Rate', fontsize=13, weight='bold')
    ax1.set_ylabel('SRCC on KonIQ-10k', fontsize=13, weight='bold')
    ax1.set_title('Learning Rate Sensitivity Analysis', fontsize=13, weight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0.930, 0.940])  # 扩大范围，看起来更稳定
    
    # 添加最优区间阴影
    ax1.axvspan(3e-7, 1e-6, alpha=0.1, color='green', label='Optimal Range')
    ax1.legend(fontsize=11)
    
    # 图2: Training Epochs vs Learning Rate
    colors = ['#FF6B6B' if e == min(epochs_values) else '#95E1D3' for e in epochs_values]
    bars = ax2.bar(range(len(lr_values)), epochs_values, color=colors, 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # 标注最优LR
    bars[best_idx].set_facecolor('gold')
    
    ax2.set_xticks(range(len(lr_values)))
    ax2.set_xticklabels([f'{lr:.0e}' for lr in lr_values], fontsize=10)
    ax2.set_xlabel('Learning Rate', fontsize=13, weight='bold')
    ax2.set_ylabel('Epochs to Converge', fontsize=13, weight='bold')
    ax2.set_title('Training Efficiency vs Learning Rate', fontsize=13, weight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 标注数值
    for i, (lr, epoch) in enumerate(zip(lr_values, epochs_values)):
        ax2.text(i, epoch + 0.3, str(epoch), ha='center', 
                 fontsize=11, weight='bold')
    
    # 添加说明文字
    info_text = (
        "Key Finding: Swin Transformer requires 200× smaller LR than ResNet50\n"
        "Optimal LR: 5e-7 achieves best SRCC (0.9378) with moderate training time (10 epochs)"
    )
    plt.figtext(0.5, 0.02, info_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/lr_sensitivity_extended.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/lr_sensitivity_extended.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 已保存: {output_dir}/lr_sensitivity_extended.pdf/.png")
except Exception as e:
    print(f"  ✗ 错误: {e}")

# ============================================================================
# 图3: 不同模型大小的性能对比
# ============================================================================
print("\n[3/5] 生成模型大小对比图...")

try:
    # 数据
    models = ['HyperIQA\n(ResNet50)', 'SMART-IQA\nTiny', 'SMART-IQA\nSmall', 'SMART-IQA\nBase']
    params = [25, 28, 50, 88]  # millions
    srcc = [0.9070, 0.9249, 0.9338, 0.9378]
    plcc = [0.9180, 0.9360, 0.9455, 0.9485]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 图1: SRCC对比
    colors = ['red', 'skyblue', 'lightgreen', 'gold']
    bars1 = ax1.bar(range(len(models)), srcc, color=colors, 
                    edgecolor='black', linewidth=2, alpha=0.8)
    
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, fontsize=11, weight='bold')
    ax1.set_ylabel('SRCC on KonIQ-10k', fontsize=13, weight='bold')
    ax1.set_title('Performance Comparison', fontsize=14, weight='bold')
    ax1.set_ylim([0.900, 0.945])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 标注数值和提升
    for i, (model, s) in enumerate(zip(models, srcc)):
        ax1.text(i, s + 0.002, f'{s:.4f}', ha='center', fontsize=11, weight='bold')
        if i > 0:
            improvement = (s - srcc[0]) / srcc[0] * 100
            ax1.text(i, s - 0.015, f'+{improvement:.2f}%', ha='center', 
                    fontsize=9, color='green' if improvement > 0 else 'red', weight='bold')
    
    # 添加基线
    ax1.axhline(y=srcc[0], color='red', linestyle='--', linewidth=2, alpha=0.5, label='Baseline')
    ax1.legend(fontsize=11)
    
    # 图2: 参数量 vs SRCC 散点图
    for i, (param, s, model) in enumerate(zip(params, srcc, models)):
        if 'SMART-IQA' in model:
            marker = 'D'
            size = 300
            edgewidth = 2.5
            color = colors[i]
        else:
            marker = 'o'
            size = 200
            edgewidth = 1.5
            color = colors[i]
        
        ax2.scatter(param, s, s=size, c=color, marker=marker, 
                   alpha=0.8, edgecolors='black', linewidths=edgewidth,
                   label=model.replace('\n', ' '))
        
        # 标注
        offset_x = -5 if 'SMART' in model else 5
        offset_y = 0.001 if param != 88 else -0.002
        ax2.annotate(f'{param}M\n{s:.4f}', (param, s), 
                    xytext=(offset_x, offset_y), textcoords='offset points',
                    fontsize=9, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', 
                             facecolor='yellow' if 'Base' in model else 'white',
                             alpha=0.7, edgecolor='black'))
    
    # 连线显示进化路径
    ax2.plot(params, srcc, 'g--', linewidth=2, alpha=0.5, label='Evolution Path')
    
    ax2.set_xlabel('Parameters (Millions)', fontsize=13, weight='bold')
    ax2.set_ylabel('SRCC on KonIQ-10k', fontsize=13, weight='bold')
    ax2.set_title('Performance vs Model Size', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim([20, 95])
    ax2.set_ylim([0.900, 0.945])
    ax2.legend(fontsize=9, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_size_performance.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/model_size_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 已保存: {output_dir}/model_size_performance.pdf/.png")
except Exception as e:
    print(f"  ✗ 错误: {e}")

# ============================================================================
# 图4: 消融实验的test SRCC随epoch变化（需要ResNet50基线数据）
# ============================================================================
print("\n[4/5] 生成消融实验SRCC变化曲线...")

try:
    # 定义消融实验的日志文件
    ablation_logs = {
        'Baseline\n(ResNet50)': 'logs/resnet50_baseline_20251220_213606.log',
        'Swin Only': 'logs/A2_no_multiscale_lr5e7_20251223_092034.log',
        'Swin+MultiScale': 'logs/A1_no_attention_lr5e7_20251223_092034.log',
        'Full Model': 'logs/batch1_gpu0_lr5e7_20251223_002208.log',
    }
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFD93D']
    markers = ['o', 's', 'D', '^']
    
    all_data_found = True
    
    for i, (name, log_path) in enumerate(ablation_logs.items()):
        if os.path.exists(log_path):
            data = parse_log_file(log_path)
            if len(data['epochs']) > 0:
                ax.plot(data['epochs'], data['srccs'], 
                       marker=markers[i], linestyle='-', linewidth=2.5, 
                       markersize=8, color=colors[i], 
                       markeredgecolor='black', markeredgewidth=1.5,
                       label=name, alpha=0.9)
                
                # 标注最终值
                final_srcc = data['srccs'][-1]
                final_epoch = data['epochs'][-1]
                ax.annotate(f'{final_srcc:.4f}', 
                           xy=(final_epoch, final_srcc),
                           xytext=(5, 0), textcoords='offset points',
                           fontsize=9, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor=colors[i], alpha=0.5))
            else:
                print(f"  ⚠ 警告: {name} 日志为空")
                all_data_found = False
        else:
            print(f"  ⚠ 警告: 找不到 {name} 日志: {log_path}")
            all_data_found = False
    
    ax.set_xlabel('Epoch', fontsize=13, weight='bold')
    ax.set_ylabel('Test SRCC', fontsize=13, weight='bold')
    ax.set_title('Ablation Study: SRCC Evolution During Training', fontsize=14, weight='bold')
    ax.legend(fontsize=11, loc='lower right', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0.88, 0.94])
    
    # 添加说明
    if all_data_found:
        status_text = "All ablation experiments completed"
    else:
        status_text = "⚠ Some ablation experiments missing - may need to run baseline"
    
    plt.figtext(0.5, 0.02, status_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', 
                         facecolor='lightgreen' if all_data_found else 'yellow', 
                         alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ablation_srcc_evolution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/ablation_srcc_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 已保存: {output_dir}/ablation_srcc_evolution.pdf/.png")
    
    if not all_data_found:
        print("  ⚠ 注意: 可能需要运行ResNet50基线实验来补充数据")
except Exception as e:
    print(f"  ✗ 错误: {e}")

# ============================================================================
# 图5: 生成训练曲线对比（所有关键实验）
# ============================================================================
print("\n[5/5] 生成综合训练曲线对比...")

try:
    key_experiments = {
        'Best Model (Base)': 'logs/batch1_gpu0_lr5e7_20251223_002208.log',
        'Small Model': 'logs/B2_small_lr5e7_20251223_092506.log',
        'Tiny Model': 'logs/B1_tiny_lr5e7_20251223_092506.log',
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['#FFD93D', '#95E1D3', '#4ECDC4']
    markers = ['o', 's', 'D']
    
    for i, (name, log_path) in enumerate(key_experiments.items()):
        if os.path.exists(log_path):
            data = parse_log_file(log_path)
            if len(data['epochs']) > 0:
                # SRCC
                axes[0].plot(data['epochs'], data['srccs'], 
                           marker=markers[i], linestyle='-', linewidth=2.5, 
                           markersize=8, color=colors[i], 
                           markeredgecolor='black', markeredgewidth=1.5,
                           label=name, alpha=0.9)
                
                # Loss
                axes[1].plot(data['epochs'], data['val_losses'], 
                           marker=markers[i], linestyle='-', linewidth=2.5, 
                           markersize=8, color=colors[i], 
                           markeredgecolor='black', markeredgewidth=1.5,
                           label=name, alpha=0.9)
    
    # SRCC图
    axes[0].set_xlabel('Epoch', fontsize=13, weight='bold')
    axes[0].set_ylabel('Test SRCC', fontsize=13, weight='bold')
    axes[0].set_title('SRCC Comparison Across Model Sizes', fontsize=14, weight='bold')
    axes[0].legend(fontsize=11, loc='lower right')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_ylim([0.90, 0.94])
    
    # Loss图
    axes[1].set_xlabel('Epoch', fontsize=13, weight='bold')
    axes[1].set_ylabel('Validation Loss (L1)', fontsize=13, weight='bold')
    axes[1].set_title('Loss Comparison Across Model Sizes', fontsize=14, weight='bold')
    axes[1].legend(fontsize=11, loc='upper right')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comprehensive_training_curves.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/comprehensive_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 已保存: {output_dir}/comprehensive_training_curves.pdf/.png")
except Exception as e:
    print(f"  ✗ 错误: {e}")

print("\n" + "=" * 80)
print("✅ 所有额外图表生成完成！")
print("=" * 80)
print(f"\n输出目录: {output_dir}/")
print("\n生成的文件:")
print("  1. main_training_curves.pdf/.png         - 主实验训练曲线（Loss+SRCC+PLCC）")
print("  2. lr_sensitivity_extended.pdf/.png      - 学习率敏感度（扩大y轴范围）")
print("  3. model_size_performance.pdf/.png       - 不同模型大小对比")
print("  4. ablation_srcc_evolution.pdf/.png      - 消融实验SRCC变化")
print("  5. comprehensive_training_curves.pdf/.png - 综合训练曲线对比")
print("\n⚠ 注意: 如果消融实验图中缺少ResNet50基线，可能需要补充实验")
print("=" * 80)

