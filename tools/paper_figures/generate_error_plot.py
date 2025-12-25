"""
生成预测误差分析图
包括：散点图（GT vs Pred）和误差最大的样本
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from PIL import Image
import os
import sys
import json
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from smart_iqa.models import smart_iqa as models
from torchvision import transforms


def load_model(model_path, device='cuda'):
    """加载训练好的模型"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    model = models.HyperNet(
        16, 112, 224, 112, 56, 28, 14, 7,
        use_multiscale=True,
        use_attention=True,
        model_size='base'
    ).to(device)
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("✓ Model loaded\n")
    
    return model, device


def get_predictions(model, device, num_patches=10):
    """
    获取测试集上的预测结果
    
    Returns:
        gt_scores: Ground truth MOS scores
        pred_scores: Predicted scores
        img_names: Image names
    """
    import csv
    
    # 读取数据
    csv_file = 'koniq-10k/koniq10k_scores_and_distributions.csv'
    mos_dict = {}
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mos_dict[row['image_name']] = float(row['MOS_zscore'])
    
    # 读取测试集split
    test_json = 'koniq-10k/koniq_test.json'
    with open(test_json) as f:
        test_data = json.load(f)
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((512, 384)),
        transforms.RandomCrop(224),  # 使用random crop来采样多个patches
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    gt_scores = []
    pred_scores = []
    img_names = []
    
    # 只使用500张测试图片
    test_subset = test_data[:500]
    print(f"Evaluating on test set ({len(test_subset)} images)...")
    # 评估测试集子集（带进度条）
    for item in tqdm(test_subset, desc="Processing images", unit="img"):
        img_name = os.path.basename(item['image'])
        img_path = f'koniq-10k/test/{img_name}'
        
        if not os.path.exists(img_path) or img_name not in mos_dict:
            continue
        
        try:
            img = Image.open(img_path).convert('RGB')
            
            # 采样多个patches并平均预测
            patches = []
            for _ in range(num_patches):
                patch = transform(img)
                patches.append(patch)
            
            patches_tensor = torch.stack(patches).to(device)
            
            with torch.no_grad():
                paras = model(patches_tensor)
                model_target = models.TargetNet(paras).to(device)
                model_target.eval()
                pred = model_target(paras['target_in_vec'])
                pred_score = float(pred.mean())
            
            gt_scores.append(mos_dict[img_name])
            pred_scores.append(pred_score)
            img_names.append(img_name)
        
        except Exception as e:
            tqdm.write(f"  Error processing {img_name}: {e}")
            continue
    
    return np.array(gt_scores), np.array(pred_scores), img_names


def plot_error_analysis(gt_scores, pred_scores, img_names, save_path):
    """
    生成误差分析图
    
    Args:
        gt_scores: Ground truth scores
        pred_scores: Predicted scores
        img_names: Image names
        save_path: Save path for the figure
    """
    # 统一字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'Liberation Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 11
    
    # 计算误差
    errors = np.abs(gt_scores - pred_scores)
    
    # 计算相关系数
    from scipy import stats
    srcc = stats.spearmanr(gt_scores, pred_scores)[0]
    plcc = stats.pearsonr(gt_scores, pred_scores)[0]
    mae = np.mean(errors)
    
    # 创建图表
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    
    # 根据误差大小设置颜色和大小
    threshold = np.percentile(errors, 90)  # 90分位数
    colors = ['#E74C3C' if e > threshold else '#3498DB' for e in errors]
    sizes = [80 if e > threshold else 40 for e in errors]
    alphas = [0.8 if e > threshold else 0.5 for e in errors]
    
    # 散点图
    for i in range(len(gt_scores)):
        ax.scatter(gt_scores[i], pred_scores[i], 
                  c=colors[i], s=sizes[i], alpha=alphas[i], 
                  edgecolors='black', linewidth=0.5)
    
    # 对角线（完美预测）
    min_val = min(gt_scores.min(), pred_scores.min())
    max_val = max(gt_scores.max(), pred_scores.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
           'k--', lw=2, label='Perfect Prediction', alpha=0.6)
    
    # 标注（无标题，无统计框）
    ax.set_xlabel('Ground Truth MOS', fontsize=13, fontweight='bold')
    ax.set_ylabel('Predicted MOS', fontsize=13, fontweight='bold')
    
    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Normal prediction',
              markerfacecolor='#3498DB', markersize=8, alpha=0.6),
        Line2D([0], [0], marker='o', color='w', label='Large error (top 10%)',
              markerfacecolor='#E74C3C', markersize=10, alpha=0.8),
        Line2D([0], [0], color='k', linestyle='--', lw=2, label='Perfect prediction')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.9)
    
    # 网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 设置坐标轴范围
    margin = 5
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    
    plt.tight_layout()
    
    # 保存
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved error analysis plot to: {save_path}")
    
    png_path = save_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    
    # 返回统计信息和误差最大的图片
    top_error_idx = np.argsort(errors)[-5:][::-1]
    top_errors = [(img_names[i], gt_scores[i], pred_scores[i], errors[i]) 
                  for i in top_error_idx]
    
    return {
        'srcc': srcc,
        'plcc': plcc,
        'mae': mae,
        'top_errors': top_errors
    }


def main():
    print("=" * 80)
    print("预测误差分析")
    print("=" * 80)
    
    model_path = 'checkpoints/koniq-10k-swin_20251223_002226/best_model_srcc_0.9378_plcc_0.9485.pkl'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # 加载模型
    model, device = load_model(model_path)
    
    # 获取预测结果
    gt_scores, pred_scores, img_names = get_predictions(model, device)
    
    print(f"\n✓ Evaluated {len(gt_scores)} images")
    
    # 生成误差分析图
    output_dir = 'paper_figures'
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, 'error_analysis.pdf')
    stats = plot_error_analysis(gt_scores, pred_scores, img_names, save_path)
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("统计结果")
    print("=" * 80)
    print(f"SRCC: {stats['srcc']:.4f}")
    print(f"PLCC: {stats['plcc']:.4f}")
    print(f"MAE:  {stats['mae']:.2f}")
    
    print("\n误差最大的5张图片:")
    for i, (img_name, gt, pred, err) in enumerate(stats['top_errors'], 1):
        print(f"  {i}. {img_name}")
        print(f"     GT: {gt:.2f}, Pred: {pred:.2f}, Error: {err:.2f}")
    
    print("\n" + "=" * 80)
    print("✅ 误差分析完成！")
    print("=" * 80)
    print(f"\n生成文件:")
    print(f"  {save_path}")
    print(f"  {save_path.replace('.pdf', '.png')}")


if __name__ == '__main__':
    main()

