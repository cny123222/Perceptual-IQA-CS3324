"""
从KonIQ-10k中选择更多样本用于注意力可视化
每个质量等级选择5张图片
"""
import json
import csv
import numpy as np

# 读取MOS分数
csv_file = 'koniq-10k/koniq10k_scores_and_distributions.csv'
images = []
with open(csv_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        images.append({
            'image': row['image_name'],
            'mos': float(row['MOS_zscore'])
        })

# 按MOS排序
images_sorted = sorted(images, key=lambda x: x['mos'])

# 计算分位数
mos_values = [img['mos'] for img in images_sorted]
q20 = np.percentile(mos_values, 20)  # 低质量上限
q40 = np.percentile(mos_values, 40)  # 中低质量上限
q60 = np.percentile(mos_values, 60)  # 中质量上限
q80 = np.percentile(mos_values, 80)  # 中高质量上限

print(f"Quality Thresholds:")
print(f"  Very Low: < {q20:.3f}")
print(f"  Low: {q20:.3f} - {q40:.3f}")
print(f"  Mid: {q40:.3f} - {q60:.3f}")
print(f"  High: {q60:.3f} - {q80:.3f}")
print(f"  Very High: > {q80:.3f}")

# 分组
quality_groups = {
    'very_low': [img for img in images_sorted if img['mos'] < q20],
    'low': [img for img in images_sorted if q20 <= img['mos'] < q40],
    'mid': [img for img in images_sorted if q40 <= img['mos'] < q60],
    'high': [img for img in images_sorted if q60 <= img['mos'] < q80],
    'very_high': [img for img in images_sorted if img['mos'] >= q80]
}

# 从每组随机选5张（用固定seed保证可重复）
np.random.seed(42)
selected = {}
for quality, imgs in quality_groups.items():
    # 如果该组图片少于5张，全选；否则随机选5张
    n_select = min(5, len(imgs))
    indices = np.random.choice(len(imgs), n_select, replace=False)
    selected[quality] = [imgs[i] for i in indices]
    
    print(f"\n{quality.upper()}: {len(imgs)} images, selected {n_select}")
    for img in selected[quality]:
        print(f"  {img['image']}: MOS={img['mos']:.4f}")

# 保存
with open('selected_images_for_viz_extended.json', 'w') as f:
    json.dump(selected, f, indent=2)

print(f"\n✓ Saved to selected_images_for_viz_extended.json")
print(f"Total: {sum(len(imgs) for imgs in selected.values())} images")

