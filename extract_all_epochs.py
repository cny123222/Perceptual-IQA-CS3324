#!/usr/bin/env python3
"""提取所有epoch的真实训练数据"""

log_file = 'logs/batch1_gpu1_lr5e7_20251223_002208.log'

print("=" * 80)
print("从真实日志提取训练数据")
print("=" * 80)

epochs = []
train_losses = []
train_srccs = []
test_srccs = []
test_plccs = []

with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        # 匹配数据行格式: 数字\t数字\t\t数字\t\t数字\t\t数字
        parts = line.strip().split('\t')
        parts = [p for p in parts if p]  # 移除空字符串
        
        if len(parts) == 5:
            try:
                epoch = int(parts[0])
                train_loss = float(parts[1])
                train_srcc = float(parts[2])
                test_srcc = float(parts[3])
                test_plcc = float(parts[4])
                
                # 确保epoch在1-10范围内
                if 1 <= epoch <= 10:
                    epochs.append(epoch)
                    train_losses.append(train_loss)
                    train_srccs.append(train_srcc)
                    test_srccs.append(test_srcc)
                    test_plccs.append(test_plcc)
                    
                    print(f"Epoch {epoch}: Loss={train_loss:.4f}, Train_SRCC={train_srcc:.4f}, Test_SRCC={test_srcc:.4f}, Test_PLCC={test_plcc:.4f}")
            except (ValueError, IndexError):
                continue

print("\n" + "=" * 80)
print(f"✅ 成功提取 {len(epochs)} 个epoch的数据")
print("=" * 80)

if epochs:
    # 保存为CSV
    with open('training_data_real.csv', 'w') as f:
        f.write("epoch,train_loss,train_srcc,test_srcc,test_plcc\n")
        for i in range(len(epochs)):
            f.write(f"{epochs[i]},{train_losses[i]:.6f},{train_srccs[i]:.6f},{test_srccs[i]:.6f},{test_plccs[i]:.6f}\n")
    
    print(f"\n✓ 数据已保存到: training_data_real.csv")
    
    # 找到最佳epoch
    best_idx = test_srccs.index(max(test_srccs))
    print(f"\n🏆 最佳模型:")
    print(f"   Epoch: {epochs[best_idx]}")
    print(f"   Train Loss: {train_losses[best_idx]:.4f}")
    print(f"   Train SRCC: {train_srccs[best_idx]:.4f}")
    print(f"   Test SRCC: {test_srccs[best_idx]:.4f}")
    print(f"   Test PLCC: {test_plccs[best_idx]:.4f}")
    
    # 显示完整数据表格
    print("\n" + "=" * 80)
    print("完整训练数据")
    print("=" * 80)
    print(f"{'Epoch':<8} {'Train_Loss':<12} {'Train_SRCC':<12} {'Test_SRCC':<12} {'Test_PLCC':<12}")
    print("-" * 80)
    for i in range(len(epochs)):
        marker = " 🏆" if i == best_idx else ""
        print(f"{epochs[i]:<8} {train_losses[i]:<12.4f} {train_srccs[i]:<12.4f} {test_srccs[i]:<12.4f} {test_plccs[i]:<12.4f}{marker}")
    
else:
    print("\n❌ 未找到训练数据")

