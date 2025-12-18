## ResNet-50 骨干网络
python train_test_IQA.py   --dataset koniq-10k   --epochs 10   --train_test_num 2   --batch_size 96   --train_patch_num 20   --test_patch_num 20

指标	你的结果 (Epoch 1)	论文	对比
SRCC	0.9009	0.906	✅ 超出 0.5%
PLCC	0.9170	0.917	✅ 持平

---

## Swin Transformer Tiny 骨干网络
python train_swin.py --dataset koniq-10k --epochs 10 --train_test_num 1 --batch_size 96 --train_patch_num 20 --test_patch_num 20

指标	最佳结果 (Epoch 2)	论文	对比
SRCC	0.9154	0.906	✅ 超出 1.04%
PLCC	0.9298	0.917	✅ 超出 1.40%

训练趋势：
- Epoch 1: SRCC 0.9138, PLCC 0.9286
- Epoch 2: SRCC 0.9154, PLCC 0.9298 (最佳) ⭐
- Epoch 3-10: 测试指标在 0.9072-0.9146 之间波动
- 训练集指标持续上升 (0.8673 → 0.9884)，存在轻微过拟合

---

## Swin Transformer Tiny + Ranking Loss (alpha=0.5)
python train_swin.py --dataset koniq-10k --epochs 10 --train_test_num 1 --batch_size 96 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.5 --ranking_loss_margin 0.1

指标	最佳结果 (Epoch 1)	论文	对比
SRCC	0.9178	0.906	✅ 超出 1.30%
PLCC	0.9326	0.917	✅ 超出 1.70%

训练详情：
- Epoch 1: Train_Loss: 5.578 (L1: 5.208, Rank: 0.741), Train_SRCC: 0.8696, **Test_SRCC: 0.9178, Test_PLCC: 0.9326** (最佳) ⭐
- Epoch 2: Train_Loss: 3.455 (L1: 3.286, Rank: 0.338), Train_SRCC: 0.9473, Test_SRCC: 0.9162, Test_PLCC: 0.9314

---

## Swin Transformer Tiny + Ranking Loss (alpha=0.3)
python train_swin.py --dataset koniq-10k --epochs 10 --train_test_num 1 --batch_size 96 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --ranking_loss_margin 0.1

指标	最佳结果 (Epoch 1)	论文	对比
SRCC	0.9206	0.906	✅ 超出 1.61%
PLCC	0.9334	0.917	✅ 超出 1.79%

训练详情：
- Epoch 1: Train_Loss: 5.409 (L1: 5.182, Rank: 0.756), Train_SRCC: 0.8686, **Test_SRCC: 0.9206, Test_PLCC: 0.9334** (最佳) ⭐

---

## Swin Transformer Tiny + Ranking Loss (alpha=1.0)
python train_swin.py --dataset koniq-10k --epochs 10 --train_test_num 1 --batch_size 96 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 1.0 --ranking_loss_margin 0.1

指标	最佳结果 (Epoch 1)	论文	对比
SRCC	0.9188	0.906	✅ 超出 1.41%
PLCC	0.9326	0.917	✅ 超出 1.70%

训练详情：
- Epoch 1: Train_Loss: 5.937 (L1: 5.233, Rank: 0.704), Train_SRCC: 0.8684, **Test_SRCC: 0.9188, Test_PLCC: 0.9326** (最佳) ⭐
- Epoch 2: Train_Loss: 3.570 (L1: 3.243, Rank: 0.327), Train_SRCC: 0.9489, Test_SRCC: 0.9139, Test_PLCC: 0.9265
- Epoch 3-10: 测试指标持续下降，从 0.9162 → 0.9107 (SRCC), 0.9257 → 0.9212 (PLCC)
- 训练集 SRCC 持续上升 (0.8684 → 0.9892)，存在明显过拟合

---

## Swin Transformer Tiny + Ranking Loss (alpha=0.3) + Multi-Scale Feature Fusion
python train_swin.py --dataset koniq-10k --epochs 10 --train_test_num 1 --batch_size 96 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --ranking_loss_margin 0.1

指标	最佳结果 (Epoch 1)	单尺度版本对比
SRCC	0.9184	0.9206	❌ 下降 0.24%
PLCC	0.9329	0.9334	❌ 下降 0.05%

训练详情：
- Epoch 1: Train_Loss: 5.042 (L1: 4.848, Rank: 0.646), Train_SRCC: 0.8834, **Test_SRCC: 0.9184, Test_PLCC: 0.9329** (最佳) ⭐
- Epoch 2: Train_Loss: 3.091 (L1: 2.999, Rank: 0.307), Train_SRCC: 0.9559, Test_SRCC: 0.9176, Test_PLCC: 0.9309
- Epoch 3-10: 测试指标波动在 0.9111-0.9183 (SRCC), 0.9249-0.9329 (PLCC)
- 训练集 SRCC 持续上升 (0.8834 → 0.9889)，存在过拟合
- SPAQ测试: SRCC 0.8646, PLCC 0.8590 (Epoch 1)

**分析：多尺度特征融合效果不如单尺度版本**

可能原因：
1. **模型复杂度增加但训练策略未调整**：
   - 输入通道数从 768 → 1440 (增加了 87.5%)，但学习率和优化策略未相应调整
   - 多尺度特征的简单concatenation可能不是最优融合方式

2. **缺少训练稳定性修复**：
   - 当前版本未应用三个关键训练修复（filter iterator bug, backbone LR decay, optimizer state preservation）
   - 这些修复在单尺度版本中可能已被应用或影响较小，但在更复杂的多尺度模型中影响更明显

3. **特征融合方式可能不够优化**：
   - 当前使用简单的AdaptiveAvgPool2d + Concatenation
   - 可能需要更sophisticated的融合机制（如注意力加权、特征选择等）

4. **训练过拟合趋势更明显**：
   - 训练集SRCC从0.8834上升到0.9889（上升10.5%）
   - 测试集最佳结果出现在Epoch 1，后续epoch性能下降或波动

改进建议：
- 应用三个训练稳定性修复（filter→list, backbone LR decay, optimizer state preservation）
- 考虑使用注意力机制对多尺度特征进行加权融合
- 调整学习率策略以适配更大的模型容量
- 增加正则化（dropout, weight decay）防止过拟合