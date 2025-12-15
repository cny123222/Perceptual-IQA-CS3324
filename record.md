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