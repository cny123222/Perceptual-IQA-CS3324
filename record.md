python train_test_IQA.py   --dataset koniq-10k   --epochs 10   --train_test_num 2   --batch_size 96   --train_patch_num 20   --test_patch_num 20

指标	你的结果 (Epoch 1)	论文	对比
SRCC	0.9009	0.906	✅ 超出 0.5%
PLCC	0.9170	0.917	✅ 持平