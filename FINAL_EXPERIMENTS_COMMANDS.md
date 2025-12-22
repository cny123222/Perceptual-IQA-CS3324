# 🚀 Final Experiments - Exact Commands

**Configuration**: 10 rounds, 10 epochs, patience=3, LR=1e-6  
**Total Experiments**: 6 (Baseline + A1 + A2 + ResNet50 + B1 + B2)  
**Estimated Time**: ~5 hours (4 GPUs parallel)

---

## 📋 All Commands (Copy-Paste Ready)

### Baseline - Full Model (Swin-Base + Multi-scale + Attention)

```bash
python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 1e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```

---

### A1 - Remove Attention Fusion

```bash
python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 1e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```
**Note**: NO `--attention_fusion` flag

---

### A2 - Remove Multi-scale Fusion

```bash
python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 1e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --no_multi_scale \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```
**Note**: Add `--no_multi_scale` flag

---

### ResNet50 - Original HyperIQA Baseline

```bash
python train_test_IQA.py \
  --dataset koniq-10k \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 1e-6 \
  --weight_decay 2e-4 \
  --lr_scheduler cosine \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```
**Note**: Uses `train_test_IQA.py` (ResNet50 backbone)

---

### B1 - Swin-Tiny

```bash
python train_swin.py \
  --dataset koniq-10k \
  --model_size tiny \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 1e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.2 \
  --dropout_rate 0.3 \
  --lr_scheduler cosine \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```
**Note**: Lower regularization (drop_path=0.2, dropout=0.3)

---

### B2 - Swin-Small

```bash
python train_swin.py \
  --dataset koniq-10k \
  --model_size small \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 1e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.25 \
  --dropout_rate 0.35 \
  --lr_scheduler cosine \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter
```
**Note**: Moderate regularization (drop_path=0.25, dropout=0.35)

---

## 🔄 Parallel Execution (4 GPUs) - RECOMMENDED

### Batch 1 (Run simultaneously on 4 GPUs):

```bash
# Terminal 1 - GPU 0: Baseline (MOST IMPORTANT)
CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 1e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter &

# Terminal 2 - GPU 1: ResNet50 (CRITICAL COMPARISON)
CUDA_VISIBLE_DEVICES=1 python train_test_IQA.py \
  --dataset koniq-10k \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 1e-6 \
  --weight_decay 2e-4 \
  --lr_scheduler cosine \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter &

# Terminal 3 - GPU 2: A1 (Remove Attention)
CUDA_VISIBLE_DEVICES=2 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 1e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter &

# Terminal 4 - GPU 3: A2 (Remove Multi-scale)
CUDA_VISIBLE_DEVICES=3 python train_swin.py \
  --dataset koniq-10k \
  --model_size base \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 1e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --lr_scheduler cosine \
  --no_multi_scale \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter &

# Wait for all to complete
wait
```

**Time**: ~3.4 hours

---

### Batch 2 (After Batch 1 completes):

```bash
# Terminal 1 - GPU 0: B1 (Swin-Tiny)
CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --dataset koniq-10k \
  --model_size tiny \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 1e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.2 \
  --dropout_rate 0.3 \
  --lr_scheduler cosine \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter &

# Terminal 2 - GPU 1: B2 (Swin-Small)
CUDA_VISIBLE_DEVICES=1 python train_swin.py \
  --dataset koniq-10k \
  --model_size small \
  --batch_size 32 \
  --epochs 10 \
  --patience 3 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --train_test_num 10 \
  --lr 1e-6 \
  --weight_decay 2e-4 \
  --drop_path_rate 0.25 \
  --dropout_rate 0.35 \
  --lr_scheduler cosine \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --test_random_crop \
  --no_spaq \
  --no_color_jitter &

# Wait for all to complete
wait
```

**Time**: ~3.2 hours

---

## 📊 Quick Reference Table

| ID | GPU | Command | Time | Priority |
|----|-----|---------|------|----------|
| **Baseline** | 0 | `train_swin.py --model_size base --attention_fusion` | 3.4h | ⭐⭐⭐ |
| **ResNet50** | 1 | `train_test_IQA.py` | 2.5h | ⭐⭐⭐⭐⭐ |
| **A1** | 2 | `train_swin.py --model_size base` (no attention) | 3.4h | ⭐⭐⭐ |
| **A2** | 3 | `train_swin.py --model_size base --no_multi_scale` | 3.4h | ⭐⭐⭐ |
| **B1** | 0 | `train_swin.py --model_size tiny` | 3.0h | ⭐⭐ |
| **B2** | 1 | `train_swin.py --model_size small` | 3.2h | ⭐⭐ |

**Total Wall Time**: ~5 hours (Batch 1: 3.4h + Batch 2: 3.2h, but ResNet50 finishes earlier)

---

## 🔍 Monitoring Progress

### Check GPU Usage:
```bash
watch -n 1 nvidia-smi
```

### Check Latest Logs:
```bash
# List recent logs
ls -lt logs/ | head -10

# Monitor specific log
tail -f logs/[latest_log_file].log

# Check SRCC results
grep "median SRCC" logs/*.log
```

### Check Running Processes:
```bash
ps aux | grep python | grep train
```

---

## ⚠️ Important Reminders

1. **Before Starting**:
   - ✅ Ensure all 4 GPUs are available
   - ✅ Check disk space (need ~20GB for checkpoints)
   - ✅ Verify code is up to date (`git pull`)
   - ✅ Test one command first to ensure no errors

2. **During Execution**:
   - ✅ Monitor GPU memory usage
   - ✅ Check log files for errors
   - ✅ Ensure early stopping is working (patience=3)
   - ✅ Watch for OOM errors

3. **After Completion**:
   - ✅ Extract all results (median SRCC, PLCC)
   - ✅ Compute statistics (mean, std dev)
   - ✅ Save best checkpoints
   - ✅ Update `EXPERIMENTS_LOG_TRACKER.md`

---

## 🎯 Expected Results (10 rounds)

| Experiment | Expected Median SRCC | Expected Std Dev | Δ vs Best |
|------------|---------------------|------------------|-----------|
| **Baseline** | **0.937 ± 0.002** | **0.002** | - |
| A1 (No Attention) | 0.932 ± 0.002 | 0.002 | -0.005 |
| A2 (No Multi-scale) | 0.930 ± 0.002 | 0.002 | -0.007 |
| ResNet50 | 0.907 ± 0.003 | 0.003 | -0.030 |
| B1 (Tiny) | 0.921 ± 0.002 | 0.002 | -0.016 |
| B2 (Small) | 0.933 ± 0.002 | 0.002 | -0.004 |

---

## 📝 Results Recording Script

After experiments complete, run this to extract all results:

```bash
#!/bin/bash
echo "=== Final Experiments Results ==="
echo ""

for log in logs/swin_multiscale_ranking_alpha0_*.log; do
    if [ -f "$log" ]; then
        echo "File: $(basename $log)"
        grep "median SRCC" "$log" | tail -1
        echo ""
    fi
done

for log in logs/resnet_*.log; do
    if [ -f "$log" ]; then
        echo "File: $(basename $log)"
        grep "median SRCC" "$log" | tail -1
        echo ""
    fi
done
```

Save as `extract_results.sh`, then run:
```bash
chmod +x extract_results.sh
./extract_results.sh
```

---

**Status**: ✅ Ready to execute  
**Next Step**: Copy commands to terminals and start Batch 1

