#!/bin/bash
# ResNet50 + Improvements 消融实验
# 验证Multi-scale和Attention是否对CNN backbone也有效

echo "======================================================================================================"
echo "ResNet50 + Improvements Ablation Study"
echo "======================================================================================================"
echo ""
echo "This script will run 3 experiments:"
echo "  1. ResNet50 Baseline (Single-scale, No attention)"
echo "  2. ResNet50 + Multi-scale"
echo "  3. ResNet50 + Multi-scale + Attention"
echo ""
echo "Estimated time: ~4.5 hours (1.5h per experiment)"
echo "======================================================================================================"
echo ""

# 设置通用参数
DATASET="koniq-10k"
DATA_PATH="koniq-10k"  # Will auto-detect correct path
EPOCHS=10
LR=1e-4
BATCH_SIZE=32
TRAIN_PATCHES=25
TEST_PATCHES=25
SEED=42
PRELOAD="--preload_images"  # Enable image preloading for faster training

# 创建日志目录
LOG_DIR="logs/resnet_ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR
echo "Logs will be saved to: $LOG_DIR"
echo ""

# ====================================================================================================
# 实验1: ResNet50 Baseline (Single-scale, No attention)
# ====================================================================================================
echo ""
echo "======================================================================================================"
echo "Experiment 1/3: ResNet50 Baseline"
echo "======================================================================================================"
echo "Configuration:"
echo "  - Single-scale (Stage 4 only)"
echo "  - No attention"
echo "  - Learning Rate: $LR"
echo "======================================================================================================"
echo ""

python3 train_resnet_improved.py \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --train_patch_num $TRAIN_PATCHES \
    --test_patch_num $TEST_PATCHES \
    --no_color_jitter \
    --test_random_crop \
    --seed $SEED \
    --save_model \
    $PRELOAD \
    2>&1 | tee $LOG_DIR/exp1_baseline.log

echo ""
echo "✓ Experiment 1 completed!"
echo "  Log saved to: $LOG_DIR/exp1_baseline.log"
echo ""

# ====================================================================================================
# 实验2: ResNet50 + Multi-scale
# ====================================================================================================
echo ""
echo "======================================================================================================"
echo "Experiment 2/3: ResNet50 + Multi-scale"
echo "======================================================================================================"
echo "Configuration:"
echo "  - Multi-scale (Stage 1,2,3,4)"
echo "  - No attention (simple concatenation)"
echo "  - Learning Rate: $LR"
echo "======================================================================================================"
echo ""

python3 train_resnet_improved.py \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --train_patch_num $TRAIN_PATCHES \
    --test_patch_num $TEST_PATCHES \
    --use_multiscale \
    --no_color_jitter \
    --test_random_crop \
    --seed $SEED \
    --save_model \
    $PRELOAD \
    2>&1 | tee $LOG_DIR/exp2_multiscale.log

echo ""
echo "✓ Experiment 2 completed!"
echo "  Log saved to: $LOG_DIR/exp2_multiscale.log"
echo ""

# ====================================================================================================
# 实验3: ResNet50 + Multi-scale + Attention
# ====================================================================================================
echo ""
echo "======================================================================================================"
echo "Experiment 3/3: ResNet50 + Multi-scale + Attention"
echo "======================================================================================================"
echo "Configuration:"
echo "  - Multi-scale (Stage 1,2,3,4)"
echo "  - Channel attention"
echo "  - Learning Rate: $LR"
echo "======================================================================================================"
echo ""

python3 train_resnet_improved.py \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --train_patch_num $TRAIN_PATCHES \
    --test_patch_num $TEST_PATCHES \
    --use_multiscale \
    --use_attention \
    --no_color_jitter \
    --test_random_crop \
    --seed $SEED \
    --save_model \
    $PRELOAD \
    2>&1 | tee $LOG_DIR/exp3_multiscale_attention.log

echo ""
echo "✓ Experiment 3 completed!"
echo "  Log saved to: $LOG_DIR/exp3_multiscale_attention.log"
echo ""

# ====================================================================================================
# 提取结果
# ====================================================================================================
echo ""
echo "======================================================================================================"
echo "All Experiments Completed!"
echo "======================================================================================================"
echo ""
echo "Extracting results..."
echo ""

echo "Results Summary:"
echo "--------------------------------------------------------------------------------"
echo "Experiment                                    SRCC      PLCC"
echo "--------------------------------------------------------------------------------"

for exp in exp1_baseline exp2_multiscale exp3_multiscale_attention; do
    log_file="$LOG_DIR/${exp}.log"
    if [ -f "$log_file" ]; then
        srcc=$(grep "Best Test SRCC:" "$log_file" | tail -1 | awk '{print $4}')
        plcc=$(grep "Best Test PLCC:" "$log_file" | tail -1 | awk '{print $4}')
        
        case $exp in
            exp1_baseline)
                exp_name="ResNet50 Baseline"
                ;;
            exp2_multiscale)
                exp_name="ResNet50 + Multi-scale"
                ;;
            exp3_multiscale_attention)
                exp_name="ResNet50 + Multi-scale + Attention"
                ;;
        esac
        
        printf "%-45s %s      %s\n" "$exp_name" "$srcc" "$plcc"
    fi
done

echo "--------------------------------------------------------------------------------"
echo ""
echo "Detailed logs saved in: $LOG_DIR"
echo ""
echo "======================================================================================================"
echo "Next Steps:"
echo "  1. Compare results with ResNet baseline (0.8998 SRCC)"
echo "  2. Analyze contribution of each component"
echo "  3. Update paper with findings"
echo "======================================================================================================"
echo ""

