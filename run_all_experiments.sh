#!/bin/bash

# Complete Experiment Runner (Baseline: Alpha=0.3)
# Runs all 16 experiments in 6 parallel batches
# Total estimated time: ~9 hours

set -e

BASE_DIR="/root/Perceptual-IQA-CS3324"
cd "$BASE_DIR"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "         Complete Ablation & Sensitivity Analysis Runner"
echo "========================================================================"
echo ""
echo "New Baseline: Alpha=0.3, SRCC 0.9352, PLCC 0.9460"
echo ""
echo "Total Experiments: 16"
echo "Total Batches: 6"
echo "Estimated Time: ~9 hours"
echo ""
echo "========================================================================"
echo ""

# Function to run a batch of experiments in parallel
run_batch() {
    local batch_num=$1
    local batch_name=$2
    shift 2
    local commands=("$@")
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Starting Batch $batch_num: $batch_name${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    # Run all commands in parallel
    local pids=()
    for i in "${!commands[@]}"; do
        echo -e "${GREEN}[Batch $batch_num, Experiment $((i+1))] Starting...${NC}"
        eval "${commands[$i]}" &
        pids+=($!)
    done
    
    # Wait for all to complete
    echo ""
    echo -e "${YELLOW}Waiting for Batch $batch_num to complete...${NC}"
    for pid in "${pids[@]}"; do
        wait $pid
    done
    
    echo ""
    echo -e "${GREEN}✓ Batch $batch_num completed!${NC}"
    echo ""
    sleep 5
}

# =============================================================================
# BATCH 1: Core Ablations (A1, A2, A3)
# =============================================================================

batch1_cmds=(
    # A1: Remove Attention
    "python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/ablation_a1_no_attention_alpha0.3_\$(date +%Y%m%d_%H%M%S).log"
    
    # A2: Remove Ranking Loss
    "python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/ablation_a2_no_ranking_alpha0.3_\$(date +%Y%m%d_%H%M%S).log"
    
    # A3: Remove Multi-scale
    "python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --no_multiscale --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/ablation_a3_no_multiscale_alpha0.3_\$(date +%Y%m%d_%H%M%S).log"
)

run_batch 1 "Core Ablations" "${batch1_cmds[@]}"

# =============================================================================
# BATCH 2: Ranking Loss Sensitivity (C1, C2, C3)
# =============================================================================

batch2_cmds=(
    # C1: Alpha=0.1
    "python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.1 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/sensitivity_c1_alpha0.1_\$(date +%Y%m%d_%H%M%S).log"
    
    # C2: Alpha=0.5
    "python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.5 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/sensitivity_c2_alpha0.5_\$(date +%Y%m%d_%H%M%S).log"
    
    # C3: Alpha=0.7
    "python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.7 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/sensitivity_c3_alpha0.7_\$(date +%Y%m%d_%H%M%S).log"
)

run_batch 2 "Ranking Loss Sensitivity" "${batch2_cmds[@]}"

# =============================================================================
# BATCH 3: Model Size Comparison (B1, B2)
# =============================================================================

batch3_cmds=(
    # B1: Swin-Tiny
    "python train_swin.py --dataset koniq-10k --model_size tiny --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/model_size_b1_tiny_alpha0.3_\$(date +%Y%m%d_%H%M%S).log"
    
    # B2: Swin-Small
    "python train_swin.py --dataset koniq-10k --model_size small --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/model_size_b2_small_alpha0.3_\$(date +%Y%m%d_%H%M%S).log"
)

run_batch 3 "Model Size Comparison" "${batch3_cmds[@]}"

# =============================================================================
# BATCH 4: Regularization Sensitivity Part 1 (D1, D2, D3)
# =============================================================================

batch4_cmds=(
    # D1: Lower Weight Decay
    "python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 5e-6 --weight_decay 1e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/reg_d1_wd1e-4_\$(date +%Y%m%d_%H%M%S).log"
    
    # D2: Higher Weight Decay
    "python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 5e-6 --weight_decay 3e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/reg_d2_wd3e-4_\$(date +%Y%m%d_%H%M%S).log"
    
    # D3: Lower Drop Path
    "python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.2 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/reg_d3_dp0.2_\$(date +%Y%m%d_%H%M%S).log"
)

run_batch 4 "Regularization Part 1" "${batch4_cmds[@]}"

# =============================================================================
# BATCH 5: Regularization Sensitivity Part 2 (D4, D5, D6)
# =============================================================================

batch5_cmds=(
    # D4: Higher Drop Path
    "python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.4 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/reg_d4_dp0.4_\$(date +%Y%m%d_%H%M%S).log"
    
    # D5: Lower Dropout
    "python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.3 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/reg_d5_do0.3_\$(date +%Y%m%d_%H%M%S).log"
    
    # D6: Higher Dropout
    "python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.5 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/reg_d6_do0.5_\$(date +%Y%m%d_%H%M%S).log"
)

run_batch 5 "Regularization Part 2" "${batch5_cmds[@]}"

# =============================================================================
# BATCH 6: Learning Rate Sensitivity (E1, E2)
# =============================================================================

batch6_cmds=(
    # E1: Lower LR
    "python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 2.5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/lr_e1_lr2.5e-6_\$(date +%Y%m%d_%H%M%S).log"
    
    # E2: Higher LR
    "python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 1e-5 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/lr_e2_lr1e-5_\$(date +%Y%m%d_%H%M%S).log"
)

run_batch 6 "Learning Rate Sensitivity" "${batch6_cmds[@]}"

# =============================================================================
# COMPLETION
# =============================================================================

echo ""
echo "========================================================================"
echo -e "${GREEN}✓ ALL EXPERIMENTS COMPLETED!${NC}"
echo "========================================================================"
echo ""
echo "Total experiments run: 16"
echo "Check logs/ directory for detailed results"
echo ""
echo "Next steps:"
echo "  1. Analyze results and update VALIDATION_AND_ABLATION_LOG.md"
echo "  2. Create visualizations for paper"
echo "  3. Run cross-dataset testing on best model"
echo ""
echo "========================================================================"

