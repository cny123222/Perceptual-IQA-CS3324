#!/bin/bash

# Tmux-based Experiment Runner for 2 GPUs (Baseline: Alpha=0.3)
# Runs 2 experiments in parallel per batch
# Total: 18 experiments, ~15 hours

set -e

BASE_DIR="/root/Perceptual-IQA-CS3324"
SESSION_NAME="ablation_2gpu"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================================================"
echo "         2-GPU Experiment Runner (18 experiments, ~15h)"
echo "========================================================================"
echo ""
echo "Session name: $SESSION_NAME"
echo "Hardware: 2 × GPU"
echo "Strategy: 2 parallel experiments per batch"
echo ""

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null || true

# Create new session
tmux new-session -d -s $SESSION_NAME -n "monitor"

# Function to create a window and run command
run_in_tmux() {
    local window_name=$1
    local gpu_id=$2
    local command=$3
    
    tmux new-window -t $SESSION_NAME -n "$window_name"
    tmux send-keys -t $SESSION_NAME:$window_name "cd $BASE_DIR" C-m
    tmux send-keys -t $SESSION_NAME:$window_name "export CUDA_VISIBLE_DEVICES=$gpu_id" C-m
    tmux send-keys -t $SESSION_NAME:$window_name "$command" C-m
    
    echo -e "${GREEN}✓ Started: $window_name (GPU $gpu_id)${NC}"
}

echo -e "${BLUE}Starting experiments...${NC}"
echo ""

# =============================================================================
# BATCH 1: A1 + A2 (Core Ablations)
# =============================================================================

echo -e "${YELLOW}Batch 1/10: A1, A2 (Core Ablations)${NC}"

run_in_tmux "a1_no_att" "0" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq"

run_in_tmux "a2_no_rank" "1" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq"

sleep 2

# =============================================================================
# BATCH 2: A3 (Core Ablations)
# =============================================================================

echo -e "${YELLOW}Batch 2/10: A3 (Core Ablations)${NC}"

run_in_tmux "a3_no_multi" "0" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --no_multiscale --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq"

sleep 2

# =============================================================================
# BATCH 3: C1 + C2 (Ranking Sensitivity)
# =============================================================================

echo -e "${YELLOW}Batch 3/10: C1, C2 (Ranking Sensitivity)${NC}"

run_in_tmux "c1_alpha0.1" "0" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.1 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq"

run_in_tmux "c2_alpha0.5" "1" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.5 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq"

sleep 2

# =============================================================================
# BATCH 4: C3 (Ranking Sensitivity)
# =============================================================================

echo -e "${YELLOW}Batch 4/10: C3 (Ranking Sensitivity)${NC}"

run_in_tmux "c3_alpha0.7" "0" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.7 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq"

sleep 2

# =============================================================================
# BATCH 5: B1 + B2 (Model Size)
# =============================================================================

echo -e "${YELLOW}Batch 5/10: B1, B2 (Model Size)${NC}"

run_in_tmux "b1_tiny" "0" \
"python train_swin.py --dataset koniq-10k --model_size tiny --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq"

run_in_tmux "b2_small" "1" \
"python train_swin.py --dataset koniq-10k --model_size small --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq"

sleep 2

# =============================================================================
# BATCH 6: D1 + D2 (Regularization - Weight Decay)
# =============================================================================

echo -e "${YELLOW}Batch 6/10: D1, D2 (Weight Decay Sensitivity)${NC}"

run_in_tmux "d1_wd_low" "0" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 5e-6 --weight_decay 1e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq"

run_in_tmux "d2_wd_high" "1" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 5e-6 --weight_decay 3e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq"

sleep 2

# =============================================================================
# BATCH 7: D3 + D4 (Regularization - Drop Path)
# =============================================================================

echo -e "${YELLOW}Batch 7/10: D3, D4 (Drop Path Sensitivity)${NC}"

run_in_tmux "d3_dp_low" "0" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.2 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq"

run_in_tmux "d4_dp_high" "1" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.4 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq"

sleep 2

# =============================================================================
# BATCH 8: D5 + D6 (Regularization - Dropout)
# =============================================================================

echo -e "${YELLOW}Batch 8/10: D5, D6 (Dropout Sensitivity)${NC}"

run_in_tmux "d5_do_low" "0" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.3 --lr_scheduler cosine --test_random_crop --no_spaq"

run_in_tmux "d6_do_high" "1" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.5 --lr_scheduler cosine --test_random_crop --no_spaq"

sleep 2

# =============================================================================
# BATCH 9: E1 + E2 (Learning Rate)
# =============================================================================

echo -e "${YELLOW}Batch 9/10: E1, E2 (Learning Rate Sensitivity)${NC}"

run_in_tmux "e1_lr_0.5x" "0" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 2.5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq"

run_in_tmux "e2_lr_0.75x" "1" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 3.75e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq"

sleep 2

# =============================================================================
# BATCH 10: E4 + E5 (Learning Rate)
# =============================================================================

echo -e "${YELLOW}Batch 10/10: E4, E5 (Learning Rate Sensitivity)${NC}"

run_in_tmux "e4_lr_1.5x" "0" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 7.5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq"

run_in_tmux "e5_lr_2x" "1" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 1e-5 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq"

# Create monitoring window
tmux select-window -t $SESSION_NAME:monitor
tmux send-keys -t $SESSION_NAME:monitor "cd $BASE_DIR" C-m
tmux send-keys -t $SESSION_NAME:monitor "watch -n 2 'nvidia-smi && echo && echo \"=== Experiment Progress ===\" && ps aux | grep train_swin.py | grep -v grep | wc -l | xargs echo \"Active jobs:\" && echo && ls -lt logs/*.log 2>/dev/null | head -5'" C-m

echo ""
echo "========================================================================"
echo -e "${GREEN}✓ All experiments started!${NC}"
echo "========================================================================"
echo ""
echo "Tmux session: $SESSION_NAME"
echo "Total windows: 19 (18 experiments + 1 monitor)"
echo "Total batches: 10"
echo "Estimated time: ~15 hours"
echo ""
echo "NOTE: E3 (LR=5e-6) is the baseline - already completed!"
echo ""
echo "To attach to the session:"
echo -e "  ${BLUE}tmux attach -t $SESSION_NAME${NC}"
echo ""
echo "To list all windows:"
echo -e "  ${BLUE}tmux list-windows -t $SESSION_NAME${NC}"
echo ""
echo "To switch between windows:"
echo -e "  ${BLUE}Ctrl+b, w${NC} (interactive selection)"
echo -e "  ${BLUE}Ctrl+b, n${NC} (next window)"
echo -e "  ${BLUE}Ctrl+b, p${NC} (previous window)"
echo ""
echo "To detach from session:"
echo -e "  ${BLUE}Ctrl+b, d${NC}"
echo ""
echo "========================================================================"

