#!/bin/bash

# Tmux-based Experiment Runner (Baseline: Alpha=0.3)
# Runs experiments in separate tmux windows for easy monitoring
# Total estimated time: ~9 hours

set -e

BASE_DIR="/root/Perceptual-IQA-CS3324"
SESSION_NAME="ablation_exp"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================================================"
echo "         Tmux-based Experiment Runner"
echo "========================================================================"
echo ""
echo "Session name: $SESSION_NAME"
echo "Total experiments: 16"
echo "Estimated time: ~9 hours"
echo ""

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null || true

# Create new session
tmux new-session -d -s $SESSION_NAME -n "monitor"

# Function to create a window and run command
run_in_tmux() {
    local window_name=$1
    local command=$2
    
    tmux new-window -t $SESSION_NAME -n "$window_name"
    tmux send-keys -t $SESSION_NAME:$window_name "cd $BASE_DIR" C-m
    tmux send-keys -t $SESSION_NAME:$window_name "$command" C-m
    
    echo -e "${GREEN}✓ Started: $window_name${NC}"
}

echo -e "${BLUE}Creating tmux windows...${NC}"
echo ""

# =============================================================================
# BATCH 1: Core Ablations
# =============================================================================

echo -e "${YELLOW}Batch 1: Core Ablations${NC}"

run_in_tmux "a1_no_att" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/ablation_a1_no_attention_alpha0.3_\$(date +%Y%m%d_%H%M%S).log"

run_in_tmux "a2_no_rank" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/ablation_a2_no_ranking_alpha0.3_\$(date +%Y%m%d_%H%M%S).log"

run_in_tmux "a3_no_multi" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --no_multiscale --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/ablation_a3_no_multiscale_alpha0.3_\$(date +%Y%m%d_%H%M%S).log"

sleep 2

# =============================================================================
# BATCH 2: Ranking Loss Sensitivity
# =============================================================================

echo -e "${YELLOW}Batch 2: Ranking Loss Sensitivity${NC}"

run_in_tmux "c1_alpha0.1" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.1 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/sensitivity_c1_alpha0.1_\$(date +%Y%m%d_%H%M%S).log"

run_in_tmux "c2_alpha0.5" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.5 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/sensitivity_c2_alpha0.5_\$(date +%Y%m%d_%H%M%S).log"

run_in_tmux "c3_alpha0.7" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.7 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/sensitivity_c3_alpha0.7_\$(date +%Y%m%d_%H%M%S).log"

sleep 2

# =============================================================================
# BATCH 3: Model Size
# =============================================================================

echo -e "${YELLOW}Batch 3: Model Size Comparison${NC}"

run_in_tmux "b1_tiny" \
"python train_swin.py --dataset koniq-10k --model_size tiny --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/model_size_b1_tiny_alpha0.3_\$(date +%Y%m%d_%H%M%S).log"

run_in_tmux "b2_small" \
"python train_swin.py --dataset koniq-10k --model_size small --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/model_size_b2_small_alpha0.3_\$(date +%Y%m%d_%H%M%S).log"

sleep 2

# =============================================================================
# BATCH 4: Regularization Part 1
# =============================================================================

echo -e "${YELLOW}Batch 4: Regularization Part 1${NC}"

run_in_tmux "d1_wd_low" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 5e-6 --weight_decay 1e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/reg_d1_wd1e-4_\$(date +%Y%m%d_%H%M%S).log"

run_in_tmux "d2_wd_high" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 5e-6 --weight_decay 3e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/reg_d2_wd3e-4_\$(date +%Y%m%d_%H%M%S).log"

run_in_tmux "d3_dp_low" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.2 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/reg_d3_dp0.2_\$(date +%Y%m%d_%H%M%S).log"

sleep 2

# =============================================================================
# BATCH 5: Regularization Part 2
# =============================================================================

echo -e "${YELLOW}Batch 5: Regularization Part 2${NC}"

run_in_tmux "d4_dp_high" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.4 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/reg_d4_dp0.4_\$(date +%Y%m%d_%H%M%S).log"

run_in_tmux "d5_do_low" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.3 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/reg_d5_do0.3_\$(date +%Y%m%d_%H%M%S).log"

run_in_tmux "d6_do_high" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.5 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/reg_d6_do0.5_\$(date +%Y%m%d_%H%M%S).log"

sleep 2

# =============================================================================
# BATCH 6: Learning Rate
# =============================================================================

echo -e "${YELLOW}Batch 6: Learning Rate Sensitivity${NC}"

run_in_tmux "e1_lr_low" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 2.5e-6 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/lr_e1_lr2.5e-6_\$(date +%Y%m%d_%H%M%S).log"

run_in_tmux "e2_lr_high" \
"python train_swin.py --dataset koniq-10k --model_size base --batch_size 32 --epochs 5 --patience 5 --train_patch_num 20 --test_patch_num 20 --ranking_loss_alpha 0.3 --attention_fusion --lr 1e-5 --weight_decay 2e-4 --drop_path_rate 0.3 --dropout_rate 0.4 --lr_scheduler cosine --test_random_crop --no_spaq 2>&1 | tee logs/lr_e2_lr1e-5_\$(date +%Y%m%d_%H%M%S).log"

# Create monitoring window
tmux select-window -t $SESSION_NAME:monitor
tmux send-keys -t $SESSION_NAME:monitor "cd $BASE_DIR" C-m
tmux send-keys -t $SESSION_NAME:monitor "watch -n 2 'nvidia-smi && echo && echo \"=== Running Experiments ===\" && ps aux | grep train_swin.py | grep -v grep | wc -l | xargs echo \"Active jobs:\"'" C-m

echo ""
echo "========================================================================"
echo -e "${GREEN}✓ All experiments started!${NC}"
echo "========================================================================"
echo ""
echo "Tmux session: $SESSION_NAME"
echo "Total windows: 17 (16 experiments + 1 monitor)"
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

