#!/bin/bash

################################################################################
# 4-GPU Optimized Experiment Runner
# Total: 14 experiments in 4 batches (~6 hours)
# 
# Execution Plan:
#   Batch 1 (1.5h): A1, A2, A3, C1    (Core Ablations + Ranking start)
#   Batch 2 (1.5h): C2, C3, B1, B2    (Ranking + Model Size)
#   Batch 3 (1.5h): D1, D2, D4, E1    (Weight Decay + LR start)
#   Batch 4 (1.5h): E3, E4            (Learning Rate)
#
# Total: ~6 hours (晚上11点开始 → 早上5点完成)
################################################################################

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Base directory
BASE_DIR="/root/Perceptual-IQA-CS3324"
cd "$BASE_DIR"

# Kill function for tmux sessions
kill_tmux_session() {
    local session_name=$1
    if tmux has-session -t "$session_name" 2>/dev/null; then
        tmux kill-session -t "$session_name"
    fi
}

# Wait for all sessions to complete
wait_for_batch() {
    local sessions=("$@")
    echo -e "${CYAN}⏳ Waiting for batch to complete...${NC}"
    
    while true; do
        all_done=true
        for session in "${sessions[@]}"; do
            if tmux has-session -t "$session" 2>/dev/null; then
                all_done=false
                break
            fi
        done
        
        if $all_done; then
            break
        fi
        sleep 30
    done
    
    echo -e "${GREEN}✅ Batch completed!${NC}"
}

# Print header
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║           🚀 4-GPU Experiment Runner (14 experiments)             ║${NC}"
echo -e "${CYAN}║                    Estimated Time: ~6 hours                        ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Confirmation
echo -e "${YELLOW}This will run 14 experiments in 4 batches:${NC}"
echo -e "  ${BLUE}Batch 1:${NC} A1, A2, A3, C1 (Core Ablations + Ranking)"
echo -e "  ${BLUE}Batch 2:${NC} C2, C3, B1, B2 (Ranking + Model Size)"
echo -e "  ${BLUE}Batch 3:${NC} D1, D2, D4, E1 (Regularization + LR)"
echo -e "  ${BLUE}Batch 4:${NC} E3, E4 (Learning Rate)"
echo ""
echo -e "${YELLOW}Press Enter to start, or Ctrl+C to cancel...${NC}"
read

START_TIME=$(date +%s)
echo -e "${GREEN}🎬 Starting experiments at $(date)${NC}"
echo ""

################################################################################
# BATCH 1: Core Ablations Start (4 experiments, ~1.5 hours)
################################################################################

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  BATCH 1/4: Core Ablations + Ranking Start (1.5 hours)${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# A1: Remove Attention (GPU 0)
echo -e "${BLUE}[1/4]${NC} Starting A1 (Remove Attention) on GPU 0..."
kill_tmux_session "exp-a1"
tmux new-session -d -s "exp-a1" \
  "cd $BASE_DIR && CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --koniq_set /root/autodl-tmp/koniq-10k \
  --patch_size 32 \
  --batch_size 4 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --num_workers 8 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --epochs 100 \
  --model_size base \
  --multiscale \
  --multiscale_stages 1 2 3 \
  --ranking_loss_alpha 0.3 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --patience 5; \
  echo 'A1 Complete'; exec bash"

# A2: Remove Ranking Loss (GPU 1)
echo -e "${BLUE}[2/4]${NC} Starting A2 (Remove Ranking Loss) on GPU 1..."
kill_tmux_session "exp-a2"
tmux new-session -d -s "exp-a2" \
  "cd $BASE_DIR && CUDA_VISIBLE_DEVICES=1 python train_swin.py \
  --koniq_set /root/autodl-tmp/koniq-10k \
  --patch_size 32 \
  --batch_size 4 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --num_workers 8 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --epochs 100 \
  --model_size base \
  --multiscale \
  --multiscale_stages 1 2 3 \
  --attention_fusion \
  --ranking_loss_alpha 0 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --patience 5; \
  echo 'A2 Complete'; exec bash"

# A3: Remove Multi-scale (GPU 2)
echo -e "${BLUE}[3/4]${NC} Starting A3 (Remove Multi-scale) on GPU 2..."
kill_tmux_session "exp-a3"
tmux new-session -d -s "exp-a3" \
  "cd $BASE_DIR && CUDA_VISIBLE_DEVICES=2 python train_swin.py \
  --koniq_set /root/autodl-tmp/koniq-10k \
  --patch_size 32 \
  --batch_size 4 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --num_workers 8 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --epochs 100 \
  --model_size base \
  --attention_fusion \
  --ranking_loss_alpha 0.3 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --patience 5; \
  echo 'A3 Complete'; exec bash"

# C1: Alpha=0.1 (GPU 3)
echo -e "${BLUE}[4/4]${NC} Starting C1 (Alpha=0.1) on GPU 3..."
kill_tmux_session "exp-c1"
tmux new-session -d -s "exp-c1" \
  "cd $BASE_DIR && CUDA_VISIBLE_DEVICES=3 python train_swin.py \
  --koniq_set /root/autodl-tmp/koniq-10k \
  --patch_size 32 \
  --batch_size 4 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --num_workers 8 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --epochs 100 \
  --model_size base \
  --multiscale \
  --multiscale_stages 1 2 3 \
  --attention_fusion \
  --ranking_loss_alpha 0.1 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --patience 5; \
  echo 'C1 Complete'; exec bash"

echo -e "${CYAN}🚀 Batch 1 started! Use 'tmux ls' to see sessions.${NC}"
wait_for_batch "exp-a1" "exp-a2" "exp-a3" "exp-c1"

################################################################################
# BATCH 2: Ranking + Model Size (4 experiments, ~1.5 hours)
################################################################################

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  BATCH 2/4: Ranking + Model Size (1.5 hours)${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# C2: Alpha=0.5 (GPU 0)
echo -e "${BLUE}[1/4]${NC} Starting C2 (Alpha=0.5) on GPU 0..."
kill_tmux_session "exp-c2"
tmux new-session -d -s "exp-c2" \
  "cd $BASE_DIR && CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --koniq_set /root/autodl-tmp/koniq-10k \
  --patch_size 32 \
  --batch_size 4 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --num_workers 8 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --epochs 100 \
  --model_size base \
  --multiscale \
  --multiscale_stages 1 2 3 \
  --attention_fusion \
  --ranking_loss_alpha 0.5 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --patience 5; \
  echo 'C2 Complete'; exec bash"

# C3: Alpha=0.7 (GPU 1)
echo -e "${BLUE}[2/4]${NC} Starting C3 (Alpha=0.7) on GPU 1..."
kill_tmux_session "exp-c3"
tmux new-session -d -s "exp-c3" \
  "cd $BASE_DIR && CUDA_VISIBLE_DEVICES=1 python train_swin.py \
  --koniq_set /root/autodl-tmp/koniq-10k \
  --patch_size 32 \
  --batch_size 4 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --num_workers 8 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --epochs 100 \
  --model_size base \
  --multiscale \
  --multiscale_stages 1 2 3 \
  --attention_fusion \
  --ranking_loss_alpha 0.7 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --patience 5; \
  echo 'C3 Complete'; exec bash"

# B1: Swin-Tiny (GPU 2)
echo -e "${BLUE}[3/4]${NC} Starting B1 (Swin-Tiny) on GPU 2..."
kill_tmux_session "exp-b1"
tmux new-session -d -s "exp-b1" \
  "cd $BASE_DIR && CUDA_VISIBLE_DEVICES=2 python train_swin.py \
  --koniq_set /root/autodl-tmp/koniq-10k \
  --patch_size 32 \
  --batch_size 4 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --num_workers 8 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --epochs 100 \
  --model_size tiny \
  --multiscale \
  --multiscale_stages 1 2 3 \
  --attention_fusion \
  --ranking_loss_alpha 0.3 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --patience 5; \
  echo 'B1 Complete'; exec bash"

# B2: Swin-Small (GPU 3)
echo -e "${BLUE}[4/4]${NC} Starting B2 (Swin-Small) on GPU 3..."
kill_tmux_session "exp-b2"
tmux new-session -d -s "exp-b2" \
  "cd $BASE_DIR && CUDA_VISIBLE_DEVICES=3 python train_swin.py \
  --koniq_set /root/autodl-tmp/koniq-10k \
  --patch_size 32 \
  --batch_size 4 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --num_workers 8 \
  --lr 5e-6 \
  --weight_decay 2e-4 \
  --epochs 100 \
  --model_size small \
  --multiscale \
  --multiscale_stages 1 2 3 \
  --attention_fusion \
  --ranking_loss_alpha 0.3 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --patience 5; \
  echo 'B2 Complete'; exec bash"

echo -e "${CYAN}🚀 Batch 2 started!${NC}"
wait_for_batch "exp-c2" "exp-c3" "exp-b1" "exp-b2"

################################################################################
# BATCH 3: Regularization + LR Start (4 experiments, ~1.5 hours)
################################################################################

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  BATCH 3/4: Regularization + LR Start (1.5 hours)${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# D1: Weight Decay=5e-5 (GPU 0)
echo -e "${BLUE}[1/4]${NC} Starting D1 (WD=5e-5) on GPU 0..."
kill_tmux_session "exp-d1"
tmux new-session -d -s "exp-d1" \
  "cd $BASE_DIR && CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --koniq_set /root/autodl-tmp/koniq-10k \
  --patch_size 32 \
  --batch_size 4 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --num_workers 8 \
  --lr 5e-6 \
  --weight_decay 5e-5 \
  --epochs 100 \
  --model_size base \
  --multiscale \
  --multiscale_stages 1 2 3 \
  --attention_fusion \
  --ranking_loss_alpha 0.3 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --patience 5; \
  echo 'D1 Complete'; exec bash"

# D2: Weight Decay=1e-4 (GPU 1)
echo -e "${BLUE}[2/4]${NC} Starting D2 (WD=1e-4) on GPU 1..."
kill_tmux_session "exp-d2"
tmux new-session -d -s "exp-d2" \
  "cd $BASE_DIR && CUDA_VISIBLE_DEVICES=1 python train_swin.py \
  --koniq_set /root/autodl-tmp/koniq-10k \
  --patch_size 32 \
  --batch_size 4 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --num_workers 8 \
  --lr 5e-6 \
  --weight_decay 1e-4 \
  --epochs 100 \
  --model_size base \
  --multiscale \
  --multiscale_stages 1 2 3 \
  --attention_fusion \
  --ranking_loss_alpha 0.3 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --patience 5; \
  echo 'D2 Complete'; exec bash"

# D4: Weight Decay=4e-4 (GPU 2)
echo -e "${BLUE}[3/4]${NC} Starting D4 (WD=4e-4) on GPU 2..."
kill_tmux_session "exp-d4"
tmux new-session -d -s "exp-d4" \
  "cd $BASE_DIR && CUDA_VISIBLE_DEVICES=2 python train_swin.py \
  --koniq_set /root/autodl-tmp/koniq-10k \
  --patch_size 32 \
  --batch_size 4 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --num_workers 8 \
  --lr 5e-6 \
  --weight_decay 4e-4 \
  --epochs 100 \
  --model_size base \
  --multiscale \
  --multiscale_stages 1 2 3 \
  --attention_fusion \
  --ranking_loss_alpha 0.3 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --patience 5; \
  echo 'D4 Complete'; exec bash"

# E1: LR=2.5e-6 (GPU 3)
echo -e "${BLUE}[4/4]${NC} Starting E1 (LR=2.5e-6) on GPU 3..."
kill_tmux_session "exp-e1"
tmux new-session -d -s "exp-e1" \
  "cd $BASE_DIR && CUDA_VISIBLE_DEVICES=3 python train_swin.py \
  --koniq_set /root/autodl-tmp/koniq-10k \
  --patch_size 32 \
  --batch_size 4 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --num_workers 8 \
  --lr 2.5e-6 \
  --weight_decay 2e-4 \
  --epochs 100 \
  --model_size base \
  --multiscale \
  --multiscale_stages 1 2 3 \
  --attention_fusion \
  --ranking_loss_alpha 0.3 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --patience 5; \
  echo 'E1 Complete'; exec bash"

echo -e "${CYAN}🚀 Batch 3 started!${NC}"
wait_for_batch "exp-d1" "exp-d2" "exp-d4" "exp-e1"

################################################################################
# BATCH 4: Learning Rate (2 experiments, ~1.5 hours)
################################################################################

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  BATCH 4/4: Learning Rate (1.5 hours)${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# E3: LR=7.5e-6 (GPU 0)
echo -e "${BLUE}[1/2]${NC} Starting E3 (LR=7.5e-6) on GPU 0..."
kill_tmux_session "exp-e3"
tmux new-session -d -s "exp-e3" \
  "cd $BASE_DIR && CUDA_VISIBLE_DEVICES=0 python train_swin.py \
  --koniq_set /root/autodl-tmp/koniq-10k \
  --patch_size 32 \
  --batch_size 4 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --num_workers 8 \
  --lr 7.5e-6 \
  --weight_decay 2e-4 \
  --epochs 100 \
  --model_size base \
  --multiscale \
  --multiscale_stages 1 2 3 \
  --attention_fusion \
  --ranking_loss_alpha 0.3 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --patience 5; \
  echo 'E3 Complete'; exec bash"

# E4: LR=1e-5 (GPU 1)
echo -e "${BLUE}[2/2]${NC} Starting E4 (LR=1e-5) on GPU 1..."
kill_tmux_session "exp-e4"
tmux new-session -d -s "exp-e4" \
  "cd $BASE_DIR && CUDA_VISIBLE_DEVICES=1 python train_swin.py \
  --koniq_set /root/autodl-tmp/koniq-10k \
  --patch_size 32 \
  --batch_size 4 \
  --train_patch_num 20 \
  --test_patch_num 20 \
  --num_workers 8 \
  --lr 1e-5 \
  --weight_decay 2e-4 \
  --epochs 100 \
  --model_size base \
  --multiscale \
  --multiscale_stages 1 2 3 \
  --attention_fusion \
  --ranking_loss_alpha 0.3 \
  --drop_path_rate 0.3 \
  --dropout_rate 0.4 \
  --patience 5; \
  echo 'E4 Complete'; exec bash"

echo -e "${CYAN}🚀 Batch 4 started!${NC}"
wait_for_batch "exp-e3" "exp-e4"

################################################################################
# ALL DONE!
################################################################################

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                   🎉 ALL EXPERIMENTS COMPLETE! 🎉                  ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}📊 Summary:${NC}"
echo -e "  ${BLUE}Total experiments:${NC} 14"
echo -e "  ${BLUE}Total time:${NC} ${HOURS}h ${MINUTES}m"
echo -e "  ${BLUE}Completed at:${NC} $(date)"
echo ""
echo -e "${CYAN}📂 Results locations:${NC}"
echo -e "  ${BLUE}Logs:${NC} logs/"
echo -e "  ${BLUE}Checkpoints:${NC} checkpoints/"
echo ""
echo -e "${YELLOW}📋 Next steps:${NC}"
echo -e "  1. Check logs for each experiment's best performance"
echo -e "  2. Update VALIDATION_AND_ABLATION_LOG.md with results"
echo -e "  3. Generate plots for sensitivity analysis"
echo -e "  4. Compare with baseline (Alpha=0.3, SRCC=0.9352)"
echo ""
echo -e "${GREEN}✨ Happy analyzing! ✨${NC}"

