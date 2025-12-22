#!/bin/bash

################################################################################
# 自动化实验脚本 - 使用tmux防止SSH断开
# 总时间: ~10小时
# 实验数: 6个 (3个batch，每个batch 2个GPU并行)
################################################################################

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
SESSION_NAME="iqa_experiments"
BASE_DIR="/root/Perceptual-IQA-CS3324"
LOG_DIR="$BASE_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 创建日志目录
mkdir -p "$LOG_DIR"

# 日志函数
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"
}

################################################################################
# 函数: 检查环境
################################################################################
check_environment() {
    log "检查环境..."
    
    # 检查tmux
    if ! command -v tmux &> /dev/null; then
        error "tmux未安装，请先安装: apt-get install tmux"
        exit 1
    fi
    
    # 检查Python
    if ! command -v python &> /dev/null; then
        error "Python未找到"
        exit 1
    fi
    
    # 检查数据集
    if [ ! -d "$BASE_DIR/koniq-10k" ]; then
        error "数据集不存在: $BASE_DIR/koniq-10k"
        exit 1
    fi
    
    # 检查GPU
    if ! command -v nvidia-smi &> /dev/null; then
        error "nvidia-smi未找到，无法检查GPU"
        exit 1
    fi
    
    local gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$gpu_count" -lt 2 ]; then
        error "需要至少2块GPU，当前只有 $gpu_count 块"
        exit 1
    fi
    
    # 检查磁盘空间 (需要至少30GB)
    local available_space=$(df "$BASE_DIR" | tail -1 | awk '{print $4}')
    local required_space=$((30 * 1024 * 1024))  # 30GB in KB
    if [ "$available_space" -lt "$required_space" ]; then
        warning "磁盘空间可能不足。可用: $(($available_space / 1024 / 1024))GB, 建议: 30GB"
    fi
    
    log "环境检查通过 ✓"
}

################################################################################
# 函数: 杀死现有session
################################################################################
kill_existing_session() {
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        warning "发现已存在的session: $SESSION_NAME"
        read -p "是否杀死现有session? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            tmux kill-session -t "$SESSION_NAME"
            log "已杀死现有session"
        else
            error "请手动处理现有session后再运行"
            exit 1
        fi
    fi
}

################################################################################
# 函数: 创建tmux session
################################################################################
create_tmux_session() {
    log "创建tmux session: $SESSION_NAME"
    
    # 创建session和第一个窗口
    tmux new-session -d -s "$SESSION_NAME" -n "monitor"
    
    # 创建额外的窗口
    tmux new-window -t "$SESSION_NAME" -n "gpu0"
    tmux new-window -t "$SESSION_NAME" -n "gpu1"
    tmux new-window -t "$SESSION_NAME" -n "controller"
    
    # 在monitor窗口设置监控命令
    tmux send-keys -t "$SESSION_NAME:monitor" "cd $BASE_DIR" C-m
    tmux send-keys -t "$SESSION_NAME:monitor" "watch -n 10 nvidia-smi" C-m
    
    log "Tmux session创建成功 ✓"
}

################################################################################
# 函数: 等待进程完成
################################################################################
wait_for_processes() {
    local gpu0_pid=$1
    local gpu1_pid=$2
    local batch_name=$3
    
    info "等待 $batch_name 完成..."
    info "  GPU 0 PID: $gpu0_pid"
    info "  GPU 1 PID: $gpu1_pid"
    
    # 等待两个进程都完成
    while kill -0 $gpu0_pid 2>/dev/null || kill -0 $gpu1_pid 2>/dev/null; do
        sleep 60  # 每分钟检查一次
        
        # 显示进度
        local gpu0_status="完成"
        local gpu1_status="完成"
        if kill -0 $gpu0_pid 2>/dev/null; then
            gpu0_status="运行中"
        fi
        if kill -0 $gpu1_pid 2>/dev/null; then
            gpu1_status="运行中"
        fi
        
        info "  $batch_name 状态: GPU0[$gpu0_status] GPU1[$gpu1_status]"
    done
    
    log "$batch_name 完成 ✓"
}

################################################################################
# 函数: 运行单个实验
################################################################################
run_experiment() {
    local gpu_id=$1
    local log_file=$2
    shift 2
    local cmd="$@"
    
    local window_name="gpu$gpu_id"
    local full_log="$LOG_DIR/$log_file"
    
    # 在对应的tmux窗口执行命令
    tmux send-keys -t "$SESSION_NAME:$window_name" "cd $BASE_DIR" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo '开始实验: $log_file'" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo '时间: \$(date)'" C-m
    tmux send-keys -t "$SESSION_NAME:$window_name" "$cmd 2>&1 | tee $full_log" C-m
}

################################################################################
# BATCH 1: Learning Rate Comparison
################################################################################
run_batch1() {
    log "========================================"
    log "BATCH 1: Learning Rate Comparison"
    log "========================================"
    
    local batch_start=$(date +%s)
    
    # GPU 0: LR=1e-6
    info "启动 GPU 0: LR=1e-6 (Base model)"
    run_experiment 0 "batch1_gpu0_lr1e6_${TIMESTAMP}.log" \
        "CUDA_VISIBLE_DEVICES=0 python train_swin.py \
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
        --no_color_jitter"
    
    sleep 5
    local gpu0_pid=$(pgrep -f "CUDA_VISIBLE_DEVICES=0.*train_swin.py.*lr 1e-6" | head -1)
    
    # GPU 1: LR=5e-7
    info "启动 GPU 1: LR=5e-7 (Base model)"
    run_experiment 1 "batch1_gpu1_lr5e7_${TIMESTAMP}.log" \
        "CUDA_VISIBLE_DEVICES=1 python train_swin.py \
        --dataset koniq-10k \
        --model_size base \
        --batch_size 32 \
        --epochs 10 \
        --patience 3 \
        --train_patch_num 20 \
        --test_patch_num 20 \
        --train_test_num 10 \
        --lr 5e-7 \
        --weight_decay 2e-4 \
        --drop_path_rate 0.3 \
        --dropout_rate 0.4 \
        --lr_scheduler cosine \
        --attention_fusion \
        --ranking_loss_alpha 0 \
        --test_random_crop \
        --no_spaq \
        --no_color_jitter"
    
    sleep 5
    local gpu1_pid=$(pgrep -f "CUDA_VISIBLE_DEVICES=1.*train_swin.py.*lr 5e-7" | head -1)
    
    # 等待完成
    wait_for_processes $gpu0_pid $gpu1_pid "Batch 1"
    
    local batch_end=$(date +%s)
    local batch_duration=$(( ($batch_end - $batch_start) / 60 ))
    log "Batch 1 完成！用时: ${batch_duration} 分钟"
}

################################################################################
# BATCH 2: Ablation Studies
################################################################################
run_batch2() {
    log "========================================"
    log "BATCH 2: Ablation Studies"
    log "========================================"
    
    local batch_start=$(date +%s)
    
    # GPU 0: A1 - Remove Attention
    info "启动 GPU 0: A1 - Remove Attention"
    run_experiment 0 "batch2_gpu0_A1_no_attention_${TIMESTAMP}.log" \
        "CUDA_VISIBLE_DEVICES=0 python train_swin.py \
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
        --no_color_jitter"
    
    sleep 5
    local gpu0_pid=$(pgrep -f "CUDA_VISIBLE_DEVICES=0.*train_swin.py" | head -1)
    
    # GPU 1: A2 - Remove Multi-scale
    info "启动 GPU 1: A2 - Remove Multi-scale"
    run_experiment 1 "batch2_gpu1_A2_no_multiscale_${TIMESTAMP}.log" \
        "CUDA_VISIBLE_DEVICES=1 python train_swin.py \
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
        --no_color_jitter"
    
    sleep 5
    local gpu1_pid=$(pgrep -f "CUDA_VISIBLE_DEVICES=1.*train_swin.py" | head -1)
    
    # 等待完成
    wait_for_processes $gpu0_pid $gpu1_pid "Batch 2"
    
    local batch_end=$(date +%s)
    local batch_duration=$(( ($batch_end - $batch_start) / 60 ))
    log "Batch 2 完成！用时: ${batch_duration} 分钟"
}

################################################################################
# BATCH 3: Model Size Comparison
################################################################################
run_batch3() {
    log "========================================"
    log "BATCH 3: Model Size Comparison"
    log "========================================"
    
    local batch_start=$(date +%s)
    
    # GPU 0: B1 - Swin-Tiny
    info "启动 GPU 0: B1 - Swin-Tiny"
    run_experiment 0 "batch3_gpu0_B1_tiny_${TIMESTAMP}.log" \
        "CUDA_VISIBLE_DEVICES=0 python train_swin.py \
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
        --no_color_jitter"
    
    sleep 5
    local gpu0_pid=$(pgrep -f "CUDA_VISIBLE_DEVICES=0.*train_swin.py.*tiny" | head -1)
    
    # GPU 1: B2 - Swin-Small
    info "启动 GPU 1: B2 - Swin-Small"
    run_experiment 1 "batch3_gpu1_B2_small_${TIMESTAMP}.log" \
        "CUDA_VISIBLE_DEVICES=1 python train_swin.py \
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
        --no_color_jitter"
    
    sleep 5
    local gpu1_pid=$(pgrep -f "CUDA_VISIBLE_DEVICES=1.*train_swin.py.*small" | head -1)
    
    # 等待完成
    wait_for_processes $gpu0_pid $gpu1_pid "Batch 3"
    
    local batch_end=$(date +%s)
    local batch_duration=$(( ($batch_end - $batch_start) / 60 ))
    log "Batch 3 完成！用时: ${batch_duration} 分钟"
}

################################################################################
# 函数: 提取结果
################################################################################
extract_results() {
    log "========================================"
    log "提取实验结果"
    log "========================================"
    
    local results_file="$BASE_DIR/FINAL_RESULTS_${TIMESTAMP}.txt"
    
    {
        echo "═══════════════════════════════════════════════════════════════"
        echo "          Final Experiments Results Summary"
        echo "          运行时间: $(date)"
        echo "═══════════════════════════════════════════════════════════════"
        echo ""
        
        echo "📊 BATCH 1: Learning Rate Comparison"
        echo "───────────────────────────────────────────────────────────────"
        for log in $LOG_DIR/batch1_*${TIMESTAMP}.log; do
            if [ -f "$log" ]; then
                echo "文件: $(basename $log)"
                grep "median SRCC" "$log" | tail -1 || echo "  结果未找到"
                grep "Best test SRCC" "$log" | tail -1 || echo "  最佳结果未找到"
                echo ""
            fi
        done
        
        echo "📊 BATCH 2: Ablation Studies"
        echo "───────────────────────────────────────────────────────────────"
        for log in $LOG_DIR/batch2_*${TIMESTAMP}.log; do
            if [ -f "$log" ]; then
                echo "文件: $(basename $log)"
                grep "median SRCC" "$log" | tail -1 || echo "  结果未找到"
                grep "Best test SRCC" "$log" | tail -1 || echo "  最佳结果未找到"
                echo ""
            fi
        done
        
        echo "📊 BATCH 3: Model Size Comparison"
        echo "───────────────────────────────────────────────────────────────"
        for log in $LOG_DIR/batch3_*${TIMESTAMP}.log; do
            if [ -f "$log" ]; then
                echo "文件: $(basename $log)"
                grep "median SRCC" "$log" | tail -1 || echo "  结果未找到"
                grep "Best test SRCC" "$log" | tail -1 || echo "  最佳结果未找到"
                echo ""
            fi
        done
        
        echo "═══════════════════════════════════════════════════════════════"
    } | tee "$results_file"
    
    log "结果已保存到: $results_file"
}

################################################################################
# 函数: 发送完成通知
################################################################################
send_completion_notification() {
    local total_time=$1
    
    log "========================================"
    log "🎉 所有实验完成！"
    log "========================================"
    log "总用时: $total_time 分钟 ($(($total_time / 60)) 小时 $(($total_time % 60)) 分钟)"
    log ""
    log "结果文件: $BASE_DIR/FINAL_RESULTS_${TIMESTAMP}.txt"
    log ""
    log "下一步:"
    log "  1. 查看结果: cat $BASE_DIR/FINAL_RESULTS_${TIMESTAMP}.txt"
    log "  2. 检查日志: ls -lh $LOG_DIR/batch*${TIMESTAMP}.log"
    log "  3. 查看checkpoints: ls -lh $BASE_DIR/checkpoints/"
    log ""
    log "Tmux session '$SESSION_NAME' 仍在运行"
    log "  - 附加: tmux attach -t $SESSION_NAME"
    log "  - 杀死: tmux kill-session -t $SESSION_NAME"
}

################################################################################
# 主函数
################################################################################
main() {
    local script_start=$(date +%s)
    
    echo "═══════════════════════════════════════════════════════════════"
    echo "     自动化实验脚本 - 使用tmux"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "配置:"
    echo "  - Session名称: $SESSION_NAME"
    echo "  - 工作目录: $BASE_DIR"
    echo "  - 日志目录: $LOG_DIR"
    echo "  - 实验数量: 6个 (3个batch)"
    echo "  - 预计时间: ~10小时"
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    
    # 确认执行
    read -p "确认开始实验? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        error "用户取消"
        exit 1
    fi
    
    # 检查环境
    check_environment
    
    # 杀死现有session
    kill_existing_session
    
    # 创建tmux session
    create_tmux_session
    
    # 运行所有batch
    run_batch1
    sleep 10
    
    run_batch2
    sleep 10
    
    run_batch3
    sleep 10
    
    # 提取结果
    extract_results
    
    # 计算总时间
    local script_end=$(date +%s)
    local total_time=$(( ($script_end - $script_start) / 60 ))
    
    # 发送完成通知
    send_completion_notification $total_time
    
    log "脚本执行完成！"
}

################################################################################
# 运行主函数
################################################################################
main "$@"

