#!/bin/bash
# 快速运行跨数据集测试的便捷脚本

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}跨数据集测试快速启动脚本${NC}"
echo -e "${BLUE}========================================${NC}"

# 检查参数
if [ $# -lt 2 ]; then
    echo -e "${RED}用法: $0 <checkpoint_path> <model_size> [test_patch_num]${NC}"
    echo ""
    echo "示例:"
    echo "  $0 checkpoints/koniq-10k-swin_20251218_232111/best_model_srcc_0.9236_plcc_0.9406.pkl tiny"
    echo "  $0 checkpoints/latest/best_model.pkl small 20"
    echo ""
    echo "或者使用预设配置:"
    echo "  $0 config1  # 测试Config 1 (Swin-Tiny最佳配置)"
    echo "  $0 latest-tiny  # 自动找到最新的Swin-Tiny checkpoint"
    echo "  $0 latest-small  # 自动找到最新的Swin-Small checkpoint"
    exit 1
fi

CHECKPOINT=$1
MODEL_SIZE=$2
TEST_PATCH_NUM=${3:-20}  # 默认20

# 预设配置
if [ "$CHECKPOINT" == "config1" ]; then
    echo -e "${GREEN}使用预设: Config 1 (Swin-Tiny 最佳配置)${NC}"
    CHECKPOINT="checkpoints/koniq-10k-swin_20251218_232111/best_model_srcc_0.9236_plcc_0.9406.pkl"
    MODEL_SIZE="tiny"
elif [ "$CHECKPOINT" == "latest-tiny" ]; then
    echo -e "${GREEN}查找最新的Swin-Tiny checkpoint...${NC}"
    LATEST_DIR=$(ls -td checkpoints/koniq-10k-swin_2025* 2>/dev/null | head -1)
    if [ -z "$LATEST_DIR" ]; then
        echo -e "${RED}错误: 未找到Swin-Tiny checkpoint${NC}"
        exit 1
    fi
    CHECKPOINT=$(ls "$LATEST_DIR"/best_model_srcc_*.pkl 2>/dev/null | sort -V | tail -1)
    MODEL_SIZE="tiny"
    echo -e "${GREEN}找到: $CHECKPOINT${NC}"
elif [ "$CHECKPOINT" == "latest-small" ]; then
    echo -e "${GREEN}查找最新的Swin-Small checkpoint...${NC}"
    LATEST_DIR=$(ls -td checkpoints/koniq-10k-swin-*small* 2>/dev/null | head -1)
    if [ -z "$LATEST_DIR" ]; then
        # 尝试其他可能的命名
        LATEST_DIR=$(ls -td checkpoints/*small* 2>/dev/null | head -1)
    fi
    if [ -z "$LATEST_DIR" ]; then
        echo -e "${RED}错误: 未找到Swin-Small checkpoint${NC}"
        exit 1
    fi
    CHECKPOINT=$(ls "$LATEST_DIR"/best_model_srcc_*.pkl 2>/dev/null | sort -V | tail -1)
    MODEL_SIZE="small"
    echo -e "${GREEN}找到: $CHECKPOINT${NC}"
fi

# 检查checkpoint是否存在
if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${RED}错误: Checkpoint不存在: $CHECKPOINT${NC}"
    exit 1
fi

# 显示配置
echo ""
echo -e "${BLUE}测试配置:${NC}"
echo "  Checkpoint: $CHECKPOINT"
echo "  Model Size: $MODEL_SIZE"
echo "  Test Patch Num: $TEST_PATCH_NUM"
echo "  Datasets: KonIQ-10k, SPAQ, KADID-10K, AGIQA-3K"
echo ""

# 运行测试
echo -e "${GREEN}开始测试...${NC}"
echo ""

python cross_dataset_test.py \
  --checkpoint "$CHECKPOINT" \
  --model_size "$MODEL_SIZE" \
  --test_patch_num "$TEST_PATCH_NUM"

# 检查是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}测试完成！${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    # 显示结果文件
    RESULT_FILE=$(ls -t cross_dataset_results_*.json 2>/dev/null | head -1)
    if [ -f "$RESULT_FILE" ]; then
        echo -e "${BLUE}结果已保存到: $RESULT_FILE${NC}"
    fi
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}测试失败！${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

