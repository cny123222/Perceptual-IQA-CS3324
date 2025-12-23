#!/bin/bash

# 保留以下重要的checkpoint目录:
# - koniq-10k-swin_20251223_002219 (batch1 lr1e-6)
# - koniq-10k-swin_20251223_002226 (batch1 lr5e-7)
# - koniq-10k-swin_20251223_035309 (batch3 B2 small)
# - koniq-10k-swin_20251223_035433 (batch3 B1 tiny)
# - koniq-10k-swin_20251222_161625 (best baseline 0.9354)
# - koniq-10k-resnet_20251221_004809 (ResNet baseline for comparison)

KEEP_DIRS=(
    "koniq-10k-swin_20251223_002219"
    "koniq-10k-swin_20251223_002226"
    "koniq-10k-swin_20251223_035309"
    "koniq-10k-swin_20251223_035433"
    "koniq-10k-swin_20251222_161625"
    "koniq-10k-resnet_20251221_004809"
)

cd /root/Perceptual-IQA-CS3324/checkpoints

echo "清理旧的checkpoint目录..."
DELETED=0
KEPT=0
FREED=0

for dir in koniq-10k-*; do
    if [ -d "$dir" ]; then
        SHOULD_KEEP=0
        for keep in "${KEEP_DIRS[@]}"; do
            if [ "$dir" = "$keep" ]; then
                SHOULD_KEEP=1
                break
            fi
        done
        
        if [ $SHOULD_KEEP -eq 0 ]; then
            SIZE=$(du -sm "$dir" | cut -f1)
            echo "删除: $dir (${SIZE}MB)"
            rm -rf "$dir"
            DELETED=$((DELETED + 1))
            FREED=$((FREED + SIZE))
        else
            echo "保留: $dir"
            KEPT=$((KEPT + 1))
        fi
    fi
done

echo ""
echo "清理完成!"
echo "  删除: $DELETED 个目录"
echo "  保留: $KEPT 个目录"
echo "  释放: ${FREED}MB 空间"
