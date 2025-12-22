#!/bin/bash

echo "═══════════════════════════════════════════════════════════════"
echo "          All Experiments Results Summary"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Function to extract results
extract_results() {
    local file=$1
    local name=$2
    
    if [ -f "$file" ]; then
        echo "📊 $name"
        echo "───────────────────────────────────────────────────────────────"
        
        # Extract median SRCC and PLCC
        median_line=$(grep "median SRCC" "$file" | tail -1)
        if [ -n "$median_line" ]; then
            echo "  $median_line"
        else
            echo "  ⏳ Experiment not finished yet or no results"
        fi
        
        # Extract best SRCC and PLCC
        best_line=$(grep "Best test SRCC" "$file" | tail -1)
        if [ -n "$best_line" ]; then
            echo "  $best_line"
        fi
        
        # Extract training config
        lr=$(grep "Learning Rate:" "$file" | head -1 | awk '{print $3}')
        model=$(grep "Model Size:" "$file" | head -1 | awk '{print $3}')
        multiscale=$(grep "Multi-Scale Fusion:" "$file" | head -1 | awk '{print $3}')
        attention=$(grep "Attention Fusion:" "$file" | head -1 | awk '{print $3}')
        
        echo "  Config: Model=$model, LR=$lr, Multi-scale=$multiscale, Attention=$attention"
        echo ""
    fi
}

echo "🔬 PHASE 1: Learning Rate Comparison"
echo ""
extract_results "phase1_lr1e6.out" "LR = 1e-6 (GPU 0)"
extract_results "phase1_lr5e7.out" "LR = 5e-7 (GPU 1)"

echo "🔬 PHASE 2: Ablation Studies"
echo ""
extract_results "phase2_A1_no_attention.out" "A1: Remove Attention (GPU 0)"
extract_results "phase2_A2_no_multiscale.out" "A2: Remove Multi-scale (GPU 1)"

echo "🔬 PHASE 3: Model Size Comparison"
echo ""
extract_results "phase3_B1_tiny.out" "B1: Swin-Tiny (GPU 0)"
extract_results "phase3_B2_small.out" "B2: Swin-Small (GPU 1)"

echo "═══════════════════════════════════════════════════════════════"
echo "          Quick Comparison Table"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "| Experiment | LR | Model | SRCC | PLCC |"
echo "|------------|-----|-------|------|------|"

# Extract and format for table
for file in phase1_lr1e6.out phase1_lr5e7.out phase2_A1_no_attention.out phase2_A2_no_multiscale.out phase3_B1_tiny.out phase3_B2_small.out; do
    if [ -f "$file" ]; then
        name=$(basename "$file" .out)
        lr=$(grep "Learning Rate:" "$file" 2>/dev/null | head -1 | awk '{print $3}')
        model=$(grep "Model Size:" "$file" 2>/dev/null | head -1 | awk '{print $3}')
        srcc=$(grep "median SRCC" "$file" 2>/dev/null | tail -1 | grep -oP "SRCC \K[0-9.]+")
        plcc=$(grep "median PLCC" "$file" 2>/dev/null | tail -1 | grep -oP "PLCC \K[0-9.]+")
        
        if [ -n "$srcc" ]; then
            echo "| $name | $lr | $model | $srcc | $plcc |"
        else
            echo "| $name | $lr | $model | - | - |"
        fi
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Save this output: ./extract_all_results.sh > RESULTS_SUMMARY.txt"
echo "═══════════════════════════════════════════════════════════════"

