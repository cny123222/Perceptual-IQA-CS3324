#!/bin/bash

################################################################################
# Overnight Experiment Launcher
# 
# This script launches the main experiment runner in a tmux session,
# so it will continue running even after SSH disconnection.
################################################################################

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Base directory
BASE_DIR="/root/Perceptual-IQA-CS3324"
cd "$BASE_DIR"

echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║          🌙 Overnight Experiment Launcher (SSH-safe) 🌙           ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo -e "${RED}❌ tmux is not installed!${NC}"
    echo -e "${YELLOW}Installing tmux...${NC}"
    apt-get update && apt-get install -y tmux
    echo -e "${GREEN}✅ tmux installed!${NC}"
    echo ""
fi

# Check if runner session already exists
if tmux has-session -t exp-runner 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Existing 'exp-runner' session found!${NC}"
    echo -e "${YELLOW}   Do you want to:${NC}"
    echo -e "   ${BLUE}1)${NC} Kill the existing session and start fresh"
    echo -e "   ${BLUE}2)${NC} Attach to the existing session"
    echo -e "   ${BLUE}3)${NC} Cancel"
    echo ""
    read -p "Choice (1/2/3): " choice
    
    case $choice in
        1)
            echo -e "${YELLOW}Killing existing session...${NC}"
            tmux kill-session -t exp-runner
            ;;
        2)
            echo -e "${GREEN}Attaching to existing session...${NC}"
            echo -e "${CYAN}Press Ctrl+B then D to detach without stopping${NC}"
            sleep 2
            tmux attach -t exp-runner
            exit 0
            ;;
        3)
            echo -e "${YELLOW}Cancelled.${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice. Exiting.${NC}"
            exit 1
            ;;
    esac
fi

# Display experiment info
echo -e "${CYAN}📊 Experiment Overview:${NC}"
echo -e "  ${BLUE}Total experiments:${NC} 14"
echo -e "  ${BLUE}Total time:${NC} ~6 hours"
echo -e "  ${BLUE}GPUs:${NC} 4 (parallel execution)"
echo ""
echo -e "${CYAN}📋 Execution Plan:${NC}"
echo -e "  ${BLUE}Batch 1 (1.5h):${NC} A1, A2, A3, C1"
echo -e "  ${BLUE}Batch 2 (1.5h):${NC} C2, C3, B1, B2"
echo -e "  ${BLUE}Batch 3 (1.5h):${NC} D1, D2, D4, E1"
echo -e "  ${BLUE}Batch 4 (1.5h):${NC} E3, E4"
echo ""
echo -e "${GREEN}🔒 SSH-safe:${NC} Running in tmux session 'exp-runner'"
echo -e "${GREEN}   You can safely disconnect SSH after starting!${NC}"
echo ""

# Confirmation
echo -e "${YELLOW}Press Enter to start, or Ctrl+C to cancel...${NC}"
read

# Create and start tmux session
echo -e "${GREEN}🚀 Starting experiment runner in tmux session...${NC}"
echo ""

tmux new-session -d -s exp-runner "cd $BASE_DIR && ./run_experiments_4gpus.sh; echo ''; echo 'Experiments complete! Press Enter to close.'; read"

# Give tmux a moment to start
sleep 1

echo -e "${GREEN}✅ Experiment runner started successfully!${NC}"
echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                    📝 Important Commands 📝                        ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}1) View experiment runner progress:${NC}"
echo -e "   ${GREEN}tmux attach -t exp-runner${NC}"
echo ""
echo -e "${BLUE}2) Detach from tmux (keep it running):${NC}"
echo -e "   ${GREEN}Press: Ctrl+B, then D${NC}"
echo ""
echo -e "${BLUE}3) Check individual experiment:${NC}"
echo -e "   ${GREEN}tmux attach -t exp-a1${NC}  (or exp-a2, exp-b1, etc.)"
echo ""
echo -e "${BLUE}4) List all tmux sessions:${NC}"
echo -e "   ${GREEN}tmux ls${NC}"
echo ""
echo -e "${BLUE}5) Monitor GPU usage:${NC}"
echo -e "   ${GREEN}watch -n 1 nvidia-smi${NC}"
echo ""
echo -e "${BLUE}6) Check recent logs:${NC}"
echo -e "   ${GREEN}ls -lth logs/ | head -20${NC}"
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}⚡ Attaching to runner session in 3 seconds...${NC}"
echo -e "${YELLOW}   (Press Ctrl+B then D to detach and let it run overnight)${NC}"
sleep 3

# Attach to the session
tmux attach -t exp-runner

echo ""
echo -e "${GREEN}✨ Detached from runner. Experiments continue in background!${NC}"
echo -e "${CYAN}   Reconnect anytime with: ${GREEN}tmux attach -t exp-runner${NC}"
echo ""

