#!/bin/bash

################################################################################
# EARTHQUAKE DATASET ANALYSIS - Quick Runner Script
################################################################################
#
# This script runs all dataset visualization and analysis scripts:
# 1. Training vs Test comparison
# 2. Baseline correction verification
# 3. Dataset endings verification
#
# Usage:
#   cd /Users/Shared/dev/git/struct-engineer-ai/matlab/datasets
#   ./scripts/analyze_datasets.sh
#   OR
#   cd /Users/Shared/dev/git/struct-engineer-ai/matlab/datasets/scripts
#   ./analyze_datasets.sh
#
# Author: Siddharth
# Date: January 2026
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="/Users/Shared/dev/git/struct-engineer-ai"
DATASETS_DIR="${PROJECT_ROOT}/matlab/datasets"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        EARTHQUAKE DATASET ANALYSIS - VISUALIZATION SUITE          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if we're in the right directory (support running from scripts/ or datasets/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$SCRIPT_DIR" == *"/scripts" ]]; then
    # Running from scripts directory
    cd "${SCRIPT_DIR}/.." || exit 1
elif [[ "$SCRIPT_DIR" == *"/datasets" ]]; then
    # Running from datasets directory
    cd "${SCRIPT_DIR}" || exit 1
else
    # Try to navigate to datasets
    cd "${DATASETS_DIR}" || {
        echo -e "${RED}Error: Could not change to datasets directory${NC}"
        echo "Expected: ${DATASETS_DIR}"
        exit 1
    }
fi

# Check if Python venv exists
if [ ! -f "${PYTHON}" ]; then
    echo -e "${RED}Error: Python virtual environment not found${NC}"
    echo "Expected: ${PYTHON}"
    echo ""
    echo "Please create virtual environment first:"
    echo "  cd ${PROJECT_ROOT}"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install numpy matplotlib scipy"
    exit 1
fi

# Check for required Python packages
echo -e "${YELLOW}Checking Python dependencies...${NC}"
${PYTHON} -c "import numpy, matplotlib, scipy" 2>/dev/null || {
    echo -e "${RED}Error: Missing required Python packages${NC}"
    echo ""
    echo "Install with:"
    echo "  source ${PROJECT_ROOT}/.venv/bin/activate"
    echo "  pip install numpy matplotlib scipy"
    exit 1
}
echo -e "${GREEN}✓ All dependencies found${NC}"
echo ""

# Run analysis scripts
echo -e "${BLUE}════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  1/2: Training vs Test Dataset Comparison${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════${NC}"
echo ""

if [ -f "scripts/plot_train_vs_test.py" ]; then
    ${PYTHON} scripts/plot_train_vs_test.py
    echo ""
else
    echo -e "${RED}⚠ Script not found: scripts/plot_train_vs_test.py${NC}"
fi

echo -e "${BLUE}════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  2/2: All Datasets Visualization${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════${NC}"
echo ""

if [ -f "scripts/plot_all_datasets.py" ]; then
    ${PYTHON} scripts/plot_all_datasets.py
    echo ""
else
    echo -e "${RED}⚠ Script not found: scripts/plot_all_datasets.py${NC}"
fi

# Summary
echo -e "${BLUE}════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  ANALYSIS COMPLETE!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${GREEN}Generated Visualizations:${NC}"
echo ""

echo "📊 Training vs Test Comparison:"
for file in analysis/train_vs_test_*.png; do
    if [ -f "$file" ]; then
        size=$(ls -lh "$file" | awk '{print $5}')
        basename_file=$(basename "$file")
        echo -e "   ${GREEN}✓${NC} $basename_file (${size})"
    fi
done

echo ""
echo "📊 Dataset Verification:"
for file in analysis/all_datasets_visualization.png analysis/baseline_correction_comparison.png analysis/dataset_endings_zoomed.png; do
    if [ -f "$file" ]; then
        size=$(ls -lh "$file" | awk '{print $5}')
        basename_file=$(basename "$file")
        echo -e "   ${GREEN}✓${NC} $basename_file (${size})"
    fi
done

echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Review visualizations in: ${DATASETS_DIR}/analysis/"
echo "  2. Read analysis: ${DATASETS_DIR}/docs/TRAIN_TEST_SPLIT_README.md"
echo "  3. Verify train/test split is proper"
echo "  4. Begin RL training with confidence!"
echo ""

echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✓ All dataset analysis complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
echo ""
