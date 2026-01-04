#!/bin/bash

################################################################################
# Dataset Folder Reorganization Script
################################################################################
# Organizes datasets folder into logical subdirectories:
# - test/          : Test (held-out) earthquake signals
# - training/      : Training earthquake signals
# - scripts/       : Python scripts for generation and analysis
# - docs/          : Documentation files
# - analysis/      : Analysis results (plots, visualizations)
# - archive/       : Old/deprecated files
################################################################################

set -e

echo "════════════════════════════════════════════════════════════════"
echo "  REORGANIZING DATASETS FOLDER"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Create directory structure
echo "Creating directory structure..."
mkdir -p test
mkdir -p training
mkdir -p scripts
mkdir -p docs
mkdir -p analysis
mkdir -p archive

# Move test files (current version only)
echo ""
echo "Moving test files..."
mv -v PEER_small_M4.5_PGA0.25g.csv test/ 2>/dev/null || true
mv -v PEER_moderate_M5.7_PGA0.35g.csv test/ 2>/dev/null || true
mv -v PEER_high_M7.4_PGA0.75g.csv test/ 2>/dev/null || true
mv -v PEER_insane_M8.4_PGA0.9g.csv test/ 2>/dev/null || true

# Move stress test files
mv -v PEER_moderate_10pct_noise.csv test/ 2>/dev/null || true
mv -v PEER_moderate_60ms_latency.csv test/ 2>/dev/null || true
mv -v PEER_moderate_8pct_dropout.csv test/ 2>/dev/null || true
mv -v PEER_moderate_combined_stress.csv test/ 2>/dev/null || true

# Move legacy test files
mv -v TEST3_small_earthquake_M4.5.csv test/ 2>/dev/null || true
mv -v TEST4_large_earthquake_M6.9.csv test/ 2>/dev/null || true
mv -v TEST5_earthquake_M6.7.csv test/ 2>/dev/null || true
mv -v TEST6a_baseline_clean.csv test/ 2>/dev/null || true
mv -v TEST6b_with_10pct_noise.csv test/ 2>/dev/null || true
mv -v TEST6c_with_50ms_latency.csv test/ 2>/dev/null || true
mv -v TEST6d_with_5pct_dropout.csv test/ 2>/dev/null || true
mv -v TEST6e_combined_stress.csv test/ 2>/dev/null || true

# Move training directories (keep current version in main, archive old)
echo ""
echo "Moving training files..."
if [ -d "training_set" ]; then
    mv -v training_set archive/ 2>/dev/null || true
fi
if [ -d "training_set_v2" ]; then
    mv -v training_set_v2 training/ 2>/dev/null || true
fi

# Move aggregated training file
mv -v aggregated_train_80pct.csv training/ 2>/dev/null || true

# Move scripts
echo ""
echo "Moving scripts..."
mv -v generate_training_earthquakes.py scripts/ 2>/dev/null || true
mv -v generate_training_earthquakes_v2.py scripts/ 2>/dev/null || true
mv -v fix_baseline_drift.py scripts/ 2>/dev/null || true
mv -v fix_baseline_drift_v2.py scripts/ 2>/dev/null || true
mv -v plot_all_datasets.py scripts/ 2>/dev/null || true
mv -v plot_train_vs_test.py scripts/ 2>/dev/null || true
mv -v analyze_datasets.sh scripts/ 2>/dev/null || true

# Move documentation
echo ""
echo "Moving documentation..."
mv -v README.md docs/ 2>/dev/null || true
mv -v README_ANALYSIS.md docs/ 2>/dev/null || true
mv -v TRAIN_TEST_SPLIT_README.md docs/ 2>/dev/null || true

# Move analysis results (plots)
echo ""
echo "Moving analysis results..."
mv -v all_datasets_visualization.png analysis/ 2>/dev/null || true
mv -v baseline_correction_comparison.png analysis/ 2>/dev/null || true
mv -v dataset_endings_zoomed.png analysis/ 2>/dev/null || true
mv -v train_vs_test_comparison.png analysis/ 2>/dev/null || true
mv -v train_vs_test_statistics.png analysis/ 2>/dev/null || true
mv -v train_vs_test_waveforms.png analysis/ 2>/dev/null || true

# Move old versions to archive
echo ""
echo "Moving old/backup files to archive..."
mv -v *.original archive/ 2>/dev/null || true
mv -v *.v1 archive/ 2>/dev/null || true

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  REORGANIZATION COMPLETE!"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Summary
echo "Directory Summary:"
echo "  test/       : $(find test -type f -name '*.csv' 2>/dev/null | wc -l | tr -d ' ') CSV files"
echo "  training/   : $(find training -type d -mindepth 1 -maxdepth 1 2>/dev/null | wc -l | tr -d ' ') subdirectories"
echo "  scripts/    : $(find scripts -type f 2>/dev/null | wc -l | tr -d ' ') files"
echo "  docs/       : $(find docs -type f 2>/dev/null | wc -l | tr -d ' ') files"
echo "  analysis/   : $(find analysis -type f 2>/dev/null | wc -l | tr -d ' ') files"
echo "  archive/    : $(find archive -type f 2>/dev/null | wc -l | tr -d ' ') files"
echo ""

