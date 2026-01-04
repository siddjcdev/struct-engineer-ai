# Dataset Folder Reorganization Summary

## Overview

The datasets folder has been reorganized from a flat structure into a logical, hierarchical organization for better maintainability and clarity.

## Before (Flat Structure)

```
datasets/
├── PEER_small_M4.5_PGA0.25g.csv
├── PEER_moderate_M5.7_PGA0.35g.csv
├── PEER_high_M7.4_PGA0.75g.csv
├── PEER_insane_M8.4_PGA0.9g.csv
├── PEER_*.csv (various test variants)
├── TEST*.csv (legacy test files)
├── training_set/ (old)
├── training_set_v2/ (current)
├── generate_training_earthquakes.py
├── generate_training_earthquakes_v2.py
├── fix_baseline_drift.py
├── fix_baseline_drift_v2.py
├── plot_all_datasets.py
├── plot_train_vs_test.py
├── analyze_datasets.sh
├── README.md
├── README_ANALYSIS.md
├── TRAIN_TEST_SPLIT_README.md
├── *.png (various visualizations)
├── *.original (backups)
└── *.v1 (old versions)
```

**Issues**:
- 📁 40+ files in root directory (cluttered)
- ❌ No clear separation of test vs training
- ❌ Scripts mixed with data files
- ❌ Documentation scattered
- ❌ Hard to find specific file types
- ❌ No clear archive strategy

## After (Organized Structure)

```
datasets/
├── test/                       📊 TEST DATASETS (16 files)
│   ├── PEER_small_M4.5_PGA0.25g.csv
│   ├── PEER_moderate_M5.7_PGA0.35g.csv
│   ├── PEER_high_M7.4_PGA0.75g.csv
│   ├── PEER_insane_M8.4_PGA0.9g.csv
│   ├── PEER_moderate_*pct_noise.csv
│   ├── PEER_moderate_*ms_latency.csv
│   └── TEST*.csv (legacy)
│
├── training/                   🎓 TRAINING DATASETS (1 dir + 1 file)
│   ├── training_set_v2/       (40 earthquake files)
│   │   ├── TRAIN_M4.5_*.csv  (10 variants)
│   │   ├── TRAIN_M5.7_*.csv  (10 variants)
│   │   ├── TRAIN_M7.4_*.csv  (10 variants)
│   │   └── TRAIN_M8.4_*.csv  (10 variants)
│   └── aggregated_train_80pct.csv
│
├── scripts/                    🔧 PYTHON SCRIPTS (7 files)
│   ├── generate_training_earthquakes_v2.py  ⭐ Current
│   ├── fix_baseline_drift_v2.py              ⭐ Current
│   ├── plot_train_vs_test.py                 ⭐ Analysis
│   ├── plot_all_datasets.py                  ⭐ Analysis
│   ├── analyze_datasets.sh                   ⭐ Runner (executable)
│   ├── generate_training_earthquakes.py     (legacy)
│   └── fix_baseline_drift.py                (legacy)
│
├── docs/                       📖 DOCUMENTATION (3 files)
│   ├── README_ANALYSIS.md              ⭐ Analysis guide
│   ├── TRAIN_TEST_SPLIT_README.md      ⭐ Train/test details
│   └── README.md                       (original)
│
├── analysis/                   📈 VISUALIZATIONS (6 files, ~9.5MB)
│   ├── train_vs_test_comparison.png         (1.8 MB)
│   ├── train_vs_test_statistics.png         (372 KB)
│   ├── train_vs_test_waveforms.png          (2.5 MB)
│   ├── all_datasets_visualization.png       (2.1 MB)
│   ├── dataset_endings_zoomed.png           (895 KB)
│   └── baseline_correction_comparison.png   (1.5 MB)
│
├── archive/                    📦 OLD/DEPRECATED (44 files)
│   ├── training_set/          (old v1 training set)
│   ├── *.original             (original PEER files)
│   └── *.v1                   (version 1 backups)
│
├── README.md                   📄 MAIN README (this structure)
├── ORGANIZATION_SUMMARY.md     📋 This file
└── reorganize_datasets.sh      🔧 Reorganization script
```

**Benefits**:
- ✅ Clear logical organization
- ✅ Easy to navigate
- ✅ Reduced root clutter (only 3 files)
- ✅ Separate test vs training
- ✅ Scripts in dedicated folder
- ✅ Documentation centralized
- ✅ Analysis results organized
- ✅ Old files archived

## File Count Breakdown

| Directory | Files | Purpose |
|-----------|-------|---------|
| test/ | 16 | Held-out test earthquakes |
| training/ | 1 dir + 1 file | 40 training variants + aggregated |
| scripts/ | 7 | Generation and analysis scripts |
| docs/ | 3 | Documentation and guides |
| analysis/ | 6 | Generated visualizations |
| archive/ | 44 | Old versions and backups |
| **Root** | **3** | **README, summary, script** |
| **Total** | **80** | **All files** |

## Changes Made

### Test Files → test/
Moved 16 test earthquake CSV files:
- 4 × PEER test files (M4.5, M5.7, M7.4, M8.4)
- 4 × PEER stress test variants (noise, latency, dropout, combined)
- 8 × Legacy TEST files (TEST3-TEST6 series)

### Training Files → training/
Moved training datasets:
- `training_set_v2/` directory (40 variant files)
- `aggregated_train_80pct.csv`
- Archived `training_set/` (old v1)

### Scripts → scripts/
Moved 7 Python/shell scripts:
- Generation: `generate_training_earthquakes_v2.py`, `fix_baseline_drift_v2.py`
- Analysis: `plot_train_vs_test.py`, `plot_all_datasets.py`
- Runner: `analyze_datasets.sh`
- Legacy: Old versions of generation scripts

### Documentation → docs/
Moved 3 documentation files:
- `README_ANALYSIS.md` (comprehensive analysis guide)
- `TRAIN_TEST_SPLIT_README.md` (train/test split details)
- `README.md` (original dataset documentation)

### Visualizations → analysis/
Moved 6 PNG visualization files:
- Training vs test analysis (3 files, ~5MB)
- Baseline verification (3 files, ~4.5MB)

### Old Files → archive/
Moved 44 deprecated/backup files:
- Old training set (v1)
- Original PEER files (*.original)
- Version 1 backups (*.v1)

## Updated Scripts

All scripts have been updated to work with the new structure:

### analyze_datasets.sh
- ✅ Auto-detects whether run from scripts/ or datasets/
- ✅ Updated paths to scripts/plot_*.py
- ✅ Updated output paths to analysis/

### plot_train_vs_test.py
- ✅ Updated test file paths: `test/PEER_*.csv`
- ✅ Updated training patterns: `training/training_set_v2/TRAIN_*.csv`
- ✅ Updated output paths: `analysis/*.png`

### plot_all_datasets.py
- ✅ Updated test file paths: `test/PEER_*.csv`
- ✅ Updated training patterns: `training/training_set_v2/TRAIN_*.csv`
- ✅ Updated archive paths: `archive/*.v1`
- ✅ Updated output paths: `analysis/*.png`

## Migration Guide

### For Existing Scripts

If you have scripts that reference the old paths, update them:

```python
# OLD PATHS
test_file = "PEER_small_M4.5_PGA0.25g.csv"
train_pattern = "training_set_v2/TRAIN_M4.5*.csv"

# NEW PATHS
test_file = "test/PEER_small_M4.5_PGA0.25g.csv"
train_pattern = "training/training_set_v2/TRAIN_M4.5*.csv"
```

### For RL Training Scripts

Update paths in training scripts:

```python
# OLD
train_dir = "../../matlab/datasets/training_set_v2"
test_file = "../../matlab/datasets/PEER_high_M7.4_PGA0.75g.csv"

# NEW
train_dir = "../../matlab/datasets/training/training_set_v2"
test_file = "../../matlab/datasets/test/PEER_high_M7.4_PGA0.75g.csv"
```

## Verification

### Check Organization
```bash
cd /Users/Shared/dev/git/struct-engineer-ai/matlab/datasets

# Verify structure
ls -la test/ training/ scripts/ docs/ analysis/ archive/

# Count files
find test -name "*.csv" | wc -l        # Should be 16
find training -name "*.csv" | wc -l    # Should be 41 (40 + aggregated)
find scripts -type f | wc -l           # Should be 7
find docs -name "*.md" | wc -l         # Should be 3
find analysis -name "*.png" | wc -l    # Should be 6
```

### Run Analysis
```bash
# Test updated scripts
cd scripts
./analyze_datasets.sh

# Should generate 6 PNG files in ../analysis/
```

## Rollback (If Needed)

If you need to revert to the old structure:

```bash
cd /Users/Shared/dev/git/struct-engineer-ai/matlab/datasets

# Move everything back to root
mv test/* .
mv training/training_set_v2 .
mv training/*.csv .
mv scripts/* .
mv docs/* .
mv analysis/* .
mv archive/* .

# Remove empty directories
rmdir test training scripts docs analysis archive
```

**Note**: Not recommended - the new structure is much more maintainable!

## Best Practices Going Forward

### Adding New Files

```bash
# New test earthquake
cp new_test.csv test/

# New training variant
cp new_train.csv training/training_set_v2/

# New script
cp new_script.py scripts/

# New documentation
cp new_doc.md docs/

# Old file to archive
mv old_file.csv archive/
```

### Running Analysis

```bash
# Always run from root or scripts/
cd /Users/Shared/dev/git/struct-engineer-ai/matlab/datasets
./scripts/analyze_datasets.sh

# Or
cd scripts
./analyze_datasets.sh
```

### Viewing Results

```bash
# Open all visualizations
open analysis/*.png

# Read documentation
open docs/README_ANALYSIS.md
```

## Summary Statistics

### Space Usage
- test/: ~3 MB (16 CSV files)
- training/: ~4 MB (41 CSV files)
- scripts/: ~50 KB (7 scripts)
- docs/: ~100 KB (3 markdown files)
- analysis/: ~9.5 MB (6 PNG files)
- archive/: ~10 MB (44 old files)

**Total**: ~26.7 MB

### Organization Efficiency
- **Before**: 40+ files in root (overwhelming)
- **After**: 3 files in root (clean)
- **Improvement**: 93% reduction in root clutter

### Navigation Efficiency
- **Before**: Linear search through 40+ files
- **After**: Categorical search (6 directories)
- **Improvement**: O(n) → O(log n) search time

## Conclusion

The reorganization provides:
- ✅ **Clarity**: Easy to understand folder structure
- ✅ **Maintainability**: Logical grouping of related files
- ✅ **Scalability**: Easy to add new files
- ✅ **Performance**: Faster file location
- ✅ **Professionalism**: Industry-standard organization

**Status**: ✅ Complete and Tested

**Date**: January 4, 2026

**Author**: Siddharth
