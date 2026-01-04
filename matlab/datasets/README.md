# Earthquake Datasets - Organized Structure

## Directory Organization

The datasets folder has been organized into logical subdirectories for easy navigation and maintenance:

```
datasets/
├── test/                    # Test (held-out) earthquake signals
├── training/                # Training earthquake signals
├── scripts/                 # Python scripts for generation and analysis
├── docs/                    # Documentation files
├── analysis/                # Analysis results (plots, visualizations)
└── archive/                 # Old/deprecated files and backups
```

## Quick Start

### Run All Analysis

```bash
cd /Users/Shared/dev/git/struct-engineer-ai/matlab/datasets
./scripts/analyze_datasets.sh
```

This will:
- Generate training vs test comparison plots
- Verify baseline correction
- Create statistical summaries
- Save all visualizations to `analysis/`

### View Results

```bash
# Open visualizations
open analysis/*.png

# Read documentation
open docs/README_ANALYSIS.md
open docs/TRAIN_TEST_SPLIT_README.md
```

## Directory Details

### 📊 test/
**Purpose**: Held-out test earthquakes (NEVER used in training)

**Contents**:
- `PEER_small_M4.5_PGA0.25g.csv` (M4.5, 0.25g PGA)
- `PEER_moderate_M5.7_PGA0.35g.csv` (M5.7, 0.35g PGA)
- `PEER_high_M7.4_PGA0.75g.csv` (M7.4, 0.75g PGA)
- `PEER_insane_M8.4_PGA0.9g.csv` (M8.4, 0.9g PGA)
- Legacy test files (TEST3-TEST6 series)
- Stress test variants (noise, latency, dropout)

**Total**: 16 test earthquake files

**Usage**: Final model evaluation only

### 🎓 training/
**Purpose**: Training earthquake signals for RL model training

**Contents**:
- `training_set_v2/` - Current training set (40 files)
  - 10 × M4.5 variants (PGA 0.25g, 20s duration)
  - 10 × M5.7 variants (PGA 0.35g, 40s duration)
  - 10 × M7.4 variants (PGA 0.75g, 60s duration)
  - 10 × M8.4 variants (PGA 0.9g, 120s duration)
- `aggregated_train_80pct.csv` - Combined training file

**Total**: 40 training earthquakes + 1 aggregated file

**Usage**: RL training with random variant selection

### 🔧 scripts/
**Purpose**: Python scripts for dataset generation and analysis

**Contents**:
- **Generation Scripts**:
  - `generate_training_earthquakes_v2.py` - Create synthetic training variants
  - `fix_baseline_drift_v2.py` - Apply baseline correction and tapering

- **Analysis Scripts**:
  - `plot_train_vs_test.py` - Training vs test comparison analysis
  - `plot_all_datasets.py` - Baseline correction verification
  - `analyze_datasets.sh` - Run all analysis (executable)

- **Legacy Scripts**:
  - `generate_training_earthquakes.py` - Old version
  - `fix_baseline_drift.py` - Old version

**Total**: 7 script files

**Usage**:
```bash
# Generate training datasets
cd scripts
python generate_training_earthquakes_v2.py

# Run analysis
./analyze_datasets.sh
```

### 📖 docs/
**Purpose**: Documentation and analysis reports

**Contents**:
- `README_ANALYSIS.md` - Complete analysis suite guide
- `TRAIN_TEST_SPLIT_README.md` - Detailed train/test split analysis
- `README.md` - Original dataset documentation

**Total**: 3 documentation files

**Usage**: Read before training to understand dataset characteristics

### 📈 analysis/
**Purpose**: Generated visualizations and analysis results

**Contents**:
- **Training vs Test Analysis**:
  - `train_vs_test_comparison.png` (1.8 MB) - Detailed 4×4 grid
  - `train_vs_test_statistics.png` (372 KB) - Statistical summary
  - `train_vs_test_waveforms.png` (2.5 MB) - Waveform overlays

- **Baseline Verification**:
  - `all_datasets_visualization.png` (2.1 MB) - All datasets grouped
  - `dataset_endings_zoomed.png` (895 KB) - Taper verification
  - `baseline_correction_comparison.png` (1.5 MB) - Before/after comparison

**Total**: 6 visualization files (~9.5 MB)

**Auto-Generated**: Created by running `scripts/analyze_datasets.sh`

### 📦 archive/
**Purpose**: Old versions and backup files

**Contents**:
- `training_set/` - Old training set (v1, deprecated)
- `*.original` files - Original PEER files before correction
- `*.v1` files - Version 1 files (with baseline drift)

**Total**: 44 archived files

**Usage**: Reference only, not used in training

## Dataset Statistics

### Test Set (Held-out)

| Magnitude | PGA | RMS | Duration | Arias Intensity | File |
|-----------|-----|-----|----------|-----------------|------|
| M4.5      | 0.250g | 0.073g | 20.0s | 1.62 m/s | PEER_small_M4.5_PGA0.25g.csv |
| M5.7      | 0.350g | 0.100g | 40.0s | 6.20 m/s | PEER_moderate_M5.7_PGA0.35g.csv |
| M7.4      | 0.750g | 0.331g | 60.0s | 101.25 m/s | PEER_high_M7.4_PGA0.75g.csv |
| M8.4      | 0.900g | 0.274g | 120.0s | 139.17 m/s | PEER_insane_M8.4_PGA0.9g.csv |

### Training Set (10 variants per magnitude)

| Magnitude | PGA (mean±σ) | RMS (mean±σ) | Variants |
|-----------|--------------|--------------|----------|
| M4.5      | 0.250±0.000g | 0.060±0.006g | 10 |
| M5.7      | 0.350±0.000g | 0.080±0.005g | 10 |
| M7.4      | 0.750±0.000g | 0.170±0.015g | 10 |
| M8.4      | 0.900±0.000g | 0.189±0.018g | 10 |

### Key Findings

✅ **Perfect PGA Match**: Training and test PGA values match exactly (0.0% difference)

✅ **Challenging Test Set**: Test has 17-49% higher RMS than training (more difficult evaluation)

✅ **Proper Separation**: Zero overlap between training and test sets

✅ **Baseline Corrected**: All signals verified to have zero final acceleration

## Common Workflows

### 1. Generate New Training Datasets

```bash
cd scripts
python generate_training_earthquakes_v2.py
```

### 2. Verify Datasets

```bash
cd scripts
./analyze_datasets.sh
```

### 3. Train RL Model

```bash
cd ../../rl/rl_cl
python train_v9_advanced_ppo.py
```

The training script automatically uses:
- Training files from: `../../matlab/datasets/training/training_set_v2/`
- Test files from: `../../matlab/datasets/test/`

### 4. View Analysis

```bash
cd analysis
open *.png
```

## File Naming Conventions

### Test Files
- Format: `PEER_{size}_{magnitude}_PGA{pga}g.csv`
- Example: `PEER_small_M4.5_PGA0.25g.csv`

### Training Files
- Format: `TRAIN_{magnitude}_PGA{pga}g_RMS{rms}g_variant{n}.csv`
- Example: `TRAIN_M4.5_PGA0.25g_RMS0.073g_variant1.csv`

### Analysis Results
- Training vs Test: `train_vs_test_*.png`
- Dataset Verification: `all_datasets_*.png`, `dataset_endings_*.png`
- Baseline Comparison: `baseline_correction_*.png`

## Reorganization History

**Date**: January 4, 2026
**Script**: `reorganize_datasets.sh`

**Changes**:
- Created logical subdirectories (test/, training/, scripts/, docs/, analysis/, archive/)
- Moved 16 test files → test/
- Moved training_set_v2/ → training/
- Moved 7 scripts → scripts/
- Moved 3 docs → docs/
- Moved 6 analysis results → analysis/
- Archived 44 old files → archive/

**Benefits**:
- ✅ Clear separation of concerns
- ✅ Easy navigation
- ✅ Reduced clutter in root directory
- ✅ Logical grouping of related files
- ✅ Archive preserves history

## Maintenance

### Adding New Test Earthquake

```bash
# Place new test file
cp new_earthquake.csv test/

# Verify baseline correction
cd scripts
python fix_baseline_drift_v2.py test/new_earthquake.csv

# Re-run analysis
./analyze_datasets.sh
```

### Generating New Training Variants

```bash
cd scripts
# Edit generate_training_earthquakes_v2.py to add new magnitude
python generate_training_earthquakes_v2.py
```

### Cleaning Up Archive

```bash
# Remove very old files if needed
rm archive/*.original
rm archive/*.v1
```

## Dependencies

### Python Packages Required
```bash
pip install numpy matplotlib scipy
```

### Scripts Compatibility
- All scripts updated to work with new directory structure
- Paths are relative to datasets/ directory
- Scripts can be run from scripts/ or datasets/ directory

## Related Documentation

- [README_ANALYSIS.md](docs/README_ANALYSIS.md) - Complete analysis guide
- [TRAIN_TEST_SPLIT_README.md](docs/TRAIN_TEST_SPLIT_README.md) - Train/test split details
- [../../rl/rl_cl/README_v9_ADVANCED.md](../../rl/rl_cl/README_v9_ADVANCED.md) - Training guide

## Support

For questions or issues:
1. Check [docs/README_ANALYSIS.md](docs/README_ANALYSIS.md) for detailed analysis
2. Review visualizations in `analysis/`
3. Verify dataset organization matches this README
4. Run `scripts/analyze_datasets.sh` to regenerate analysis

---

**Author**: Siddharth
**Date**: January 2026
**Version**: 2.0 (Organized Structure)
**Status**: Production Ready