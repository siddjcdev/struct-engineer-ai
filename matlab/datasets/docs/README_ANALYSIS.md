# Earthquake Dataset Analysis Suite

## Overview

This directory contains comprehensive tools for analyzing, visualizing, and verifying the earthquake datasets used for RL training.

## Quick Start

### Run All Analysis (Recommended)

```bash
cd /Users/Shared/dev/git/struct-engineer-ai/matlab/datasets
./analyze_datasets.sh
```

This will generate all visualizations and statistics automatically.

### Manual Execution

```bash
# Training vs Test comparison
/Users/Shared/dev/git/struct-engineer-ai/.venv/bin/python plot_train_vs_test.py

# All datasets visualization
/Users/Shared/dev/git/struct-engineer-ai/.venv/bin/python plot_all_datasets.py
```

## Analysis Scripts

### 1. Training vs Test Comparison (`plot_train_vs_test.py`)

**Purpose**: Comprehensive comparison of training and test datasets to verify proper train/test split.

**Generates**:
- `train_vs_test_comparison.png` (1.8 MB)
  - 4×4 grid with detailed per-magnitude analysis
  - Time-domain waveforms overlaid
  - PGA distribution (violin plots)
  - RMS distribution
  - Power spectral density (frequency domain)

- `train_vs_test_statistics.png` (372 KB)
  - Statistical summary across all magnitudes
  - Bar charts comparing test vs training
  - Error bars showing variance
  - Summary table with differences

- `train_vs_test_waveforms.png` (2.5 MB)
  - All waveforms overlaid by magnitude
  - Test signals highlighted in red
  - Training variants in color-coded layers

**Key Metrics Analyzed**:
- Peak Ground Acceleration (PGA)
- RMS Acceleration
- Duration
- Arias Intensity (energy content)
- Dominant Frequency
- Power Spectral Density

### 2. All Datasets Visualization (`plot_all_datasets.py`)

**Purpose**: Verify baseline correction and tapering for all earthquake signals.

**Generates**:
- `all_datasets_visualization.png` (2.1 MB)
  - All earthquakes grouped by magnitude
  - Shows baseline correction effectiveness
  - Highlights taper region (last 5s)

- `dataset_endings_zoomed.png` (895 KB)
  - Zoomed view of last 10 seconds
  - Verifies zero final acceleration
  - Shows cosine taper application

- `baseline_correction_comparison.png` (1.5 MB)
  - Before/after comparison (V1 vs V2)
  - Shows drift reduction
  - Velocity and displacement integration

## Dataset Organization

### Directory Structure

```
matlab/datasets/
├── PEER_small_M4.5_PGA0.25g.csv           # Test (M4.5)
├── PEER_moderate_M5.7_PGA0.35g.csv        # Test (M5.7)
├── PEER_high_M7.4_PGA0.75g.csv            # Test (M7.4)
├── PEER_insane_M8.4_PGA0.9g.csv           # Test (M8.4)
├── training_set_v2/
│   ├── TRAIN_M4.5_*.csv                   # 10 variants
│   ├── TRAIN_M5.7_*.csv                   # 10 variants
│   ├── TRAIN_M7.4_*.csv                   # 10 variants
│   └── TRAIN_M8.4_*.csv                   # 10 variants
├── plot_train_vs_test.py                  # Analysis script
├── plot_all_datasets.py                   # Visualization script
├── analyze_datasets.sh                    # Quick runner
├── TRAIN_TEST_SPLIT_README.md             # Detailed analysis
└── README_ANALYSIS.md                     # This file
```

### Training Set (40 earthquakes)
- **M4.5**: 10 synthetic variants (PGA 0.25g, 20s duration)
- **M5.7**: 10 synthetic variants (PGA 0.35g, 40s duration)
- **M7.4**: 10 synthetic variants (PGA 0.75g, 60s duration)
- **M8.4**: 10 synthetic variants (PGA 0.9g, 120s duration)

### Test Set (4 earthquakes - HELD OUT)
- **M4.5**: PEER ground motion (NEVER used in training)
- **M5.7**: PEER ground motion (NEVER used in training)
- **M7.4**: PEER ground motion (NEVER used in training)
- **M8.4**: PEER ground motion (NEVER used in training)

## Key Findings

### ✅ Proper Train/Test Split Verified

1. **Zero Overlap**: Test signals are real PEER recordings, training uses synthetic variants
2. **PGA Matching**: Perfect match (0.0% difference) across all magnitudes
3. **Distribution Coverage**: Training variants cover appropriate ranges
4. **Challenging Test Set**: Test has 17-49% higher RMS (more difficult)

### Statistical Summary

| Magnitude | Datasets | Test PGA | Train PGA (μ±σ) | RMS Diff |
|-----------|----------|----------|-----------------|----------|
| M4.5      | 10+1     | 0.250g   | 0.250±0.000g    | 17.7%    |
| M5.7      | 10+1     | 0.350g   | 0.350±0.000g    | 20.1%    |
| M7.4      | 10+1     | 0.750g   | 0.750±0.000g    | 48.6%    |
| M8.4      | 10+1     | 0.900g   | 0.900±0.000g    | 31.1%    |

### Baseline Correction Verified

All signals verified to have:
- ✅ Zero final acceleration (< 1e-10 m/s²)
- ✅ Smooth cosine taper (last 5 seconds)
- ✅ Minimal baseline drift
- ✅ Physical validity (integrable to displacement)

## Interpretation Guide

### What to Look For

#### ✅ Good Signs:
- Test PGA matches training PGA (magnitude calibration correct)
- Test RMS is similar or higher than training (challenging evaluation)
- Training variants show diversity (prevents overfitting)
- Frequency content spans realistic range
- Zero final acceleration (no drift)

#### ⚠️ Warning Signs:
- Test PGA significantly different from training (wrong magnitude)
- Test RMS much lower than training (too easy evaluation)
- All training variants identical (no diversity)
- Non-zero final acceleration (baseline drift)

### Understanding RMS Differences

The test earthquakes have **higher RMS** than training:
- **M4.5**: 17.7% higher
- **M5.7**: 20.1% higher
- **M7.4**: 48.6% higher
- **M8.4**: 31.1% higher

**Why This Is Good**:
1. Makes evaluation more challenging (conservative estimates)
2. Tests generalization capability
3. Prevents overfitting to training distribution
4. Realistic worst-case scenarios

## Usage in RL Training

### Curriculum Learning

The v9 Advanced PPO training uses these datasets in a curriculum:

```python
Stage 1: M4.5 @ 50kN  (300k steps) - Learn basics
Stage 2: M5.7 @ 100kN (300k steps) - Moderate difficulty
Stage 3: M7.4 @ 150kN (400k steps) - High difficulty
Stage 4: M8.4 @ 150kN (400k steps) - Extreme difficulty
```

### Random Selection

During training, earthquakes are randomly selected from variants:
```python
eq_file = random.choice(train_files)
env = make_improved_tmd_env(eq_file, max_force=force_limit)
```

This provides data augmentation and prevents memorization.

### Evaluation Protocol

**Training**: Use all 10 variants per magnitude (random selection)

**Testing**: Use only the single held-out PEER signal (deterministic)

**Metrics**:
- Peak roof displacement (cm)
- RMS acceleration
- Demand-Capacity Ratio (DCR)
- Percentage improvement vs uncontrolled

## Expected Performance

Based on the dataset characteristics:

### M4.5 (Easy)
- Training should achieve near-perfect control
- Test should match or slightly exceed training
- Target: >90% reduction vs uncontrolled

### M5.7 (Moderate)
- Training should achieve excellent control
- Test should be close to training performance
- Target: >85% reduction vs uncontrolled

### M7.4 (Challenging)
- Training should achieve good control
- Test is significantly harder (48% higher RMS)
- Target: >70% reduction vs uncontrolled
- **This is the key benchmark for performance**

### M8.4 (Extreme)
- Training should show some control capability
- Test is harder (31% higher RMS)
- Target: >50% reduction vs uncontrolled
- Partial success expected (very challenging)

## Troubleshooting

### No Visualizations Generated

**Problem**: Script runs but no PNG files created

**Solutions**:
1. Check Python backend: `matplotlib.use('Agg')`
2. Verify write permissions: `ls -la *.png`
3. Check for errors in script output

### Missing Training Files

**Problem**: "No training files found"

**Solutions**:
1. Generate training set first:
   ```bash
   python generate_training_earthquakes_v2.py
   ```
2. Check directory: `ls training_set_v2/`
3. Verify file patterns match: `TRAIN_M*.csv`

### Python Module Errors

**Problem**: `ModuleNotFoundError: No module named 'numpy'`

**Solutions**:
```bash
# Activate virtual environment
source /Users/Shared/dev/git/struct-engineer-ai/.venv/bin/activate

# Install required packages
pip install numpy matplotlib scipy

# Use correct Python
/Users/Shared/dev/git/struct-engineer-ai/.venv/bin/python plot_train_vs_test.py
```

## Related Documentation

- [TRAIN_TEST_SPLIT_README.md](TRAIN_TEST_SPLIT_README.md) - Detailed statistical analysis
- [README.md](README.md) - Dataset overview
- [../../rl/rl_cl/README_v9_ADVANCED.md](../../rl/rl_cl/README_v9_ADVANCED.md) - Training guide
- [../../rl/rl_cl/USAGE_EXAMPLES.md](../../rl/rl_cl/USAGE_EXAMPLES.md) - Training examples

## References

### Data Sources
- **PEER NGA-West2**: https://ngawest2.berkeley.edu/
- Ground motion database for real earthquake recordings

### Generation Scripts
- `generate_training_earthquakes_v2.py` - Creates synthetic training variants
- `fix_baseline_drift_v2.py` - Applies baseline correction and tapering

### Analysis Scripts
- `plot_train_vs_test.py` - Train/test comparison (this script)
- `plot_all_datasets.py` - Baseline verification

## Version History

### V2 (Current - January 2026)
- ✅ Proper baseline correction (zero final acceleration)
- ✅ Smooth cosine tapering (last 5 seconds)
- ✅ Comprehensive train/test analysis
- ✅ Frequency domain analysis (PSD)
- ✅ Energy content analysis (Arias Intensity)

### V1 (Deprecated - December 2025)
- ❌ Baseline drift issues
- ❌ Non-zero final acceleration
- ❌ Abrupt endings (no tapering)

## Best Practices

### ✅ DO:
1. Run analysis BEFORE starting RL training
2. Verify train/test split is proper
3. Check baseline correction is effective
4. Review all visualizations
5. Use test set ONLY for final evaluation
6. Report both training and test performance

### ❌ DON'T:
1. Train on test earthquakes (contamination)
2. Tune hyperparameters on test set (overfitting)
3. Mix training and test data
4. Ignore RMS differences (important for understanding difficulty)
5. Skip verification step (may train on corrupted data)

## Citation

If using these datasets or analysis tools, please acknowledge:

```
Earthquake Dataset Analysis Suite
Author: Siddharth
Date: January 2026
Repository: struct-engineer-ai
Data Source: PEER NGA-West2 Ground Motion Database
```

---

**Author**: Siddharth
**Date**: January 2026
**Version**: 2.0
**Status**: Production Ready
