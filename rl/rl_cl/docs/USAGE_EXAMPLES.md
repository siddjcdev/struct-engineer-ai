# Usage Examples - v9 Advanced PPO Training

## Command-Line Arguments

The v9 advanced PPO training script now supports command-line arguments for flexible experiment organization.

### Available Arguments

```bash
python train_v9_advanced_ppo.py --help
```

**Arguments:**
- `--model-dir` - Directory to save model checkpoints (default: `models/rl_v9_advanced`)
- `--log-dir` - Directory for TensorBoard logs (default: `logs/rl_v9_advanced`)
- `--run-name` - Run name for organizing experiments (creates subdirectories)

## Basic Usage

### Default Training (No Arguments)

```bash
python train_v9_advanced_ppo.py
```

**Output:**
- Models saved to: `models/rl_v9_advanced/`
- Logs saved to: `logs/rl_v9_advanced/`

### Custom Model Directory

```bash
python train_v9_advanced_ppo.py --model-dir models/my_experiment
```

**Output:**
- Models saved to: `models/my_experiment/`
- Logs saved to: `logs/rl_v9_advanced/` (default)

### Custom Log Directory

```bash
python train_v9_advanced_ppo.py --log-dir logs/my_logs
```

**Output:**
- Models saved to: `models/rl_v9_advanced/` (default)
- Logs saved to: `logs/my_logs/`

### Custom Both Directories

```bash
python train_v9_advanced_ppo.py \
    --model-dir models/experiment_001 \
    --log-dir logs/experiment_001
```

**Output:**
- Models saved to: `models/experiment_001/`
- Logs saved to: `logs/experiment_001/`

## Organized Experiments with Run Names

### Using Run Name (Recommended for Multiple Experiments)

```bash
python train_v9_advanced_ppo.py --run-name exp_baseline
```

**Output:**
- Models saved to: `models/rl_v9_advanced/exp_baseline/`
- Logs saved to: `logs/rl_v9_advanced/exp_baseline/`

**Benefit:** Keeps experiments organized under the same parent directory

### Multiple Experiments Example

```bash
# Baseline experiment
python train_v9_advanced_ppo.py --run-name baseline

# Experiment with different hyperparameters
python train_v9_advanced_ppo.py --run-name high_lr

# Experiment with different network architecture
python train_v9_advanced_ppo.py --run-name deep_network
```

**Directory Structure:**
```
models/rl_v9_advanced/
├── baseline/
│   ├── stage1_50kN.zip
│   ├── stage2_100kN.zip
│   └── ...
├── high_lr/
│   ├── stage1_50kN.zip
│   └── ...
└── deep_network/
    ├── stage1_50kN.zip
    └── ...

logs/rl_v9_advanced/
├── baseline/
│   ├── v9_advanced_stage1_1/
│   └── ...
├── high_lr/
│   ├── v9_advanced_stage1_1/
│   └── ...
└── deep_network/
    ├── v9_advanced_stage1_1/
    └── ...
```

### Compare All Experiments in TensorBoard

```bash
tensorboard --logdir=logs/rl_v9_advanced
```

TensorBoard will show all runs (baseline, high_lr, deep_network) for easy comparison!

## Advanced Use Cases

### Hyperparameter Sweep

```bash
# Sweep learning rates
python train_v9_advanced_ppo.py --run-name lr_3e4
python train_v9_advanced_ppo.py --run-name lr_1e4
python train_v9_advanced_ppo.py --run-name lr_5e5

# View all in TensorBoard
tensorboard --logdir=logs/rl_v9_advanced
```

### Date-Stamped Experiments

```bash
# Add date to run name
python train_v9_advanced_ppo.py --run-name "exp_$(date +%Y%m%d_%H%M%S)"
```

**Output:**
- `models/rl_v9_advanced/exp_20260104_143022/`
- `logs/rl_v9_advanced/exp_20260104_143022/`

### Custom Directory with Run Name

```bash
python train_v9_advanced_ppo.py \
    --model-dir models/ppo_experiments \
    --log-dir logs/ppo_experiments \
    --run-name tanh_activation
```

**Output:**
- Models saved to: `models/ppo_experiments/tanh_activation/`
- Logs saved to: `logs/ppo_experiments/tanh_activation/`

## Resuming from Checkpoint

The checkpoint recovery works automatically with any directory structure:

```bash
# Initial training
python train_v9_advanced_ppo.py --run-name my_exp

# ... training interrupted after Stage 2 ...

# Resume automatically (use SAME arguments!)
python train_v9_advanced_ppo.py --run-name my_exp
```

The script will:
1. Look in `models/rl_v9_advanced/my_exp/` for checkpoints
2. Find `stage2_100kN.zip`
3. Resume from Stage 3

**IMPORTANT:** Always use the same `--model-dir` and `--run-name` when resuming!

## Cloud GPU Training Examples

### Google Colab / Kaggle

```bash
# Save to mounted Google Drive
python train_v9_advanced_ppo.py \
    --model-dir /content/drive/MyDrive/models/v9_advanced \
    --log-dir /content/drive/MyDrive/logs/v9_advanced \
    --run-name colab_run1
```

### AWS / Azure / GCP VM

```bash
# Save to persistent disk
python train_v9_advanced_ppo.py \
    --model-dir /mnt/data/models/v9_advanced \
    --log-dir /mnt/data/logs/v9_advanced \
    --run-name gpu_tesla_t4
```

## Viewing TensorBoard Logs

### View Specific Run

```bash
tensorboard --logdir=logs/rl_v9_advanced/baseline
```

### View All Runs (Compare)

```bash
tensorboard --logdir=logs/rl_v9_advanced
```

### View Custom Log Directory

```bash
tensorboard --logdir=logs/my_experiment
```

### Remote Server Access

```bash
# On remote server
tensorboard --logdir=logs/rl_v9_advanced --host=0.0.0.0

# On local machine, create SSH tunnel
ssh -L 6006:localhost:6006 user@remote-server

# Open http://localhost:6006
```

## Best Practices

### 1. Use Run Names for Organization

```bash
# Good - organized
python train_v9_advanced_ppo.py --run-name baseline
python train_v9_advanced_ppo.py --run-name lr_tuning
python train_v9_advanced_ppo.py --run-name final_model

# Less organized (but works)
python train_v9_advanced_ppo.py --model-dir models/exp1
python train_v9_advanced_ppo.py --model-dir models/exp2
```

### 2. Descriptive Run Names

```bash
# Good - descriptive
python train_v9_advanced_ppo.py --run-name "v9_tanh_lr3e4_batch512"

# Less helpful
python train_v9_advanced_ppo.py --run-name "test1"
```

### 3. Keep Logs and Models Together

```bash
# Good - consistent naming
python train_v9_advanced_ppo.py \
    --model-dir models/experiments \
    --log-dir logs/experiments \
    --run-name baseline

# Confusing - different structures
python train_v9_advanced_ppo.py \
    --model-dir models/baseline \
    --log-dir logs/exp1
```

### 4. Document Your Experiments

Create a README in your experiment directory:

```bash
python train_v9_advanced_ppo.py --run-name lr_sweep_3e4

# Create experiment notes
cat > models/rl_v9_advanced/lr_sweep_3e4/README.md <<EOF
# Experiment: LR Sweep - 3e-4

**Date:** 2026-01-04
**Goal:** Test learning rate 3e-4
**Config:** Default v9 advanced
**Results:** Peak M7.4: 215.3 cm (beats baseline!)
EOF
```

## Quick Reference

| Use Case | Command |
|----------|---------|
| Default training | `python train_v9_advanced_ppo.py` |
| Named experiment | `python train_v9_advanced_ppo.py --run-name my_exp` |
| Custom directories | `python train_v9_advanced_ppo.py --model-dir models/exp --log-dir logs/exp` |
| Resume training | Use same arguments as initial run |
| View logs | `tensorboard --logdir=logs/rl_v9_advanced` |
| View specific run | `tensorboard --logdir=logs/rl_v9_advanced/my_exp` |
| Compare runs | `tensorboard --logdir=logs/rl_v9_advanced` (shows all) |

## Troubleshooting

### Problem: Checkpoint not found when resuming

**Cause:** Using different arguments than initial run

**Solution:**
```bash
# Initial run
python train_v9_advanced_ppo.py --run-name exp1

# Resume (use SAME arguments!)
python train_v9_advanced_ppo.py --run-name exp1  # ✓ Correct
python train_v9_advanced_ppo.py --run-name exp2  # ✗ Wrong - won't find checkpoint
```

### Problem: TensorBoard shows old runs

**Solution:** Use specific run directory or clear cache
```bash
# View only new run
tensorboard --logdir=logs/rl_v9_advanced/my_new_run

# Or clear TensorBoard cache
rm -rf ~/.tensorboard_cache
```

### Problem: Out of disk space

**Solution:** Clean up old experiments
```bash
# Remove old experiment
rm -rf models/rl_v9_advanced/old_exp
rm -rf logs/rl_v9_advanced/old_exp

# Or archive
tar -czf old_exp_backup.tar.gz \
    models/rl_v9_advanced/old_exp \
    logs/rl_v9_advanced/old_exp
rm -rf models/rl_v9_advanced/old_exp
rm -rf logs/rl_v9_advanced/old_exp
```
