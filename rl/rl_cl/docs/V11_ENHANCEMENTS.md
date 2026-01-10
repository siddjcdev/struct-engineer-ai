# V11 Advanced PPO - File Logging and DCR/ISDR Metrics
## What's New in V11

V11 builds on V9 by adding:
1. **Comprehensive file logging**: All console output saved to timestamped log files
2. **DCR and ISDR metrics in TensorBoard**: Track structural safety metrics during training
3. **Enhanced TensorboardCallback**: Automatically extracts and logs episode-level metrics

## Quick Start

### Run Training
```bash
cd c:\Dev\dAmpIng26\git\struct-engineer-ai\rl\rl_cl
python train_v11.py
```

The script will:
- Save model checkpoints to: `models/rl_v11_advanced/`
- Save TensorBoard logs to: `logs/rl_v11_advanced/`
- Save console output to: `logs/rl_v11_advanced/training_YYYYMMDD_HHMMSS.log`

### Launch TensorBoard (in separate terminal)
```bash
cd c:\Dev\dAmpIng26\git\struct-engineer-ai\rl\rl_cl
tensorboard --logdir=logs/rl_v11_advanced
```

Open browser: [http://localhost:6006](http://localhost:6006)

## New TensorBoard Metrics

In addition to all V9 metrics, V11 now logs:

### Structural Safety Metrics
- `metrics/avg_isdr_percent` - Average ISDR across recent episodes
- `metrics/max_isdr_percent` - Maximum ISDR across recent episodes
- `metrics/avg_dcr` - Average DCR across recent episodes
- `metrics/max_dcr` - Maximum DCR across recent episodes
- `metrics/avg_peak_displacement_cm` - Average peak displacement (cm)
- `metrics/max_peak_displacement_cm` - Maximum peak displacement (cm)

### How It Works
The enhanced `TensorboardCallback`:
1. Extracts episode metrics from environment info dicts
2. Accumulates metrics during each rollout
3. Logs averages and max values at rollout end
4. Resets metric buffers for next rollout

## File Logging

### TeeLogger Class
V11 implements a `TeeLogger` that writes to both:
- Console (stdout) - you see real-time output including progress bars
- Log file - permanent record saved to disk (progress bars filtered out for clean logs)

**Progress Bar Support:**
- TeeLogger is fully compatible with tqdm progress bars
- Progress bars display normally in the console
- Log files only contain completed lines (no progress bar clutter)
- Implements `encoding` property and `isatty()` method for tqdm compatibility

### Log File Location
```
logs/rl_v11_advanced/training_YYYYMMDD_HHMMSS.log
```

Example: `training_20260107_143052.log`

### Log File Contents
Everything you see in the console is also saved to the log file:
- Training configuration
- Stage progress
- Episode statistics
- Checkpoint saves
- Error messages
- Final results

### Benefits
1. **Reproducibility**: Exact record of what happened during training
2. **Debugging**: Can review logs after training completes
3. **Sharing**: Easy to share training logs with collaborators
4. **Analysis**: Can parse log files for automated analysis

## Environment Changes

Updated `tmd_environment_adaptive_reward.py` to include episode-level metrics in info dict when episodes end:
- `max_isdr_percent`: Maximum ISDR for the episode
- `max_dcr`: DCR for the episode
- `peak_displacement`: Peak roof displacement for the episode

This enables the TensorboardCallback to extract and log these metrics.

## Comparison: V9 vs V11

| Feature | V9 | V11 |
|---------|----|----|
| PPO hyperparameters | ✓ Advanced | ✓ Same as V9 |
| Curriculum learning | ✓ 4 stages | ✓ Same as V9 |
| TensorBoard basics | ✓ Built-in | ✓ Same as V9 |
| File logging | ✗ Console only | ✓ Dual output |
| DCR metrics | ✗ Not logged | ✓ Tracked |
| ISDR metrics | ✗ Not logged | ✓ Tracked |
| Displacement tracking | ✗ Basic | ✓ Enhanced |

## Monitoring Training Health

### Target Metrics (M4.5 earthquake)
- Peak displacement: **14 cm**
- Max ISDR: **0.4%**
- DCR: **1.0-1.1**

### Watch These Metrics in TensorBoard
1. **metrics/max_isdr_percent**: Should decrease toward 0.4% for M4.5
2. **metrics/max_dcr**: Should stay near 1.0-1.1 (avoid weak story)
3. **metrics/avg_peak_displacement_cm**: Should decrease toward 14 cm
4. **rollout/ep_rew_mean**: Should increase (less negative) over time
5. **train/value_loss**: Should stay stable (< 100)
6. **train/explained_variance**: Should stay > 0.7

### Good Training Signs ✅
1. ISDR decreasing or staying below 1.5%
2. DCR staying below 1.75 (safety limit)
3. Displacement improving compared to baseline
4. Value function stable (explained_variance > 0.7)
5. Rewards improving steadily

### Warning Signs ⚠️
1. ISDR increasing above 2.0%
2. DCR spiking above 2.0
3. Displacement not improving
4. Value function unstable (explained_variance < 0.3)
5. Rewards not improving after 50k steps

## Command-Line Options

### All Available Arguments

```bash
python train_v11.py [OPTIONS]

Options:
  --train-dir DIR      Directory containing training dataset variants
                       (default: ../../matlab/datasets/training_set_v2)

  --test-dir DIR       Directory containing test earthquake files
                       (default: ../../matlab/datasets)

  --model-dir DIR      Directory to save model checkpoints
                       (default: models/rl_v11_advanced)

  --log-dir DIR        Directory for TensorBoard logs
                       (default: logs/rl_v11_advanced)

  --run-name NAME      Name for this training run (auto-generated if not specified)
                       Creates subdirectories under model-dir and log-dir
```

### Examples

#### Custom Run Name
```bash
python train_v11.py --run-name gentle_constraints_v1
```

Saves to:
- Models: `models/rl_v11_advanced/gentle_constraints_v1/`
- Logs: `logs/rl_v11_advanced/gentle_constraints_v1/`
- Log file: `logs/rl_v11_advanced/gentle_constraints_v1/training_YYYYMMDD_HHMMSS.log`

#### Custom Dataset Directories
```bash
python train_v11.py \
  --train-dir /path/to/training/data \
  --test-dir /path/to/test/data
```

#### Custom Output Directories
```bash
python train_v11.py \
  --model-dir experiments/models \
  --log-dir experiments/logs \
  --run-name test_run_1
```

#### Complete Custom Configuration
```bash
python train_v11.py \
  --train-dir ../../matlab/datasets/training_set_v3 \
  --test-dir ../../matlab/datasets/test \
  --model-dir my_models \
  --log-dir my_logs \
  --run-name final_experiment
```

## Resume Training

V11 automatically resumes from the last checkpoint:
```bash
# Start training
python train_v11.py

# Press Ctrl+C to interrupt
# Script saves emergency checkpoint

# Run again to resume
python train_v11.py
# Automatically continues from last stage!
```

## Troubleshooting

### No DCR/ISDR metrics in TensorBoard
- Wait a few episodes - metrics only appear when episodes complete
- Check that environment is `tmd_environment_adaptive_reward.py`
- Verify `get_episode_metrics()` is working in environment

### Log file not created
- Check that log directory exists: `logs/rl_v11_advanced/`
- Verify write permissions on directory
- Look for error messages in console

### Metrics look wrong
- Check reward function is working correctly
- Run `test_fixed_reward.py` to verify environment
- Review episode-level info dicts in debugger

## Files Modified

1. [train_v11.py](train_v11.py) - New training script with file logging
2. [tmd_environment_adaptive_reward.py](../../restapi/rl_cl/tmd_environment_adaptive_reward.py) - Added episode metrics to info dict
3. This document - Usage guide

## Why Use V11?

**Use V11 if you want:**
- ✓ Permanent record of training runs
- ✓ Detailed structural safety tracking
- ✓ Easy comparison of ISDR/DCR across experiments
- ✓ All V9 optimizations preserved

**Stay with V9 if:**
- You don't need file logging
- You don't need DCR/ISDR tracking in TensorBoard
- You prefer minimal code changes

## Example Training Session

1. **Start Training**:
   ```bash
   python train_v11.py --run-name gentle_constraints_test
   ```

2. **In another terminal, start TensorBoard**:
   ```bash
   tensorboard --logdir=logs/rl_v11_advanced
   ```

3. **Open browser**: http://localhost:6006

4. **Monitor metrics**:
   - Check `metrics/max_isdr_percent` trends downward
   - Check `metrics/max_dcr` stays below 1.75
   - Watch `metrics/avg_peak_displacement_cm` improve
   - Verify `train/explained_variance` > 0.7

5. **After training**:
   - Review log file: `logs/rl_v11_advanced/gentle_constraints_test/training_*.log`
   - Check final model: `models/rl_v11_advanced/gentle_constraints_test/stage4_*.zip`
   - Export TensorBoard data for analysis

## Next Steps

Ready to train with gentle ISDR/DCR constraints! The enhanced logging and metrics will help you verify that:
- Displacement targets are being met (14 cm for M4.5)
- Structural safety constraints are working (ISDR < 1.5%, DCR < 1.75)
- Training is stable and improving over time
