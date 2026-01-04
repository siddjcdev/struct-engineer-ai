# TensorBoard Integration Guide - v9 Advanced PPO Training

## Overview

The v9 advanced PPO training now includes **comprehensive TensorBoard logging** to track training metrics, visualize learning curves, and monitor model performance in real-time.

## What Gets Logged

### Automatic Metrics (from Stable-Baselines3)

1. **Rollout Metrics** (logged every n_steps):
   - `rollout/ep_len_mean` - Average episode length
   - `rollout/ep_rew_mean` - Average episode reward

2. **Training Metrics** (logged every update):
   - `train/approx_kl` - Approximate KL divergence
   - `train/clip_fraction` - Fraction of samples clipped
   - `train/clip_range` - Current clip range
   - `train/entropy_loss` - Entropy loss
   - `train/explained_variance` - Explained variance
   - `train/loss` - Total loss
   - `train/policy_gradient_loss` - Policy gradient loss
   - `train/value_loss` - Value function loss
   - `train/n_updates` - Number of parameter updates

3. **Time Metrics**:
   - `time/fps` - Frames per second
   - `time/iterations` - Training iterations
   - `time/time_elapsed` - Total time elapsed
   - `time/total_timesteps` - Total timesteps processed

### Custom Metrics (from TensorboardCallback)

4. **Learning Rate Tracking**:
   - `train/learning_rate` - Current learning rate (from cosine schedule)

5. **Entropy Coefficient Tracking**:
   - `train/entropy_coef` - Current entropy coefficient

6. **Stage Information**:
   - `stage/stage_number` - Current training stage (1-4)

## Usage

### Start Training with TensorBoard Logging

```bash
cd /Users/Shared/dev/git/struct-engineer-ai/rl/rl_cl
python train_v9_advanced_ppo.py
```

The training script automatically:
- Creates log directory: `logs/rl_v9_advanced/`
- Logs metrics during training
- Saves separate logs for each stage

### View Training Metrics in Real-Time

In a **separate terminal** (while training is running or after):

```bash
cd /Users/Shared/dev/git/struct-engineer-ai/rl/rl_cl
tensorboard --logdir=logs/rl_v9_advanced
```

Then open your browser to: **http://localhost:6006**

### View on Different Port

If port 6006 is already in use:

```bash
tensorboard --logdir=logs/rl_v9_advanced --port=6007
```

### View from Remote Server

If training on a remote GPU server:

```bash
# On remote server
tensorboard --logdir=logs/rl_v9_advanced --host=0.0.0.0 --port=6006

# On local machine, create SSH tunnel
ssh -L 6006:localhost:6006 user@remote-server
```

Then access at: **http://localhost:6006**

## TensorBoard Dashboard Tabs

### 1. SCALARS Tab
**Most important tab** - Shows all numeric metrics over time.

Key metrics to monitor:

**Episode Performance:**
- `rollout/ep_rew_mean` - Is reward improving? Should increase over time
- `rollout/ep_len_mean` - Episode lengths

**Learning Stability:**
- `train/approx_kl` - Should stay small (<0.1), large values indicate unstable updates
- `train/clip_fraction` - Around 10-30% is good, too high means aggressive updates
- `train/explained_variance` - Should be high (>0.5), indicates value function is learning

**Loss Trends:**
- `train/loss` - Total loss (should decrease)
- `train/policy_gradient_loss` - Policy loss
- `train/value_loss` - Value function loss

**Learning Rate Schedule:**
- `train/learning_rate` - Should show cosine decay pattern

**Stage Progress:**
- `stage/stage_number` - Shows which stage is currently training

### 2. GRAPHS Tab
Shows the computational graph of your neural network.

### 3. DISTRIBUTIONS Tab
Shows weight and bias distributions across layers (if enabled).

### 4. HISTOGRAMS Tab
Shows histograms of weights, gradients, and activations over time.

## Comparing Multiple Runs

TensorBoard can compare different training runs:

```bash
# Compare v8 and v9 training
tensorboard --logdir_spec=v8:logs/rl_v8_ppo_optimized,v9:logs/rl_v9_advanced
```

This allows you to:
- Compare different hyperparameter settings
- Track improvements across versions
- Identify which configuration works best

## Analyzing Results

### Healthy Training Indicators

✅ **Good Signs:**
- `rollout/ep_rew_mean` steadily increasing
- `train/approx_kl` staying below 0.1
- `train/clip_fraction` between 0.1-0.3
- `train/explained_variance` > 0.5
- `train/value_loss` decreasing
- Learning rate following smooth cosine decay

⚠️ **Warning Signs:**
- `rollout/ep_rew_mean` oscillating wildly → Reduce learning rate
- `train/approx_kl` > 0.1 → Policy updates too aggressive, reduce LR
- `train/clip_fraction` > 0.5 → Too much clipping, possibly hitting local minima
- `train/explained_variance` < 0 → Value function not learning, check reward scaling

### Stage-by-Stage Analysis

The training proceeds through 4 stages:
1. **Stage 1 (M4.5)**: Fast learning, rewards should improve quickly
2. **Stage 2 (M5.7)**: Moderate difficulty, steady improvement
3. **Stage 3 (M7.4)**: Challenging, slower learning expected
4. **Stage 4 (M8.4)**: Extreme difficulty, may plateau

Monitor `stage/stage_number` to see transitions between stages.

## Log File Structure

```
logs/rl_v9_advanced/
├── v9_advanced_stage1_1/        # Stage 1 logs (first run)
│   ├── events.out.tfevents.*
│   └── ...
├── v9_advanced_stage2_2/        # Stage 2 logs (first run)
│   ├── events.out.tfevents.*
│   └── ...
├── v9_advanced_stage3_3/        # Stage 3 logs (first run)
│   ├── events.out.tfevents.*
│   └── ...
├── v9_advanced_stage4_4/        # Stage 4 logs (first run)
│   ├── events.out.tfevents.*
│   └── ...
```

Each stage gets a separate subdirectory with its own event files.

## Exporting Data

### Export as CSV

```python
from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator('logs/rl_v9_advanced/v9_advanced_stage1_1')
ea.Reload()

# Get all scalar tags
print(ea.Tags())

# Export a specific metric
import pandas as pd
rewards = pd.DataFrame(ea.Scalars('rollout/ep_rew_mean'))
rewards.to_csv('rewards.csv', index=False)
```

### Save as Images

In TensorBoard web interface:
1. Click the download icon on any plot
2. Saves as PNG/SVG

## Troubleshooting

### TensorBoard not showing data

**Problem**: TensorBoard starts but shows "No data found"

**Solutions:**
1. Check if training has started writing logs:
   ```bash
   ls -la logs/rl_v9_advanced/
   ```

2. Make sure path is correct:
   ```bash
   tensorboard --logdir=logs/rl_v9_advanced  # Relative path
   # OR
   tensorboard --logdir=/full/path/to/logs/rl_v9_advanced  # Absolute path
   ```

3. Try refreshing browser or clearing cache

### Port already in use

**Problem**: `ERROR: TensorBoard could not bind to port 6006`

**Solution:**
```bash
# Use different port
tensorboard --logdir=logs/rl_v9_advanced --port=6007

# Or kill existing TensorBoard process
pkill -f tensorboard
```

### Logs taking too much space

**Problem**: Log files growing very large

**Solution:**
```bash
# Remove old logs (keep only latest run)
rm -rf logs/rl_v9_advanced/v9_advanced_stage*_[0-9]/

# Or compress old logs
tar -czf old_logs_backup.tar.gz logs/rl_v9_advanced/
rm -rf logs/rl_v9_advanced/
```

### Cannot connect from browser

**Problem**: TensorBoard running but browser can't connect

**Solutions:**
1. Check firewall settings
2. Try explicit host:
   ```bash
   tensorboard --logdir=logs/rl_v9_advanced --host=127.0.0.1
   ```
3. Use SSH tunnel if on remote server

## Advanced: Custom Metrics

To log additional custom metrics (e.g., peak displacement per stage):

```python
# In the TensorboardCallback class
def _on_rollout_end(self) -> None:
    # ... existing code ...

    # Add custom metrics
    self.logger.record("custom/peak_displacement", peak_disp_cm)
    self.logger.record("custom/dcr", dcr_value)
```

## Useful TensorBoard Options

```bash
# Reduce memory usage
tensorboard --logdir=logs/rl_v9_advanced --max_reload_threads=1 --reload_interval=30

# Load only specific runs
tensorboard --logdir=logs/rl_v9_advanced --path_prefix=v9_advanced_stage3

# Set update frequency
tensorboard --logdir=logs/rl_v9_advanced --reload_interval=10  # seconds

# Disable warnings
tensorboard --logdir=logs/rl_v9_advanced --reload_multifile=true
```

## Example Analysis Workflow

1. **Start training**:
   ```bash
   python train_v9_advanced_ppo.py
   ```

2. **Launch TensorBoard** (in separate terminal):
   ```bash
   tensorboard --logdir=logs/rl_v9_advanced
   ```

3. **Monitor during training**:
   - Check `rollout/ep_rew_mean` every 10-15 minutes
   - Verify `train/approx_kl` stays below 0.1
   - Watch for stage transitions in `stage/stage_number`

4. **After training**:
   - Compare all 4 stages
   - Export reward curves as images
   - Analyze learning rate decay pattern
   - Check for overfitting signs

## Resources

- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)
- [Stable-Baselines3 Logging Guide](https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html)
- [TensorBoard.dev](https://tensorboard.dev/) - Share results publicly

## Summary

TensorBoard provides powerful visualization for:
- ✅ Real-time training monitoring
- ✅ Learning curve analysis
- ✅ Hyperparameter comparison
- ✅ Debugging training issues
- ✅ Publishing results

The v9 advanced training automatically logs all essential metrics, making it easy to track and optimize your earthquake control model!
