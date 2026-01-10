# V9 Advanced PPO - TensorBoard Guide
## How to Use TensorBoard with train_v9_advanced_ppo.py

V9 training script already has **full TensorBoard integration built-in**. No modifications needed!

## Quick Start

### 1. Start Training
```bash
cd c:\Dev\dAmpIng26\git\struct-engineer-ai\rl\rl_cl
python train_v9_advanced_ppo.py
```

The script will automatically:
- Save model checkpoints to: `models/rl_v9_advanced/`
- Save TensorBoard logs to: `logs/rl_v9_advanced/`

### 2. Launch TensorBoard (in a separate terminal)
```bash
cd c:\Dev\dAmpIng26\git\struct-engineer-ai\rl\rl_cl
tensorboard --logdir=logs/rl_v9_advanced
```

### 3. Open TensorBoard in Your Browser
Navigate to: [http://localhost:6006](http://localhost:6006)

## What V9 Logs to TensorBoard

### Built-in Metrics (from Stable-Baselines3)
- `rollout/ep_len_mean` - Average episode length
- `rollout/ep_rew_mean` - Average episode reward
- `time/fps` - Frames per second (training speed)
- `time/iterations` - Number of update iterations
- `time/time_elapsed` - Total training time
- `time/total_timesteps` - Total environment steps
- `train/approx_kl` - Approximate KL divergence
- `train/clip_fraction` - Fraction of clipped policy updates
- `train/clip_range` - PPO clip range (epsilon)
- `train/entropy_loss` - Entropy loss (exploration)
- `train/explained_variance` - How well value function predicts returns (should be > 0.7)
- `train/loss` - Combined loss
- `train/policy_gradient_loss` - Policy loss
- `train/value_loss` - Value function loss (should be stable, not millions!)

### Custom Metrics (from TensorboardCallback)
- `train/learning_rate` - Current learning rate (from schedule)
- `train/entropy_coef` - Current entropy coefficient
- `stage/stage_number` - Current curriculum stage (1-4)

## Monitoring Training Health

### Good Training Signs ✅
1. **explained_variance > 0.7** - Value function is learning
2. **value_loss < 100** - Value estimates are stable
3. **ep_rew_mean improving** - Agent is getting better rewards
4. **approx_kl < 0.05** - Policy updates are stable
5. **entropy_loss decreasing slowly** - Policy becoming more confident

### Warning Signs ⚠️
1. **explained_variance < 0.3** - Value function struggling
2. **value_loss > 1000** - Value estimates exploding
3. **ep_rew_mean not improving** - Agent not learning
4. **approx_kl > 0.1** - Policy changing too fast
5. **clip_fraction near 0 or 1** - Need to adjust clip_range

### Catastrophic Signs ❌ (like we had before)
1. **explained_variance < 0.05** - Value function completely broken
2. **value_loss > 1,000,000** - Reward magnitudes too large
3. **ep_rew_mean getting worse** - Agent actively making things worse
4. **Training extremely slow** - Gradient issues

## Advanced Usage

### Custom Run Name (organize experiments)
```bash
python train_v9_advanced_ppo.py --run-name experiment1
```

Logs will be saved to: `logs/rl_v9_advanced/experiment1/`

### Compare Multiple Runs
```bash
tensorboard --logdir=logs/rl_v9_advanced
```

TensorBoard will automatically show all runs in the directory, allowing you to compare them.

### Resume Training
V9 automatically resumes from the last checkpoint if you interrupt and restart:
```bash
# Start training
python train_v9_advanced_ppo.py

# Press Ctrl+C to interrupt
# Script saves emergency checkpoint

# Run again to resume
python train_v9_advanced_ppo.py
# Automatically continues from last stage!
```

## TensorBoard Tips

### Smoothing
- Adjust the "Smoothing" slider (default 0.6) to see trends more clearly
- Lower smoothing (0.0) shows raw data
- Higher smoothing (0.9) shows overall trend

### Scalar Selection
- Click checkboxes to show/hide specific metrics
- Use regex to filter (e.g., `train/.*` shows only training metrics)

### Download Data
- Click "Show data download links" to export CSV
- Use for further analysis or plotting

### Refresh
- TensorBoard auto-refreshes every 30 seconds
- Click refresh icon to force update

## Stage-by-Stage Monitoring

V9 trains in 4 curriculum stages:

| Stage | Magnitude | Force Limit | Timesteps | What to Monitor |
|-------|-----------|-------------|-----------|-----------------|
| 1 | M4.5 | 150 kN | 200k | Smooth learning, value_loss < 50 |
| 2 | M5.7 | 200 kN | 200k | Transfer learning working |
| 3 | M7.4 | 300 kN | 250k | Handling larger earthquakes |
| 4 | M8.4 | 400 kN | 300k | Final performance |

Use `stage/stage_number` to see which stage is currently training.

## Troubleshooting

### "No scalar data found"
- Training just started, wait a few minutes
- Check that training is actually running
- Verify log directory path is correct

### "TensorBoard not found"
```bash
pip install tensorboard
```

### "Port 6006 already in use"
```bash
# Use a different port
tensorboard --logdir=logs/rl_v9_advanced --port=6007
```

### Logs not updating
- TensorBoard caches data, try hard refresh (Ctrl+F5)
- Restart TensorBoard
- Check that training is still running

## Why Use V9 Instead of V10?

**V9 is the "back to basics" stable version:**
- ✅ Proven TensorBoard integration
- ✅ Mature curriculum learning
- ✅ Advanced PPO hyperparameters
- ✅ Automatic checkpoint resumption
- ✅ Well-documented configuration

**V10 with simplified rewards is experimental:**
- ⚠️ New untested reward function
- ⚠️ Less mature codebase
- ⚠️ May need reward tuning

If you want stable training with full observability, **V9 is the better choice**.

## Example TensorBoard Session

1. **Start Training**:
   ```bash
   python train_v9_advanced_ppo.py
   ```

2. **In another terminal, start TensorBoard**:
   ```bash
   tensorboard --logdir=logs/rl_v9_advanced
   ```

3. **Open browser**: http://localhost:6006

4. **Monitor**:
   - Check `train/explained_variance` is > 0.7
   - Check `train/value_loss` is stable (< 100)
   - Watch `rollout/ep_rew_mean` improve over time
   - See `stage/stage_number` advance through curriculum

5. **Training completes**: All checkpoints saved in `models/rl_v9_advanced/`
