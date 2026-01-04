# V9 Advanced PPO Training - Complete Guide

## Overview

The v9 Advanced PPO training system is a production-ready earthquake control training pipeline with comprehensive features for robust, resumable, and monitorable training.

## Key Features

### 1. Advanced PPO Hyperparameters
- **Deeper Network**: 3-layer architecture [256, 256, 256]
- **Tanh Activation**: Bounded actions for earthquake control
- **Larger Rollouts**: 2048-8192 n_steps (2× increase from v8)
- **Balanced Training**: Optimized batch_size and n_epochs per stage
- **Cosine LR Annealing**: Smooth learning rate decay
- **Entropy Scheduling**: Averaged coefficients for stable exploration

### 2. Automatic Checkpoint Recovery
- **Auto-Resume**: Automatically detects and resumes from last completed stage
- **Emergency Saves**: Saves checkpoints on Ctrl+C or crashes
- **Smart Recovery**: Only uses successfully completed checkpoints
- **Zero Data Loss**: Continue training seamlessly after interruptions

### 3. TensorBoard Integration
- **Comprehensive Logging**: All PPO metrics + custom metrics
- **Real-Time Monitoring**: Track training progress live
- **Multi-Run Comparison**: Compare different experiments easily
- **Stage Tracking**: Logs stage number, LR, entropy coefficient

### 4. Flexible Experiment Organization
- **Command-Line Arguments**: `--model-dir`, `--log-dir`, `--run-name`
- **Run Names**: Organize multiple experiments under same parent directory
- **Custom Directories**: Save to any location (local, cloud, mounted drives)

## Quick Start

### Default Training
```bash
cd /Users/Shared/dev/git/struct-engineer-ai/rl/rl_cl
python train_v9_advanced_ppo.py
```

Output:
- Models: `models/rl_v9_advanced/`
- Logs: `logs/rl_v9_advanced/`

### Named Experiment
```bash
python train_v9_advanced_ppo.py --run-name baseline
```

Output:
- Models: `models/rl_v9_advanced/baseline/`
- Logs: `logs/rl_v9_advanced/baseline/`

### Custom Directories
```bash
python train_v9_advanced_ppo.py \
    --model-dir models/my_experiment \
    --log-dir logs/my_experiment \
    --run-name trial_001
```

Output:
- Models: `models/my_experiment/trial_001/`
- Logs: `logs/my_experiment/trial_001/`

## Viewing Training Progress

### Launch TensorBoard
In a separate terminal:
```bash
tensorboard --logdir=logs/rl_v9_advanced
```

Then open: http://localhost:6006

### Compare Multiple Runs
```bash
# Train different experiments
python train_v9_advanced_ppo.py --run-name baseline
python train_v9_advanced_ppo.py --run-name high_lr
python train_v9_advanced_ppo.py --run-name deep_net

# View all in TensorBoard
tensorboard --logdir=logs/rl_v9_advanced
```

TensorBoard will show all runs side-by-side for comparison!

## Resuming Training

### After Interruption (Ctrl+C)
Simply run the same command again:
```bash
python train_v9_advanced_ppo.py --run-name my_exp
```

The script will:
1. Detect the last completed stage
2. Load the checkpoint
3. Resume from the next stage

**IMPORTANT**: Always use the SAME arguments when resuming!

### After Crash
Same as above - the system automatically saves emergency checkpoints and resumes from the last successfully completed stage.

## Training Curriculum

### Stage 1: M4.5 @ 50kN
- **Timesteps**: 300,000
- **n_steps**: 2,048
- **Learning Rate**: 3e-4 (fixed)
- **Entropy**: 0.025 (fixed)
- **Goal**: Learn basic control with high exploration

### Stage 2: M5.7 @ 100kN
- **Timesteps**: 300,000
- **n_steps**: 4,096
- **Learning Rate**: 3e-4 → 2e-4 (cosine)
- **Entropy**: 0.015 (fixed)
- **Goal**: Longer rollouts with smooth LR decay

### Stage 3: M7.4 @ 150kN
- **Timesteps**: 400,000
- **n_steps**: 8,192
- **Learning Rate**: 2e-4 → 1e-4 (cosine)
- **Entropy**: 0.008 → 0.003 (averaged)
- **Goal**: Large rollouts, careful learning

### Stage 4: M8.4 @ 150kN
- **Timesteps**: 400,000
- **n_steps**: 8,192
- **Learning Rate**: 1e-4 → 5e-5 (cosine)
- **Entropy**: 0.005 → 0.001 (averaged)
- **Goal**: Maximum stability, ultra-careful learning

**Total**: 1,400,000 timesteps × 4 envs = 5,600,000 effective samples

## TensorBoard Metrics

### Automatic Metrics (from Stable-Baselines3)
- `rollout/ep_rew_mean` - Average episode reward
- `rollout/ep_len_mean` - Average episode length
- `train/loss` - Total loss
- `train/policy_gradient_loss` - Policy loss
- `train/value_loss` - Value function loss
- `train/entropy_loss` - Entropy loss
- `train/approx_kl` - KL divergence
- `train/clip_fraction` - Clipping fraction
- `train/explained_variance` - Value function accuracy
- `time/fps` - Training speed

### Custom Metrics (from TensorboardCallback)
- `train/learning_rate` - Current learning rate
- `train/entropy_coef` - Current entropy coefficient
- `stage/stage_number` - Current training stage (1-4)

## File Structure

```
models/rl_v9_advanced/
├── stage1_50kN.zip          # Stage 1 checkpoint
├── stage2_100kN.zip         # Stage 2 checkpoint
├── stage3_150kN.zip         # Stage 3 checkpoint
├── stage4_150kN.zip         # Stage 4 checkpoint
└── final_v9_advanced.zip    # Final trained model

logs/rl_v9_advanced/
├── v9_advanced_stage1_1/    # Stage 1 TensorBoard logs
├── v9_advanced_stage2_2/    # Stage 2 TensorBoard logs
├── v9_advanced_stage3_3/    # Stage 3 TensorBoard logs
└── v9_advanced_stage4_4/    # Stage 4 TensorBoard logs
```

## Configuration Files

- **[train_v9_advanced_ppo.py](train_v9_advanced_ppo.py)** - Main training script
- **[ppo_config_v9_advanced.py](ppo_config_v9_advanced.py)** - Hyperparameter configuration
- **[USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)** - Command-line usage examples
- **[TENSORBOARD_GUIDE.md](TENSORBOARD_GUIDE.md)** - TensorBoard integration guide
- **[CHECKPOINT_RECOVERY.md](CHECKPOINT_RECOVERY.md)** - Checkpoint recovery documentation

## Improvements Over v8

### 1. Network Architecture
- **v8**: 2 layers [256, 256]
- **v9**: 3 layers [256, 256, 256] with Tanh activation

### 2. Rollout Buffer (n_steps)
- **M4.5**: 2048 (was 1024) - 2× larger
- **M5.7**: 4096 (was 2048) - 2× larger
- **M7.4**: 8192 (was 4096) - 2× larger
- **M8.4**: 8192 (was 4096) - 2× larger

### 3. Batch Size & Epochs
- **M5.7**: batch=512, epochs=12 (was 256/10)
- **M7.4**: batch=512, epochs=15 (was 256/10)
- **M8.4**: batch=512, epochs=20 (was 256/10)

### 4. Learning Rate Schedule
- Smoother transitions between stages
- Cosine annealing within stages
- 3e-4 → 2e-4 → 1e-4 → 5e-5 progression

### 5. Value Function Clipping
- Tighter: 0.15 (was 0.2) for more stable value learning

## Best Practices

### 1. Use Run Names for Organization
```bash
# Good - organized
python train_v9_advanced_ppo.py --run-name baseline
python train_v9_advanced_ppo.py --run-name lr_tuning
python train_v9_advanced_ppo.py --run-name final_model
```

### 2. Descriptive Run Names
```bash
# Good - descriptive
python train_v9_advanced_ppo.py --run-name "v9_tanh_lr3e4_batch512"

# Less helpful
python train_v9_advanced_ppo.py --run-name "test1"
```

### 3. Monitor Training in Real-Time
```bash
# Terminal 1: Training
python train_v9_advanced_ppo.py --run-name my_exp

# Terminal 2: TensorBoard
tensorboard --logdir=logs/rl_v9_advanced
```

### 4. Document Your Experiments
Create notes for each experiment:
```bash
cat > models/rl_v9_advanced/my_exp/README.md <<EOF
# Experiment: My Experiment

**Date:** 2026-01-04
**Goal:** Test deeper network architecture
**Config:** v9 advanced with 3-layer network
**Results**: [To be filled after training]
EOF
```

## Troubleshooting

### Checkpoint not found when resuming
**Problem**: Using different arguments than initial run

**Solution**: Use SAME arguments
```bash
# Initial run
python train_v9_advanced_ppo.py --run-name exp1

# Resume (correct)
python train_v9_advanced_ppo.py --run-name exp1
```

### TensorBoard shows no data
**Problem**: Wrong directory or training hasn't started

**Solution**: Check directory and refresh
```bash
ls -la logs/rl_v9_advanced/
tensorboard --logdir=logs/rl_v9_advanced
# Refresh browser (Ctrl+R)
```

### Out of disk space
**Problem**: Checkpoints and logs taking too much space

**Solution**: Clean up old experiments
```bash
# Remove old experiment
rm -rf models/rl_v9_advanced/old_exp
rm -rf logs/rl_v9_advanced/old_exp

# Or compress
tar -czf old_exp.tar.gz models/rl_v9_advanced/old_exp logs/rl_v9_advanced/old_exp
rm -rf models/rl_v9_advanced/old_exp logs/rl_v9_advanced/old_exp
```

## Common Workflows

### Hyperparameter Sweep
```bash
# Test different learning rates
python train_v9_advanced_ppo.py --run-name lr_3e4
python train_v9_advanced_ppo.py --run-name lr_1e4
python train_v9_advanced_ppo.py --run-name lr_5e5

# Compare in TensorBoard
tensorboard --logdir=logs/rl_v9_advanced
```

### Cloud GPU Training
```bash
# Google Colab / Kaggle
python train_v9_advanced_ppo.py \
    --model-dir /content/drive/MyDrive/models/v9 \
    --log-dir /content/drive/MyDrive/logs/v9 \
    --run-name colab_run1

# AWS / Azure / GCP
python train_v9_advanced_ppo.py \
    --model-dir /mnt/data/models/v9 \
    --log-dir /mnt/data/logs/v9 \
    --run-name gpu_t4
```

### Fresh Start
```bash
# Delete all checkpoints
rm -rf models/rl_v9_advanced/
python train_v9_advanced_ppo.py
```

## Command Reference

| Use Case | Command |
|----------|---------|
| Default training | `python train_v9_advanced_ppo.py` |
| Named experiment | `python train_v9_advanced_ppo.py --run-name my_exp` |
| Custom directories | `python train_v9_advanced_ppo.py --model-dir models/exp --log-dir logs/exp` |
| Resume training | Use same arguments as initial run |
| View logs | `tensorboard --logdir=logs/rl_v9_advanced` |
| View specific run | `tensorboard --logdir=logs/rl_v9_advanced/my_exp` |
| Compare runs | `tensorboard --logdir=logs/rl_v9_advanced` |
| Help | `python train_v9_advanced_ppo.py --help` |

## Expected Training Time

On GPU (NVIDIA Tesla T4):
- Stage 1 (M4.5): ~15-20 minutes
- Stage 2 (M5.7): ~15-20 minutes
- Stage 3 (M7.4): ~20-30 minutes
- Stage 4 (M8.4): ~20-30 minutes
- **Total**: ~70-100 minutes

On CPU:
- **Total**: ~6-10 hours (significantly slower)

## Next Steps After Training

1. **Evaluate on test set**:
   ```bash
   python evaluate_v9_advanced.py
   ```

2. **Compare with baselines**:
   - v8 PPO
   - v7 SAC
   - Uncontrolled structure

3. **Deploy to production**:
   - Export model
   - Integrate with control API
   - Monitor real-time performance

## Support

For questions or issues:
1. Check [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) for detailed examples
2. Check [TENSORBOARD_GUIDE.md](TENSORBOARD_GUIDE.md) for monitoring help
3. Check [CHECKPOINT_RECOVERY.md](CHECKPOINT_RECOVERY.md) for resume issues
4. Review training logs and TensorBoard metrics

## Version History

- **v9-advanced**: Advanced PPO with deeper network, larger rollouts, cosine LR annealing
- **v8**: PPO with curriculum learning and optimized hyperparameters
- **v7**: SAC with adaptive entropy tuning
- **v6**: Basic PPO implementation
- **v5**: DQN baseline

---

**Author**: Siddharth
**Date**: January 2026
**Model**: PPO (Proximal Policy Optimization)
**Framework**: Stable-Baselines3
**Application**: Earthquake control with TMD (Tuned Mass Damper)
