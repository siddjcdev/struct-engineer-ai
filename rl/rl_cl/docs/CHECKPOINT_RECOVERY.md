# Checkpoint Recovery System - v9 Advanced PPO Training

## Overview

The v9 advanced PPO training script (`train_v9_advanced_ppo.py`) now includes an **automatic checkpoint recovery system** that allows training to resume from the last completed stage if it fails or is interrupted.

## How It Works

### 1. Checkpoint Detection

When you run the training script, it automatically:
- Scans the `models/rl_v9_advanced/` directory for existing checkpoints
- Identifies the last **successfully completed** stage
- Loads the checkpoint and resumes from the next stage

### 2. Checkpoint Files

The system saves three types of checkpoints:

#### Successfully Completed Stages
```
models/rl_v9_advanced/stage1_50kN.zip
models/rl_v9_advanced/stage2_100kN.zip
models/rl_v9_advanced/stage3_150kN.zip
models/rl_v9_advanced/stage4_150kN.zip
```

#### Emergency Checkpoints (User Interruption)
```
models/rl_v9_advanced/stage3_INTERRUPTED_150kN.zip
```
Saved when you press Ctrl+C to interrupt training.

#### Error Checkpoints (Training Failure)
```
models/rl_v9_advanced/stage3_ERROR_150kN.zip
```
Saved when training crashes due to an exception.

### 3. Resumption Logic

**Scenario 1: Normal Interruption (Ctrl+C)**
```bash
# Stage 1 completes → stage1_50kN.zip saved
# Stage 2 completes → stage2_100kN.zip saved
# Stage 3 running... [User presses Ctrl+C]
# → stage3_INTERRUPTED_150kN.zip saved

# Re-run the script:
python train_v9_advanced_ppo.py

# Output:
# 🔄 RESUMING FROM CHECKPOINT
# Found checkpoint: models/rl_v9_advanced/stage2_100kN.zip
# Last completed: Stage 2/4
# Will resume from Stage 3/4
```

**Scenario 2: Training Error**
```bash
# Stage 1 completes → stage1_50kN.zip saved
# Stage 2 running... [Error occurs]
# → stage2_ERROR_100kN.zip saved

# Re-run the script:
python train_v9_advanced_ppo.py

# Output:
# 🔄 RESUMING FROM CHECKPOINT
# Found checkpoint: models/rl_v9_advanced/stage1_50kN.zip
# Last completed: Stage 1/4
# Will resume from Stage 2/4
```

**Scenario 3: Fresh Start**
```bash
# No checkpoints exist

# Run the script:
python train_v9_advanced_ppo.py

# Output:
# No checkpoints found - starting fresh training
```

## Usage

### Normal Training
```bash
cd /Users/Shared/dev/git/struct-engineer-ai/rl/rl_cl
python train_v9_advanced_ppo.py
```

### Resume After Interruption
Simply run the same command again:
```bash
python train_v9_advanced_ppo.py
```
The script will automatically detect checkpoints and resume.

### Manual Checkpoint Management

#### View Available Checkpoints
```bash
ls -lh models/rl_v9_advanced/
```

#### Start Fresh (Delete All Checkpoints)
```bash
rm -rf models/rl_v9_advanced/
python train_v9_advanced_ppo.py
```

#### Resume from Specific Stage
Delete checkpoints after the desired stage:
```bash
# To resume from Stage 2:
rm models/rl_v9_advanced/stage3_150kN.zip
rm models/rl_v9_advanced/stage4_150kN.zip
rm models/rl_v9_advanced/final_v9_advanced.zip

python train_v9_advanced_ppo.py
```

## Benefits

1. **Resilience**: Training can survive crashes, network disconnections, or system restarts
2. **Flexibility**: Pause training anytime (Ctrl+C) and resume later
3. **Resource Efficiency**: Don't waste computation by restarting from scratch
4. **Progress Tracking**: Clear visibility into which stages are completed
5. **GPU Time Optimization**: Important for cloud GPU instances with time limits

## Important Notes

### Checkpoint Files Are Large
Each checkpoint is ~50-100 MB. Make sure you have sufficient disk space:
- 4 stage checkpoints: ~200-400 MB
- Emergency checkpoints (if any): ~50-100 MB each
- Final model: ~50-100 MB

### Training Parameters Are Preserved
When resuming from a checkpoint:
- The model architecture (weights, optimizer state) is fully restored
- Hyperparameters are updated for the new stage
- Total timesteps counter continues from where it left off

### Emergency Checkpoints Are Not Used for Auto-Resume
- `INTERRUPTED` and `ERROR` checkpoints are for manual recovery only
- Auto-resume only uses successfully completed stage checkpoints
- This prevents resuming from potentially corrupted partial training

## Example Training Session

```bash
$ python train_v9_advanced_ppo.py

======================================================================
  V9 ADVANCED PPO TRAINING (GPU ACCELERATED)
======================================================================

🚀 GPU: Tesla T4
   Device: cuda

No checkpoints found - starting fresh training

======================================================================
  STAGE 1/4: M4.5 @ 50kN
======================================================================
...
✅ Stage 1 complete!

======================================================================
  STAGE 2/4: M5.7 @ 100kN
======================================================================
...
[Ctrl+C pressed]

⚠️  Training interrupted by user!
   Saving emergency checkpoint...
   💾 Emergency checkpoint saved: models/rl_v9_advanced/stage2_INTERRUPTED_100kN.zip

   You can resume training by running the script again.
   Training will automatically resume from Stage 2.

$ python train_v9_advanced_ppo.py

======================================================================
  🔄 RESUMING FROM CHECKPOINT
======================================================================

   Found checkpoint: models/rl_v9_advanced/stage1_50kN.zip
   Last completed: Stage 1/4
   Loading model from checkpoint...

   ✅ Model loaded successfully!
   Will resume from Stage 2/4

======================================================================
  STAGE 2/4: M5.7 @ 100kN
======================================================================
...
[Training continues from Stage 2]
```

## Troubleshooting

### Problem: Checkpoint exists but training starts from scratch
**Solution**: Check if the checkpoint file is corrupted:
```bash
# Try loading the checkpoint manually
python -c "from stable_baselines3 import PPO; PPO.load('models/rl_v9_advanced/stage1_50kN.zip')"
```

### Problem: Want to force restart from a specific stage
**Solution**: Delete checkpoints after that stage:
```bash
# Delete all checkpoints and start fresh
rm -rf models/rl_v9_advanced/

# Or delete specific stage checkpoints
rm models/rl_v9_advanced/stage3_150kN.zip
rm models/rl_v9_advanced/stage4_150kN.zip
```

### Problem: Out of disk space
**Solution**: Clean up old emergency checkpoints:
```bash
# Remove emergency checkpoints (keep only successful ones)
rm models/rl_v9_advanced/*_INTERRUPTED_*.zip
rm models/rl_v9_advanced/*_ERROR_*.zip
```

## Technical Details

### Implementation
- **Checkpoint detection**: `find_last_completed_stage()` function
- **Auto-resume logic**: Checks for checkpoints at startup
- **Error handling**: Try-except blocks around `model.learn()`
- **Emergency saves**: Saves checkpoint before re-raising exceptions

### Checkpoint Contents
Each `.zip` file contains:
- Model architecture (policy network, value network)
- Trained weights (all layers)
- Optimizer state (Adam optimizer momentum, learning rate)
- Normalization statistics (if using observation normalization)

### Performance Impact
- **Checkpoint save time**: ~2-5 seconds per stage
- **Checkpoint load time**: ~2-5 seconds
- **Disk I/O**: Minimal impact on training speed
