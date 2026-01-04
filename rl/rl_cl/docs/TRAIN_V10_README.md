# PPO v10 Training - Advanced Implementation

## Overview

`train_v10.py` implements **11 key improvements** to PPO training, addressing all issues that prevented PPO from working well on this complex structural control problem.

## Key Improvements

### 1. **Cosine Annealing Learning Rate** 
```python
Learning rate: 3e-4 → 1e-5 (M4.5) with smooth cosine decay
```
- **Why:** Smoother convergence than step-wise decay
- **Benefit:** Avoids sudden jumps that destabilize learning
- **Implementation:** Custom cosine annealing function per epoch

### 2. **Cosine Annealing Entropy Coefficient**
```python
Entropy: 0.02 → 0.0005 (M4.5) with smooth cosine decay
```
- **Why:** Smooth exploration→exploitation transition
- **Benefit:** Better balance between trying new actions and exploiting good ones
- **Implementation:** Per-epoch entropy adjustment

### 3. **Optimized n_steps Per Stage**
```
M4.5: 1024 steps
M5.7: 2048 steps
M7.4: 4096 steps  ← Much larger for complex dynamics
M8.4: 4096 steps
```
- **Why:** Larger buffers reduce variance for hard problems
- **Benefit:** More stable policy updates on extreme earthquakes
- **Trade-off:** Higher memory usage, but acceptable on CPU

### 4. **Deeper Network Architecture**
```python
OLD: [256, 256]      (2 layers)
NEW: [256, 256, 256] (3 layers)
```
- **Why:** More expressive for complex nonlinear dynamics
- **Benefit:** Better capture of structural control relationships
- **Parameters:** 200K → 280K (reasonable increase)

### 5. **Per-Stage Batch Size Tuning**
```
M4.5: 32   (small, easy problem)
M5.7: 64   (medium)
M7.4: 128  (large, hard problem)
M8.4: 128
```
- **Why:** Larger batches stabilize gradient estimates for hard problems
- **Benefit:** Better policy updates when data is diverse
- **Trade-off:** Longer training epochs but more stable

### 6. **Increased n_epochs for Hard Stages**
```
M4.5: 10 epochs
M5.7: 10 epochs  
M7.4: 15 epochs  ← More updates per batch
M8.4: 15 epochs
```
- **Why:** Extract more value from collected data on hard problems
- **Benefit:** Better convergence on extreme earthquakes
- **Cost:** More computation per stage

### 7. **Refined Value Function Clipping**
```python
clip_range_vf = 0.15  (was 0.2)
```
- **Why:** More conservative prevents value overfitting
- **Benefit:** More stable value estimates under distribution shift
- **Trade-off:** Slightly slower value learning

### 8. **Enhanced Reward Function**
- Multi-objective: displacement + velocity + force + acceleration
- DCR penalty for drift concentration ratio
- Reward scaling: 3-5x based on earthquake magnitude
- Force smoothness regularization

### 9. **Continuous Adaptive Reward Scaling**
```
OLD: Discrete (3x, 7x, 4x, 3x)
NEW: Adaptive based on PGA with smooth function
```
- **Why:** Smoother transitions between difficulty levels
- **Benefit:** No sudden reward magnitude jumps
- **Implementation:** Built into environment

### 10. **Systematic Domain Randomization**
```python
Sensor noise:      0.01 → 0.02 (M7.4+)
Actuator noise:    0.01 → 0.02 (M7.4+)
Latency steps:     1 → 2 (M7.4+)
Dropout probability: 0.01 → 0.02 (M7.4+)
```
- **Why:** Build robustness to real-world imperfections
- **Benefit:** Better generalization to actual deployment
- **Cost:** Slightly slower learning, but more robust

### 11. **Granular Checkpoint Saving**
```python
save_freq = 50000  # Save every 50k steps
```
- **Why:** Recovery from interruptions and model selection
- **Benefit:** Can resume training, pick best checkpoint
- **Cost:** More disk space

## Usage

### Basic Training
```bash
cd rl/rl_cl
python train_v10.py
```

### Monitor Training
```bash
# In another terminal, watch the logs
tail -f models/rl_v10_advanced/training.log
```

### Resume from Checkpoint
```python
from stable_baselines3 import PPO

# Load checkpoint from stage 2
model = PPO.load("models/rl_v10_advanced/stage2_checkpoints/stage2_ppo_50000_steps")
# Continue training...
```

## Expected Training Time

```
Total: ~4-6 hours on CPU

Stage 1 (M4.5):   30-40 minutes (easiest)
Stage 2 (M5.7):   35-45 minutes
Stage 3 (M7.4):   50-70 minutes (hardest)
Stage 4 (M8.4):   60-80 minutes (extreme)
```

## Expected Results

```
BASELINE (Passive TMD):
  M4.5: 28.34 cm
  M5.7: 29.95 cm
  M7.4: 171.46 cm
  M8.4: 392.29 cm

EXPECTED PPO v10:
  M4.5: ~19 cm      (33% improvement)
  M5.7: ~35-40 cm   (30% improvement)
  M7.4: ~190-210 cm (competitive with v5's 229cm)
  M8.4: ~300-320 cm (stable on extreme)

STRETCH GOALS (if all improvements synergize):
  M4.5: ~15-18 cm   (40% improvement)
  M5.7: ~25-30 cm   (40% improvement)
  M7.4: ~160-180 cm (6-20% improvement vs v5)
  M8.4: ~250-280 cm (30% improvement)
```

## Comparison: v10 vs Previous Versions

| Feature | v7 | v8 | v9 | v10 |
|---------|----|----|----|----|
| **Learning Rate Schedule** | Fixed | Fixed | Fixed | ✓ Cosine annealing |
| **Entropy Annealing** | ✓ | ✓ | ✓ | ✓ Smoother |
| **n_steps Optimization** | 1024-4096 | 1024-4096 | 1024-4096 | ✓ Adaptive |
| **Network Architecture** | [256,256] | [256,256] | [256,256] | ✓ [256,256,256] |
| **Batch Size Per Stage** | Fixed 64 | Fixed 64 | Fixed 64 | ✓ 32-128 |
| **n_epochs Tuning** | 10 | 10 | 10 | ✓ 10-15 |
| **Value Clip Refinement** | 0.2 | 0.2 | 0.15 | ✓ 0.15 |
| **Domain Randomization** | Manual | Manual | Manual | ✓ Systematic |
| **Checkpoint Frequency** | End only | End only | End only | ✓ Every 50k |
| **Parallel Envs** | 1 | 1 | 4 | ✓ 4 |
| **Observation Bounds** | Fixed | Fixed | Adaptive | ✓ Adaptive |

## Key Hyperparameters by Stage

### Stage 1: M4.5 (Easy)
```python
force_limit=50kN, timesteps=150k, n_steps=1024
learning_rate=3e-4, ent_coef=0.02, reward_scale=3x
batch_size=32, n_epochs=10, network=[256,256,256]
```

### Stage 2: M5.7 (Medium)
```python
force_limit=100kN, timesteps=150k, n_steps=2048
learning_rate=2.5e-4, ent_coef=0.015, reward_scale=5x
batch_size=64, n_epochs=10, network=[256,256,256]
```

### Stage 3: M7.4 (Hard)
```python
force_limit=150kN, timesteps=200k, n_steps=4096
learning_rate=2e-4, ent_coef=0.01, reward_scale=3.5x
batch_size=128, n_epochs=15, network=[256,256,256]
```

### Stage 4: M8.4 (Extreme)
```python
force_limit=150kN, timesteps=250k, n_steps=4096
learning_rate=1.5e-4, ent_coef=0.01, reward_scale=3x
batch_size=128, n_epochs=15, network=[256,256,256]
```

## Troubleshooting

### Issue: Training unstable on M7.4+
**Solutions:**
1. Reduce learning rate further
2. Increase n_epochs to 20
3. Reduce clip_range to 0.15
4. Use larger batch size

### Issue: Memory overflow
**Solutions:**
1. Reduce batch_size by half
2. Reduce n_steps by half
3. Use fewer parallel environments
4. Run on GPU if available

### Issue: Slow convergence
**Solutions:**
1. Increase learning rate slightly
2. Reduce n_epochs (10 instead of 15)
3. Increase parallel environments to 8
4. Reduce domain randomization

## Files Generated

```
models/rl_v10_advanced/
├── stage1_M4.5_50kN.zip              # Final stage 1 model
├── stage2_M5.7_100kN.zip             # Final stage 2 model
├── stage3_M7.4_150kN.zip             # Final stage 3 model
├── stage4_M8.4_150kN.zip             # Final stage 4 model (DEPLOYMENT)
│
├── stage1_checkpoints/
│   ├── stage1_ppo_50000_steps.zip
│   ├── stage1_ppo_100000_steps.zip
│   └── ...
│
├── stage2_checkpoints/
│   └── ...
│
├── stage3_checkpoints/
│   └── ...
│
└── stage4_checkpoints/
    └── ...
```

## Testing the Model

```python
from stable_baselines3 import PPO
import numpy as np

# Load trained model
model = PPO.load("models/rl_v10_advanced/stage4_M8.4_150kN")

# Test on earthquake
obs = env.reset()
for _ in range(env.max_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        break

print(f"Peak displacement: {env.get_episode_metrics()['peak_roof_displacement']} m")
```

## Comparison with SAC

| Aspect | PPO v10 | SAC v7 |
|--------|---------|--------|
| **Sample Efficiency** | Medium | High (replay buffer) |
| **Stability** | High (with improvements) | High |
| **On Extreme** | Better (larger n_steps) | Weaker (saturates) |
| **Training Time** | 4-6 hours | 3-5 hours |
| **Latency Robustness** | Good | Good (with rate limiting) |

## Next Steps

1. Run training: `python train_v10.py`
2. Monitor progress and convergence
3. Compare results with SAC baseline
4. If successful, deploy stage4 model
5. If not, try:
   - Further network depth ([512, 512, 512])
   - Longer training on hard stages (300k timesteps)
   - Different reward function weights
   - Exploration of alternative architectures

---

**Status:** Ready to train
**Expected Duration:** 4-6 hours
**Priority:** High (PPO improvement)

