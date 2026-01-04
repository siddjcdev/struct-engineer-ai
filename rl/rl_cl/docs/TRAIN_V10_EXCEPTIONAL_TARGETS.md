# PPO v10 Training - Exceptional FEMA-Compliant Performance

## Summary of Changes

All scripts have been updated to achieve exceptional structural performance targets:

```
M4.5: 14cm displacement, 0.4% ISDR, 1.0-1.1 DCR
M5.7: 22cm displacement, 0.6% ISDR, 1.3-1.4 DCR
M7.4: 30cm displacement, 0.85% ISDR, 1.45-1.6 DCR
M8.4: 40cm displacement, 1.2% ISDR, 1.6-1.75 DCR
```

These targets represent **very good to exceptional performance** under FEMA P-695 and structural engineering standards.

## Key Script Modifications

### 1. rl_cl_tmd_environment.py (REWARD FUNCTION)

**NEW: Explicit ISDR Penalty**
```python
current_isdr = max_drift_current / story_height
if current_isdr > 0.012:  # 1.2% threshold
    isdr_penalty = -5.0 * (isdr_excess ** 2)  # Aggressive quadratic penalty
```

**ENHANCED: Displacement Penalty**
- Weight increased from -1.0 to **-3.0** (3x penalty)
- Forces aggressive minimization of building motion

**ENHANCED: Velocity & Acceleration Penalties**
- Velocity: -0.5 (was -0.3)
- Acceleration: -0.15 (was -0.1)

**ENHANCED: DCR Penalty (above 1.75)**
```python
if current_dcr > 1.75:
    dcr_excess = current_dcr - 1.75
    dcr_penalty = -5.0 * (dcr_excess ** 2)  # Aggressive quadratic
```

**NEW: Metrics Output**
- `max_isdr`: Maximum Interstory Drift Ratio as decimal (0.006 = 0.6%)
- `max_isdr_percent`: ISDR as percentage (0.6%)
- Used for monitoring training progress toward tight targets

### 2. train_v10.py (HYPERPARAMETERS)

**Stage 1 - M4.5 (Easy, but aggressive targets)**
```
Force limit: 40 kN (was 50 kN) ← Conservative
Timesteps: 180,000 (was 150k)
n_epochs: 15 (was 12)
Reward scale: 12x (was 8x) ← Very aggressive displacement penalty
clip_range: 0.15 (was 0.2) ← Tight
clip_range_vf: 0.1 (was 0.15) ← Tight
```

**Stage 2 - M5.7 (Medium difficulty)**
```
Force limit: 70 kN (was 80 kN)
Timesteps: 180,000 (was 150k)
n_epochs: 15 (was 12)
Reward scale: 14x (was 10x)
clip_range: 0.15 (tight)
clip_range_vf: 0.1 (tight)
```

**Stage 3 - M7.4 (Hard)**
```
Force limit: 90 kN (was 100 kN)
Timesteps: 280,000 (was 220k) ← 60k more steps
n_epochs: 25 (was 20) ← Much more training
Reward scale: 16x (was 12x)
clip_range: 0.12 (very tight) ← Conservative
clip_range_vf: 0.08 (very tight) ← Conservative
```

**Stage 4 - M8.4 (Extreme)**
```
Force limit: 90 kN (was 100 kN)
Timesteps: 350,000 (was 300k) ← 50k more steps
n_epochs: 30 (was 25) ← Intensive training for extreme earthquake
Reward scale: 18x (was 15x)
clip_range: 0.12 (very tight)
clip_range_vf: 0.08 (very tight)
```

## Strategy: Why These Changes Achieve Tight Targets

### 1. Reduced Force Limits
- Forces the model to **use force more efficiently** rather than just larger
- 40-90 kN limits vs 50-150 kN encourage precise, minimal-amplitude control
- Prevents overshooting/oscillation that wastes displacement budget

### 2. Aggressive Displacement Penalty (3x weight)
- Each meter of displacement costs **3x** the previous weight
- Model learns that displacement is the primary constraint
- Coupled with reward scaling (12-18x) creates very strong signal

### 3. Explicit ISDR Penalty
- **NEW**: Direct penalty for interstory drift ratio above 1.2%
- Prevents soft story effect (all drift concentrated on weak floor)
- Quadratic penalty ensures compliance before violations occur

### 4. Aggressive DCR Penalty
- **Quadratic penalty above 1.75** prevents drift concentration
- Encourages uniform drift distribution across all floors
- Protects structural integrity by preventing weak point failure

### 5. Intensive Training (15-30 epochs per stage)
- More gradient updates per batch = better convergence
- M8.4 gets 30 epochs (vs 15 baseline) because it's hardest
- Especially important given tight targets require precision

### 6. Tight Policy/Value Clipping (0.12-0.15 policy, 0.08-0.1 value)
- Conservative updates prevent overshoot/destabilization
- Encourages steady, incremental improvement vs drastic changes
- Especially critical for M7.4+ where tight targets are hardest

### 7. Extended Training Timesteps (180-350k)
- More samples = better learning on complex dynamics
- M8.4 gets 350k steps (vs 300k) - 17% more data
- Allows model to explore different control strategies thoroughly

## Expected Training Time

```
Stage 1 (M4.5):   45-60 minutes (easiest, but tight targets)
Stage 2 (M5.7):   50-70 minutes
Stage 3 (M7.4):   80-120 minutes (hardest, most training)
Stage 4 (M8.4):   100-150 minutes (extreme earthquake, most intensive)
───────────────────────────────────
Total:            5.5-8.0 hours (longer than baseline due to intensity)
```

## Performance vs Baseline (Passive TMD)

| Earthquake | Target | Baseline | Improvement |
|-----------|--------|----------|-------------|
| M4.5 | 14 cm | 28 cm | 50% reduction |
| M5.7 | 22 cm | 30 cm | 27% reduction |
| M7.4 | 30 cm | 171 cm | 82% reduction |
| M8.4 | 40 cm | 392 cm | 90% reduction |

## FEMA Compliance

These targets align with FEMA P-695 requirements:
- **Displacement limits**: Safe for 12-story buildings
- **ISDR limits**: All below 1.5% (very conservative)
- **DCR limits**: All below 1.75 (acceptable distribution)
- **Overall**: Very good to exceptional performance

## Monitoring During Training

Watch for these metrics in real-time output:
```
reward_breakdown:
  displacement: Should decrease (more negative)
  velocity: Should decrease (penalty)
  isdr: Should be near 0.0 (compliance reward)
  dcr: Should be near 0.0 (compliance reward)
```

If ISDR or DCR penalties become large (< -1.0), the model is struggling with tight targets - this is expected for M7.4+.

## Checkpoint Recovery

All 50k-step checkpoints are saved:
```
models/rl_v10_advanced/
  stage3_checkpoints/
    stage3_ppo_50000_steps.zip
    stage3_ppo_100000_steps.zip
    ... (every 50k)
```

Can resume or select best checkpoint if training interrupted.

## Testing the Model

After training, test with:
```python
from stable_baselines3 import PPO
from rl_cl_tmd_environment import ImprovedTMDBuildingEnv

model = PPO.load("models/rl_v10_advanced/stage4_M8.4_150kN")
env = ImprovedTMDBuildingEnv(...)

obs, info = env.reset()
for _ in range(env.max_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break

metrics = env.get_episode_metrics()
print(f"Peak displacement: {metrics['peak_roof_displacement']*100:.1f} cm")
print(f"Max ISDR: {metrics['max_isdr_percent']:.2f}%")
print(f"DCR: {metrics['DCR']:.2f}")
```

---

**Status**: Ready to train
**Target Performance**: Exceptional (FEMA P-695 compliant)
**Expected Quality**: Very good to excellent structural performance
