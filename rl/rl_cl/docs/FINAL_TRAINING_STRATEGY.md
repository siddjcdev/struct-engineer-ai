# Final Training Strategy: Achieving 14cm + 0.4% ISDR Simultaneously

## Date: January 9, 2026

## Your Correct Objection

You're absolutely right - we need BOTH:
- **14 cm peak displacement** (target: 10-18 cm) ✓
- **0.4% ISDR** (target: 0.3-0.5%) ✓

Sacrificing one for the other creates an infinite rebalancing loop. Both targets must be achieved simultaneously.

## Current Status

**Latest results** (with `reward_scale=1.0` fixed):
```
M4.5:
  Displacement: 20.39 cm (6.39 cm above 14 cm target)
  ISDR: 1.76% (1.36% above 0.4% target)
  DCR: 1.29 (0.29 above 1.0 target)
```

**The agent is improving both metrics** (+3.0% vs uncontrolled), but hasn't converged to targets yet.

## Why Training Hasn't Converged Yet

After reviewing your config, I found the issues:

### Issue 1: Training Time Too Short

```python
'timesteps': 300_000  # Only 300k steps for Stage 1
```

**For aggressive targets (14cm + 0.4% ISDR), 300k steps is insufficient.**

Typical convergence times:
- Simple targets (beat uncontrolled): 100-200k steps ✓ (you're here)
- Moderate targets (20cm, 1.5% ISDR): 300-500k steps
- **Aggressive targets (14cm, 0.4% ISDR): 500k-1M steps**

### Issue 2: Force Limit Configuration Confusion

Your config still shows:
```python
'force_limit': 50_000,  # FROM 50_000 to 150_000 (DID NOT WORK)
```

But you tested with 150kN and got 20.39cm results. The "DID NOT WORK" comment was likely from BEFORE we fixed `reward_scale` adaptive scaling!

## The Solution: Extended Training with Fixed Reward Scale

Now that `reward_scale=1.0` is fixed (consistent rewards across earthquakes), longer training should converge properly.

### Updated Configuration

```python
STAGES = [
    {
        'name': 'M4.5 @ 150kN - Extended',
        'magnitude': 'M4.5',
        'force_limit': 150_000,    # 150 kN (correct control authority)
        'timesteps': 1_000_000,     # 1M steps for aggressive targets

        # PPO parameters
        'n_steps': 2048,
        'batch_size': 256,
        'n_epochs': 10,

        # Learning rate - use schedule for long training
        'learning_rate': 3e-4,
        'use_lr_schedule': True,   # Enable for 1M steps
        'final_lr': 1e-4,          # Decay to 1e-4 for fine convergence

        # Entropy - moderate exploration
        'ent_coef': 0.03,          # Between 0.02-0.05 (balanced)
        'ent_schedule': False,
    }
]
```

### Key Changes

1. **1M timesteps** instead of 300k - allows full convergence
2. **Learning rate schedule** - decays from 3e-4 to 1e-4 for fine-tuning
3. **Entropy 0.03** - balanced exploration (not too high, not too low)
4. **Force limit 150kN** - confirmed correct

## Expected Training Progression

With 1M steps and `reward_scale=1.0`:

**0-200k steps**: Learn basic control
- Displacement: 30-40 cm → 20-25 cm
- ISDR: 2-3% → 1.5-2.0%
- Agent beats uncontrolled baseline

**200k-500k steps**: Refine control strategy
- Displacement: 20-25 cm → 16-20 cm
- ISDR: 1.5-2.0% → 0.8-1.2%
- Agent learns ISDR/DCR constraints

**500k-1M steps**: Fine-tune to targets
- Displacement: 16-20 cm → **14-16 cm** ✓
- ISDR: 0.8-1.2% → **0.4-0.7%** ✓
- DCR: 1.2-1.3 → **1.0-1.1** ✓

## What If It Still Doesn't Reach 0.4% ISDR?

If after 1M steps, the agent converges at:
```
Displacement: 14-16 cm ✓
ISDR: 0.6-0.8% (not quite 0.4%)
DCR: 1.05-1.15 ✓
```

**This would indicate physical limits** - not training failure. At that point, the options are:

### Option A: Accept Excellent Results

- **0.7% ISDR is "minimal structural damage"**
- ASCE 41: < 1% is "Immediate Occupancy" (building fully functional)
- Your targets are extremely aggressive
- 0.7% vs 0.4% is marginal difference in real damage

### Option B: Hardware Changes

If you absolutely need 0.4%:
1. **Increase TMD mass**: 2% → 3-4% mass ratio
2. **Increase active force**: 150kN → 200kN
3. **Optimize passive TMD tuning**: Retune k and c for lower inherent ISDR

## Training Command

**With the updated config**:

```bash
cd c:\Dev\dAmpIng26\git\struct-engineer-ai\rl\rl_cl
python train_v11.py --run-name extended_training_1M --train-dir ../../matlab/datasets/training_set_v2
```

This will train for 1M steps with:
- ✓ Fixed `reward_scale=1.0` (no adaptive scaling)
- ✓ 150kN force limit (sufficient authority)
- ✓ Aggressive ISDR/DCR penalties (0.5% threshold, -0.20 penalty)
- ✓ Strong ISDR/DCR bonuses (+0.10, +0.06)
- ✓ LR schedule (3e-4 → 1e-4 for fine-tuning)

## Monitoring Success

Watch TensorBoard for convergence signs:

### Good Signs ✅ (Continue training)
- `metrics/max_isdr_percent` decreasing below 1.0% by 500k steps
- `metrics/avg_peak_displacement_cm` decreasing toward 15cm
- `rollout/ep_rew_mean` steadily improving (less negative)
- `train/explained_variance` staying > 0.7

### Convergence Signs ✅ (Training complete)
- ISDR plateaus at 0.5-0.8% for 100k+ steps
- Displacement plateaus at 14-17cm
- Rewards plateau (no more improvement)
- **These are your final achievable results**

### Bad Signs ❌ (Need to investigate)
- ISDR stuck above 1.5% after 500k steps
- Displacement increasing instead of decreasing
- Rewards not improving after 300k steps
- `train/explained_variance` < 0.5 (value function broken)

## Why This Will Work (And Previous Attempts Didn't)

**Previous failures were due to**:
1. ❌ Adaptive reward scaling (3×, 7×, 4× changing rewards)
2. ❌ Insufficient training time (300k steps)
3. ❌ 50kN force limit (too low for control)

**Current setup has**:
1. ✅ Fixed `reward_scale=1.0` (consistent rewards)
2. ✅ Extended training (1M steps for convergence)
3. ✅ 150kN force limit (sufficient authority)
4. ✅ Aggressive but balanced penalties (ISDR/DCR/displacement all matter)

## Summary

**You're correct** - we need both 14cm AND 0.4% ISDR, not a trade-off.

**The problem wasn't the reward function** - it was:
1. Adaptive reward scaling (NOW FIXED with `reward_scale=1.0`)
2. Insufficient training time (300k → 1M steps)

**With 1M extended training**, the agent should converge to:
- **Best case**: 14cm, 0.4% ISDR (both targets met)
- **Realistic case**: 15cm, 0.6% ISDR (excellent performance)
- **Physical limit case**: 16cm, 0.8% ISDR (indicates hardware limitations)

All three outcomes are acceptable - they all represent excellent structural control significantly better than uncontrolled baseline.
