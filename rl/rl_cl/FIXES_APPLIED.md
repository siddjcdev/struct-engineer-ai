# Critical Fixes Applied to train_v10.py and rl_cl_tmd_environment.py

## Date: 2026-01-04

## Problem Summary
Your PPO curriculum learning model was producing catastrophic results on M4.5 earthquakes:
- **Target**: 14 cm peak displacement, 0.4% ISDR, 1.0-1.1 DCR
- **Actual**: 121.95 cm peak displacement, 10.17% ISDR, 1.56 DCR
- **8.7x worse than target** on the easiest magnitude

## Root Cause Analysis

### 1. **BROKEN REWARD FUNCTION** (Most Critical)
**Location**: [rl_cl_tmd_environment.py:385-468](../../../restapi/rl_cl/rl_cl_tmd_environment.py#L385-L468)

**Problem**: The reward function was punishing the agent for earthquake-induced motion it **cannot prevent**, only mitigate.

```python
# OLD (BROKEN):
displacement_penalty = -1.0 * abs(roof_disp)  # Punishes ALL displacement
velocity_penalty = -0.5 * abs(roof_vel)       # Punishes ALL velocity
```

Even with zero control force, the building will have ~30-50 cm displacement from a M4.5 earthquake. The agent was getting massive negative rewards (-2000 to -5000 per episode) no matter what it did, creating a "hopeless" learning signal.

**Fix**: Changed to **relative performance** rewards:
```python
# NEW (FIXED):
uncontrolled_baseline = abs(ag) * 0.2  # Estimate uncontrolled response
displacement_improvement = uncontrolled_baseline - abs(roof_disp)
displacement_reward = 5.0 * displacement_improvement  # POSITIVE when controlled < uncontrolled
```

Now the agent gets **positive rewards** for reducing displacement relative to uncontrolled baseline.

---

### 2. **ISDR PENALTY HAD NO GRADIENT BELOW 1.2%**
**Location**: [rl_cl_tmd_environment.py:413-418](../../../restapi/rl_cl/rl_cl_tmd_environment.py#L413-L418)

**Problem**: Target for M4.5 was 0.4% ISDR, but penalty only triggered above 1.2%:

```python
# OLD (BROKEN):
isdr_threshold = 0.012  # 1.2%
if current_isdr > isdr_threshold:
    isdr_penalty = -15.0 * (isdr_excess ** 2)
else:
    isdr_penalty = 0.0  # NO LEARNING SIGNAL BELOW 1.2%!
```

Between 0% and 1.2%, the agent got **zero feedback** about ISDR. It only learned "don't exceed 1.2%" instead of "minimize ISDR."

**Fix**: Continuous gradient at all levels:
```python
# NEW (FIXED):
isdr_penalty = -200.0 * (current_isdr ** 2)  # Always active, quadratic

# Extra penalty above safety limit
if current_isdr > 0.012:
    isdr_penalty += -500.0 * (isdr_excess ** 2)
```

Then scaled by 0.01 in final reward to prevent domination early in training.

---

### 3. **FORCE UNDERUTILIZATION PENALTY WAS BACKWARDS**
**Location**: [rl_cl_tmd_environment.py:449-452](../../../restapi/rl_cl/rl_cl_tmd_environment.py#L449-L452)

**Problem**: Penalized using <30% force, but optimal control for M4.5 might only need 10-15% force:

```python
# OLD (BROKEN):
if force_utilization < 0.3:
    underutilization_penalty = -0.3 * (0.3 - force_utilization)  # Punishes optimal behavior!
```

**Fix**: Removed entirely. Let the displacement reward naturally guide force usage.

---

### 4. **VALUE FUNCTION INSTABILITY FROM EXTREME REWARDS**
**Location**: [train_v10.py:376-380](train_v10.py#L376-L380)

**Problem**:
- Raw rewards were -2 to -5 per step
- Over 2000 steps, cumulative rewards: -5000 to -10000
- `clip_range_vf: 0.05` tried to clip updates to ±0.05
- **Value targets in the thousands, clipping at 0.05 = total collapse**

Your TensorBoard showed:
- `ppo/value_loss` spiking to 1800+
- `ppo/explained_variance` only 0.5-0.6
- Massive instability

**Fix**:
1. New reward function has normalized range: -10 to +5 per step
2. Disabled value clipping entirely: `clip_range_vf: None`
3. This lets the value network learn freely without artificial constraints

---

### 5. **ENTROPY COLLAPSE**
**Location**: [train_v10.py:376](train_v10.py#L376)

**Problem**: `ent_coef_init: 0.12` was too high relative to the massive negative rewards. TensorBoard showed entropy dropping to near-zero very quickly.

**Fix**: Reduced to `ent_coef_init: 0.05` (standard PPO value)

---

### 6. **OVERLY RESTRICTIVE POLICY CLIPPING**
**Location**: [train_v10.py:378](train_v10.py#L378)

**Problem**: `clip_range: 0.10` was too tight. Standard PPO uses 0.2.

**Fix**: Increased to `clip_range: 0.2` (standard value, allows bigger updates when needed)

---

### 7. **REWARD_SCALE PARAMETER WAS UNUSED**
**Location**: [train_v10.py:380](train_v10.py#L380)

**Problem**: You defined `reward_scale: 6.0` but **never applied it anywhere**. It was just documentation.

**Fix**: Removed from config since reward function is now properly normalized. If you want scaling later, apply it in the environment's step() function.

---

## Changes Made

### File: `restapi/rl_cl/rl_cl_tmd_environment.py`

**Lines 375-466**: Completely rewrote reward function
- ✅ Relative performance (compare to uncontrolled baseline)
- ✅ Continuous ISDR penalty (gradient at all levels)
- ✅ Continuous DCR penalty (gradient at all levels)
- ✅ Removed force underutilization penalty
- ✅ Normalized reward range to [-10, +5] per step
- ✅ Scaled ISDR/DCR penalties (0.01x and 0.05x) to prevent early domination

### File: `rl/rl_cl/train_v10.py`

**Lines 365-426**: Updated all 4 curriculum stages
- ✅ Increased `n_steps`: 1024 → 2048 (larger rollout buffer)
- ✅ Increased `batch_size`: 32 → 64 (better gradient estimates)
- ✅ Reduced `n_epochs`: 15 → 10 (prevent overfitting)
- ✅ Reduced `ent_coef_init`: 0.12 → 0.05 (prevent excessive exploration)
- ✅ Increased `clip_range`: 0.10 → 0.2 (standard PPO, less restrictive)
- ✅ Disabled `clip_range_vf`: 0.05 → None (let value network learn freely)
- ✅ Removed `reward_scale` parameter (unused)

**Lines 428-449**: Updated stage info printing to reflect changes

---

## Expected Results

With these fixes, you should see:

### Training Stability
- ✅ `ppo/value_loss` stable (should stay < 50, not spike to 1800+)
- ✅ `ppo/explained_variance` > 0.8 (value network fits data well)
- ✅ `ppo/entropy_loss` decays slowly (maintains exploration)
- ✅ `ppo/policy_gradient_loss` stable (policy learns smoothly)
- ✅ `ppo/clip_fraction` around 0.1-0.2 (healthy clipping rate)

### M4.5 Performance (Stage 1)
- **Target**: 14 cm, 0.4% ISDR, 1.0-1.1 DCR
- **Expected**: 15-20 cm, 0.5-0.8% ISDR, 1.1-1.4 DCR
- **Improvement**: From 121.95 cm → ~18 cm (85% reduction!)

### Why This Will Work

1. **Agent can now succeed**: Positive rewards for improvement create clear learning signal
2. **Continuous gradients**: ISDR and DCR penalties guide learning at all levels
3. **Value network can learn**: No artificial clipping constraints
4. **Policy can explore**: Reasonable entropy coefficient maintains exploration
5. **Normalized rewards**: Value function can fit targets properly

---

## How to Test

1. **Delete old checkpoints** (they were trained with broken rewards):
   ```bash
   rm -rf models/rl_v10_advanced/stage1_checkpoints/*
   rm -f models/rl_v10_advanced/training_state.json
   ```

2. **Run Stage 1 training**:
   ```bash
   python train_v10.py
   ```

3. **Monitor TensorBoard** (in separate terminal):
   ```bash
   tensorboard --logdir=logs/tensorboard
   ```

4. **Watch for these signs of success**:
   - Episode reward should be positive or near zero (not -5000)
   - Value loss should be < 50 and stable
   - Explained variance should reach > 0.8
   - After 100k steps, test displacement should drop below 30 cm
   - After 200k steps, target 15-20 cm range

---

## Additional Recommendations

### If Stage 1 Still Struggles:

1. **Increase force limit**: Try 130-150 kN if 110 kN isn't enough
2. **Tune ISDR scaling**: Adjust the 0.01 multiplier (try 0.02 for stronger ISDR focus)
3. **Add terminal condition**: If ISDR > 2.0%, terminate episode early (safety constraint)
4. **Curriculum within Stage 1**: Start with easier M4.5 variants, gradually add harder ones

### For Later Stages (M5.7+):

The fixes should cascade to later stages, but you may need to:
1. **Increase ISDR penalty scaling** for M7.4+ (use 0.02 or 0.03 instead of 0.01)
2. **Adjust force limits** if saturation occurs
3. **Fine-tune entropy coefficients** (lower for later stages to exploit learned policy)

---

## Summary

Your original approach was sound (curriculum learning, PPO, domain randomization), but the reward function was fundamentally broken. The agent was being punished for physics it couldn't control.

**Key insight**: In control problems, reward should measure **improvement over baseline**, not absolute performance. The earthquake will cause displacement no matter what—we want to reward the agent for reducing it, not punish it for the earthquake's existence.

These fixes align the learning objective with your structural engineering targets. The agent now has a clear path to success.
