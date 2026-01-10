# CRITICAL FIX: Adaptive Reward Scaling Was Destroying Training

## Date: January 9, 2026

## Problem Discovered

Training results showed the agent performing **worse than uncontrolled** across all earthquake magnitudes:

```
M4.5: 21.22 cm (uncontrolled: 21.02 cm) - agent does NOTHING
M5.7: 51.86 cm (uncontrolled: 46.02 cm) - agent makes it WORSE
M7.4: 246.30 cm (uncontrolled: 235.55 cm) - catastrophic
M8.4: 380.41 cm (uncontrolled: 357.06 cm) - catastrophic
```

The agent learned nothing useful despite:
- ✅ Correct force limit (150 kN)
- ✅ Proper network architecture (3-4 layers)
- ✅ Tuned hyperparameters (learning rate, entropy, etc.)
- ✅ Sufficient training time (300k-950k timesteps)

## Root Cause: Inconsistent Reward Signals

The environment was using **adaptive reward scaling** that varied by earthquake magnitude:

```python
# From tmd_environment_adaptive_reward.py _compute_adaptive_reward_scale()
if pga_g < 0.30:      # M4.5 range
    scale = 3.0
elif pga_g < 0.55:    # M5.7 range
    scale = 7.0       # 2.3× different from M4.5!
elif pga_g < 0.85:    # M7.4 range
    scale = 4.0
else:                 # M8.4 range
    scale = 3.0
```

### Why This Destroys Training

During curriculum learning, the agent experiences:

1. **M4.5 earthquake** → takes action A → gets reward -100 (with 3× scale)
2. **M5.7 earthquake** → takes **same action A** → gets reward -233 (with 7× scale)
3. Agent thinks: "That action was terrible on M5.7!" (but it was the same action!)
4. Agent learns: "Different earthquakes need completely different strategies"
5. **Result**: Policy never converges because reward signal keeps changing

### The Math

With the same control performance:
- M4.5: `reward = -33.3 × 3.0 = -100`
- M5.7: `reward = -33.3 × 7.0 = -233` (2.3× worse for same performance!)

The agent cannot learn a general control policy when the reward function changes based on earthquake magnitude.

## The Fix

### 1. Added `reward_scale` Parameter to Factory Function

**File**: `tmd_environment_adaptive_reward.py`

```python
def make_improved_tmd_env(
    earthquake_file: str,
    earthquake_name: str = None,
    max_force: float = 150000.0,
    sensor_noise_std: float = 0.0,
    actuator_noise_std: float = 0.0,
    latency_steps: int = 0,
    dropout_prob: float = 0.0,
    obs_bounds: dict = None,
    reward_scale: float = None  # NEW: Allow fixed reward scale
) -> ImprovedTMDBuildingEnv:
    """
    Args:
        reward_scale: Fixed reward scaling multiplier (None = auto-compute based on earthquake PGA)
                     Use 1.0 for consistent training across different magnitudes
    """
    return ImprovedTMDBuildingEnv(
        earthquake_data=accelerations,
        dt=dt,
        max_force=max_force,
        earthquake_name=earthquake_name,
        sensor_noise_std=sensor_noise_std,
        actuator_noise_std=actuator_noise_std,
        latency_steps=latency_steps,
        dropout_prob=dropout_prob,
        obs_bounds=obs_bounds,
        reward_scale=reward_scale  # Pass through reward_scale
    )
```

### 2. Updated Training to Use Fixed Scaling

**File**: `train_v11.py`

```python
# Training environments - use reward_scale=1.0 for consistency
def make_env_factory(train_files, force_limit):
    def _make_env():
        eq_file = random.choice(train_files)
        env = make_improved_tmd_env(eq_file, max_force=force_limit, reward_scale=1.0)
        return Monitor(env)
    return _make_env

# Testing environments - also use reward_scale=1.0
def test_on_earthquake(model, test_file, force_limit):
    test_env = make_improved_tmd_env(test_file, max_force=force_limit, reward_scale=1.0)
    # ...
```

## What This Changes

### Before (Adaptive Scaling)
```
M4.5 training: reward × 3.0
M5.7 training: reward × 7.0
M7.4 training: reward × 4.0
M8.4 training: reward × 3.0

Agent sees: "Same action = different rewards depending on earthquake"
Result: Policy never converges, performs worse than uncontrolled
```

### After (Fixed Scaling = 1.0)
```
M4.5 training: reward × 1.0
M5.7 training: reward × 1.0
M7.4 training: reward × 1.0
M8.4 training: reward × 1.0

Agent sees: "Same action = consistent reward across all earthquakes"
Result: Policy can learn general control strategy
```

## Why Adaptive Scaling Was Implemented

The original intent (from v7 comments):
- "Automatically adjusts reward signal based on earthquake intensity"
- "Prevents gradient instability on extreme earthquakes"

**The problem**: This is backwards! The solution to different earthquake intensities is:
1. ✅ **Curriculum learning** - train on easy → hard earthquakes progressively
2. ✅ **Normalization** - observations are already normalized to [-1, 1] bounds
3. ❌ **NOT adaptive reward scaling** - this breaks policy convergence

## Expected Improvement

With fixed `reward_scale=1.0`, the agent should:

1. **Learn consistent policy** - same action gets same reward
2. **Generalize across magnitudes** - learns "reduce displacement" not "survive M5.7 differently than M4.5"
3. **Beat uncontrolled baseline** - actually learn useful control
4. **Achieve realistic targets**:
   - M4.5: 15-20 cm (was 21.22 cm doing nothing)
   - M5.7: 30-35 cm (was 51.86 cm making it worse)
   - M7.4: ~100 cm (was 246.30 cm catastrophic)
   - M8.4: ~200 cm (was 380.41 cm catastrophic)

## Files Modified

1. **tmd_environment_adaptive_reward.py**:
   - Added `reward_scale` parameter to `make_improved_tmd_env()` (line 647)
   - Updated docstring to document new parameter (line 662-663)
   - Pass `reward_scale` to environment constructor (line 713)

2. **train_v11.py**:
   - Training env factory uses `reward_scale=1.0` (line 195)
   - Test function uses `reward_scale=1.0` (line 382)

## How to Retrain

Simply run training again - the fix is already in place:

```bash
cd c:\Dev\dAmpIng26\git\struct-engineer-ai\rl\rl_cl
python train_v11.py --run-name fixed_reward_scale_v1
```

You should see:
- NO more "Adaptive reward scale: X.X× for PGA=..." messages during training
- Rewards improving consistently across all curriculum stages
- Final results beating uncontrolled baseline

## Comparison: Before vs After Fix

| Metric | Before (Adaptive) | After (Fixed 1.0) | Expected |
|--------|------------------|-------------------|----------|
| M4.5 displacement | 21.22 cm | ? | 15-20 cm |
| M4.5 vs uncontrolled | -1.0% worse | ? | +25-30% better |
| M5.7 displacement | 51.86 cm | ? | 30-35 cm |
| M5.7 vs uncontrolled | -12.7% worse | ? | +20-25% better |
| Agent learning | None | ? | Consistent |
| Policy convergence | Failed | ? | Success |

## Key Takeaway

**Never use adaptive reward scaling during curriculum learning!**

The curriculum already handles different difficulty levels. Adaptive scaling breaks the fundamental assumption of RL: **consistent reward signal for consistent behavior**.

If you need magnitude-specific adjustments, do it through:
1. Observation normalization (already done)
2. Curriculum progression (already done)
3. Per-magnitude hyperparameters (learning rate schedules)
4. **NOT** changing the reward function itself!
