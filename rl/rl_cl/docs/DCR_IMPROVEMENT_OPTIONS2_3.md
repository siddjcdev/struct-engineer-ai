# DCR Improvement - Options 2 + 3 Implementation

## Problem
After implementing Option 1 (increased DCR penalty weight from 0.5 to 5.0), DCR did not improve significantly. The RL agent still showed DCR values of 2.3-2.7, indicating poor drift distribution.

## Root Cause Analysis
The agent couldn't "see" what was happening at vulnerable floors. With only 4 observations (roof + TMD), the agent had:
- **No visibility** into floor 8 (weak floor where drift concentrates)
- **No visibility** into other floors
- **No direct incentive** to minimize max drift (only indirect through DCR calculation)

## Solution: Combined Options 2 + 3

### Option 2: Expanded Observation Space
**Changed observation from 4 → 8 values**

#### Before:
```python
obs = [roof_disp, roof_vel, tmd_disp, tmd_vel]  # 4 values
```

#### After:
```python
obs = [
    roof_disp, roof_vel,      # Roof (floor 12) - global response
    floor8_disp, floor8_vel,  # Floor 8 - weak floor (critical!)
    floor6_disp, floor6_vel,  # Floor 6 - mid-height reference
    tmd_disp, tmd_vel         # TMD - control device
]  # 8 values
```

**Why these floors?**
- **Floor 8 (index 7)**: Weak floor with 50% reduced stiffness - critical for DCR
- **Floor 6 (index 5)**: Mid-height, provides reference for distribution
- **Roof (floor 12)**: Overall building response
- **TMD**: Control device state

### Option 3: Direct Max Drift Penalty
**Added explicit penalty on maximum interstory drift**

```python
# NEW: Compute max drift across all floors
floor_drifts = self._compute_interstory_drifts(self.displacement[:12])
max_drift = np.max(floor_drifts)
max_drift_penalty = -2.0 * max_drift  # Strong direct penalty
```

This gives the agent a **direct, immediate signal** to reduce peak drift, complementing the DCR penalty.

## Updated Reward Function

### Weights (in order of priority):

| Component | Weight | Purpose | Change |
|-----------|--------|---------|--------|
| **Max Drift** | **2.0** | **Minimize peak drift** | **NEW** ⬅️ |
| Displacement | 1.0 | Minimize roof displacement | - |
| Velocity | 0.3 | Dampen oscillations | - |
| DCR | 5.0 | Uniform drift distribution | Increased from 0.5 |
| Acceleration | 0.1 | Occupant comfort | - |
| Smoothness | 0.005 | Prevent force chattering | - |
| Force | 0.0 | No penalty (allow full force) | - |

### Combined Reward:
```python
reward = (
    displacement_penalty +     # -1.0 * |roof_disp|
    velocity_penalty +         # -0.3 * |roof_vel|
    force_penalty +            # 0.0
    smoothness_penalty +       # -0.005 * |force_change|
    acceleration_penalty +     # -0.1 * |roof_accel|
    max_drift_penalty +        # -2.0 * max(floor_drifts)  ← NEW!
    dcr_penalty                # -5.0 * (dcr-1)²
)
```

## Impact on Training

### What the Agent Now Learns:

1. **Direct Drift Awareness**: Agent sees floor 8 displacement/velocity in real-time
2. **Proactive Control**: Can adjust TMD force based on weak floor response
3. **Strong Max Drift Penalty**: Immediate feedback when drift concentrates
4. **Distributed Control**: Learns to keep drift uniform across all floors

### Expected Improvements:

| Metric | Before | Target | Mechanism |
|--------|--------|--------|-----------|
| DCR | 2.3-2.7 | 1.1-1.3 | Direct floor visibility + penalties |
| Max Drift | High at floor 8 | Distributed | Max drift penalty + floor observations |
| Roof Disp | ~30cm | ~35cm | May increase slightly (acceptable trade-off) |

### Training Considerations:

1. **Observation space changed 4→8**: **Must train from scratch** (can't use old models)
2. **Increased complexity**: More observations = slightly harder to learn
3. **Training time**: May need 10-20% more timesteps to converge
4. **Better generalization**: More information → smarter controller

## Files Modified

### Training Environment:
- `rl/rl_cl/tmd_environment.py`
  - Line 74-85: Expanded observation space 4→8
  - Line 235-245: Updated reset() observation
  - Line 304-340: Updated step() observation with domain randomization
  - Line 365-369: Added max drift penalty
  - Line 396-404: Added max_drift_penalty to reward
  - Line 437-445: Added max_drift to reward_breakdown

### API Environment:
- `restapi/rl_cl/rl_cl_tmd_environment.py`
  - Same changes as training environment

## How to Train

### 1. Delete Old Models
```bash
rm -rf rl/rl_cl/rl_cl_robust_models/*
```
Old models have 4-value observations and **cannot** work with new 8-value environment.

### 2. Train New Model
```bash
cd rl/rl_cl
python train_final_robust_rl_cl.py
```

### 3. Monitor Training
Watch for:
- **DCR in reward breakdown** - should decrease over time
- **Max drift penalty** - should become less negative (drift reducing)
- **Total reward** - should increase (will be more negative initially due to new penalties)

### 4. Expected Training Output
```
Reward breakdown (example):
  displacement: -0.25
  velocity: -0.15
  max_drift: -0.08  ← Should decrease over time
  dcr: -1.2         ← Should decrease over time
  Total: -1.73
```

## Validation

After training, check final metrics:
```python
# Test on PEER M7.4 earthquake
Peak displacement: ~35-40 cm (acceptable increase)
DCR: ~1.1-1.3 (major improvement from 2.5)
Max drift: Distributed across floors (not concentrated at floor 8)
```

## Rollback

If results are worse, restore original environment:
```bash
git checkout rl/rl_cl/tmd_environment.py
git checkout restapi/rl_cl/rl_cl_tmd_environment.py
```

## Technical Details

### Observation Space:
- **Shape**: Changed from (4,) to (8,)
- **Bounds**: All displacements ±1.2m, velocities ±3.0m/s
- **Floor Indexing**:
  - Floor 6 = index 5 (Python 0-indexed)
  - Floor 8 = index 7 (weak floor)
  - Floor 12 (roof) = index 11

### Max Drift Calculation:
```python
# Compute interstory drifts for all 12 floors
drifts = [
    |disp[0]|,                    # Floor 1 (relative to ground)
    |disp[1] - disp[0]|,         # Floor 2
    ...
    |disp[11] - disp[10]|        # Floor 12 (roof)
]
max_drift = max(drifts)
```

### Why Weight = 2.0 for Max Drift?
- Displacement weight = 1.0 (per meter of roof displacement)
- Max drift weight = 2.0 (per meter of floor drift)
- This makes the agent care **twice as much** about concentrated drift vs overall displacement
- Combined with DCR penalty (5.0), total drift-related penalty dominates

## Summary

**Changes Made:**
1. ✅ Expanded observations from 4 → 8 (added floor 8, floor 6)
2. ✅ Added max drift penalty (weight = 2.0)
3. ✅ Kept DCR penalty at increased weight (5.0)
4. ✅ Updated both training and API environments

**Expected Outcome:**
- DCR improves from 2.5 → 1.2 (comparable to Fuzzy controller)
- Drift distributes uniformly instead of concentrating at floor 8
- Roof displacement may increase slightly (~10-15%) but still acceptable

**Next Action:**
**Train new model from scratch** - old models incompatible with new observation space!

---

**Date**: December 29, 2025
**Author**: Siddharth (with Claude)
