# DCR Improvement - Ablation Study (Option C)

## Problem with Options 2 + 3
After implementing both Option 2 (expanded observations) and Option 3 (max drift penalty), the peak displacement **INCREASED catastrophically** from ~30cm to **1553cm** (15+ meters).

### Root Cause
The combined penalty weights were **too aggressive**:
- DCR penalty: `-5.0 × (dcr-1)²`
- Max drift penalty: `-2.0 × max_drift`

For typical values (DCR=2.5, max_drift=0.15m):
- DCR penalty: `-5.0 × 2.25 = -11.25`
- Max drift penalty: `-2.0 × 0.15 = -0.30`
- **Total drift penalties: -11.55**

This is **100x larger** than the displacement penalty (`-1.0 × 0.30 = -0.30`).

The agent learned: **"Minimize DCR at all costs, even if the building collapses"**

---

## Solution: Ablation Study (Option C)

### Strategy
Test improvements **incrementally** to isolate their effects:

1. **Keep Option 2**: Expanded observations (4→8 values)
   - Gives agent visibility into floor 8 (weak floor) and floor 6

2. **Moderate Option 1**: Reduce DCR penalty from 5.0 → **2.0**
   - 4x increase from original (0.5) - noticeable but not dominating

3. **Skip Option 3**: Remove max drift penalty entirely
   - Avoid the extreme penalty that caused catastrophic failure

### Hypothesis
**Expanded observations alone** (Option 2) should improve DCR, because:
- Agent can now "see" floor 8 drift in real-time
- Agent can proactively adjust TMD force based on weak floor response
- Moderate DCR penalty (2.0) provides guidance without dominating

---

## Implementation

### Reward Function Changes

#### Before (Options 2+3, FAILED):
```python
# DCR penalty
dcr_penalty = -5.0 * (dcr_deviation ** 2)  # TOO STRONG!

# Max drift penalty
max_drift_penalty = -2.0 * max_drift  # TOO STRONG!

reward = (
    displacement_penalty +    # -1.0 * |roof_disp|
    velocity_penalty +        # -0.3 * |roof_vel|
    force_penalty +           # 0.0
    smoothness_penalty +      # -0.005 * |force_change|
    acceleration_penalty +    # -0.1 * |roof_accel|
    max_drift_penalty +       # -2.0 * max_drift  ← REMOVED
    dcr_penalty               # -5.0 * (dcr-1)²   ← REDUCED
)
```

#### After (Option C, Ablation Study):
```python
# DCR penalty - MODERATE
dcr_penalty = -2.0 * (dcr_deviation ** 2)  # 4x increase (reasonable)

# NO max drift penalty

reward = (
    displacement_penalty +    # -1.0 * |roof_disp|
    velocity_penalty +        # -0.3 * |roof_vel|
    force_penalty +           # 0.0
    smoothness_penalty +      # -0.005 * |force_change|
    acceleration_penalty +    # -0.1 * |roof_accel|
    dcr_penalty               # -2.0 * (dcr-1)²  ← Moderate penalty only
)
```

### Penalty Comparison

| DCR Value | Original (0.5) | Option 1 (5.0) | **Ablation (2.0)** |
|-----------|----------------|----------------|--------------------|
| 1.0 (ideal) | 0 | 0 | **0** |
| 1.5 | -0.125 | -1.25 | **-0.5** |
| 2.0 | -0.5 | -5.0 | **-2.0** |
| 2.5 | -1.125 | -11.25 | **-4.5** |
| 3.0 | -2.0 | -20.0 | **-8.0** |

**Key insight**: With DCR=2.5 and displacement=0.30m:
- Displacement penalty: `-1.0 × 0.30 = -0.30`
- **Ablation DCR penalty: `-2.0 × 2.25 = -4.5`**
- Ratio: **15:1** (DCR penalty is 15x larger)

This is still significant but **not catastrophic** like the 100:1 ratio before.

---

## Expected Results

### What We're Testing
1. **Do expanded observations alone improve DCR?**
   - Agent can see floor 8 drift directly
   - Agent can react proactively to weak floor response

2. **Is moderate DCR penalty enough?**
   - 4x increase from original (0.5 → 2.0)
   - Strong enough to encourage uniform drift
   - Not so strong it dominates displacement

### Success Criteria
- ✅ Peak displacement: **30-40cm** (acceptable for M7.4)
- ✅ DCR: **1.5-2.0** (improvement from 2.5, may not reach 1.2)
- ✅ Training stability: No catastrophic failures

### If This Works
- Validates that **expanded observations are key**
- Shows that **extreme penalties are counterproductive**
- Provides baseline for future improvements

### If This Still Fails
- May need to train longer (more timesteps)
- May need to adjust network architecture
- May need to revisit observation normalization

---

## Files Modified

### Training Environment
**`rl/rl_cl/tmd_environment.py`**
- Line 365-388: Removed max drift penalty, reduced DCR penalty to 2.0
- Line 390-399: Removed max_drift_penalty from reward calculation
- Line 432-439: Removed max_drift from reward breakdown

### API Environment
**`restapi/rl_cl/rl_cl_tmd_environment.py`**
- Same changes as training environment

### Key Changes
```python
# REMOVED this entire section:
# max_drift_penalty = -2.0 * max_drift

# CHANGED from 5.0 to 2.0:
dcr_penalty = -2.0 * (dcr_deviation ** 2)  # Was: -5.0

# REMOVED from reward:
# max_drift_penalty +
```

---

## Training Instructions

### 1. Delete Old Failed Models
```bash
cd rl/rl_cl
rm -rf rl_cl_robust_models/*
```

The catastrophic models (1553cm peak displacement) must be deleted.

### 2. Train New Model
```bash
python train_final_robust_rl_cl.py
```

### 3. Monitor Training
Watch for:
- **DCR in reward breakdown** - should decrease gradually
- **Peak displacement** - should stay under 50cm for M7.4
- **Total reward** - should improve over time
- **No catastrophic failures** - displacement should never exceed 200cm

### 4. Expected Training Time
- 4 curriculum stages
- Total: ~700,000 timesteps
- Estimated: 1-2 hours (depends on hardware)

---

## Reward Weight Summary (Ablation Study)

| Component | Weight | Purpose | Change from Options 2+3 |
|-----------|--------|---------|------------------------|
| Displacement | 1.0 | Minimize roof displacement | - |
| Velocity | 0.3 | Dampen oscillations | - |
| **DCR** | **2.0** | **Uniform drift distribution** | **Reduced from 5.0** ⬇️ |
| Acceleration | 0.1 | Occupant comfort | - |
| Smoothness | 0.005 | Prevent force chattering | - |
| Force | 0.0 | No penalty (allow full force) | - |
| **Max Drift** | **REMOVED** | - | **Removed entirely** ❌ |

---

## Observations to Track

### 8-Value Observation Space (Unchanged)
```python
obs = [
    roof_disp, roof_vel,      # Roof (floor 12) - global response
    floor8_disp, floor8_vel,  # Floor 8 - weak floor (CRITICAL!)
    floor6_disp, floor6_vel,  # Floor 6 - mid-height reference
    tmd_disp, tmd_vel         # TMD - control device
]
```

This expanded observation space is **the key innovation** that should enable DCR improvement.

---

## Next Steps After Training

### If Successful (DCR < 2.0, Peak < 50cm)
1. ✅ Copy model to API: `cp rl_cl_robust_models/perfect_rl_final_robust.zip restapi/rl_cl/rl_cl_robust_models/`
2. ✅ Run full comparison with all controllers
3. ✅ Document final performance
4. 🎯 **Consider**: If DCR still not low enough (~1.2), try slightly increasing DCR penalty to 3.0

### If Still Failing
1. ❌ Check reward breakdown during training - is DCR penalty too weak?
2. ❌ Verify observation normalization - are floor values in reasonable range?
3. ❌ Consider curriculum modification - train longer on each stage?
4. ❌ Try alternative: Train with ONLY displacement penalty first, then fine-tune with DCR

---

## Comparison Table

| Approach | DCR Penalty | Max Drift Penalty | Observation Space | Result |
|----------|-------------|-------------------|-------------------|--------|
| **Original** | 0.5 | None | 4 values (roof+TMD) | DCR=2.5 ❌ |
| **Option 1** | 5.0 | None | 4 values | DCR=2.5 ❌ |
| **Options 2+3** | 5.0 | 2.0 | 8 values | Peak=1553cm 💥 |
| **Ablation (C)** | **2.0** | **None** | **8 values** | **TBD** 🤞 |

---

## Date
December 30, 2025

## Author
Siddharth (with Claude)
