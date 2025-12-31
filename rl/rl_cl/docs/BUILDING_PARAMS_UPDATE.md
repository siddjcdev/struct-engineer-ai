# Building Parameters Update - MATLAB Alignment

## Summary
Updated Python environments to match MATLAB building configuration for consistent, comparable results.

**Date**: December 30, 2025
**Status**: ✅ APPLIED

---

## Problem

MATLAB and Python were using **completely different building models**, making results incomparable:

| Parameter | MATLAB (Original) | Python (Old) | Ratio |
|-----------|-------------------|--------------|-------|
| **Floor mass** | 200,000 kg | 300,000 kg | 1.5x heavier |
| **Story stiffness** | 20 MN/m | 800 MN/m | **40x stiffer** |
| **Soft story factor** | 60% | 50% | Different |
| **Damping ratio** | 1.5% | 2.0% | Different |
| **Time step** | 0.01s (10ms) | 0.02s (20ms) | 2x coarser |

### Impact
- ❌ MATLAB showed 35cm peak displacement
- ❌ Python showed 1553cm peak displacement (before bug fixes)
- ❌ Results could not be compared between platforms
- ❌ Trained model performed poorly on actual MATLAB building

The Python building was **40x stiffer** - like comparing a concrete tower to a steel frame building!

---

## Solution

Updated both training and API environments to use **identical MATLAB parameters**.

### Files Changed

1. **`rl/rl_cl/tmd_environment.py`** (lines 52-65)
2. **`restapi/rl_cl/rl_cl_tmd_environment.py`** (lines 52-65)

### Changes Applied

```python
# OLD (Mismatched):
self.floor_mass = 300000           # 300,000 kg
k_typical = 800e6                  # 800 MN/m (very stiff!)
self.story_stiffness[7] = 0.5 * k_typical  # 50% soft story
self.damping_ratio = 0.02          # 2% damping

# NEW (Matches MATLAB):
self.floor_mass = 2.0e5            # 200,000 kg - matches MATLAB m0
k_typical = 2.0e7                  # 20 MN/m - matches MATLAB k0
self.story_stiffness[7] = 0.60 * k_typical  # 60% - matches MATLAB soft_story_factor
self.damping_ratio = 0.015         # 1.5% - matches MATLAB zeta_target
```

### MATLAB Reference
From `matlab/src/tmd/RUN_4WAY_COMPARISON.m`:
```matlab
m0 = 2.0e5;                  % kg per floor
k0 = 2.0e7;                  % N/m stiffness
zeta_target = 0.015;         % 1.5% damping
soft_story_factor = 0.60;    % 60% stiffness
```

---

## Impact on Training

### Before This Change
Model was trained on a **much stiffer building** than what it would be tested on:
- Training: 800 MN/m stiffness → smaller displacements, different dynamics
- Testing (MATLAB): 20 MN/m stiffness → larger displacements, softer structure
- Result: Model learned wrong control strategy

### After This Change
✅ Model trains on **exact same building** as MATLAB testing:
- Training: 20 MN/m stiffness
- Testing (MATLAB): 20 MN/m stiffness
- Result: Direct comparison, consistent behavior

---

## Expected Improvements

### 1. Comparable Results
- Python and MATLAB should now show similar peak displacements
- DCR metrics should be directly comparable
- Control force strategies should transfer correctly

### 2. More Realistic Training
- Building will have **larger displacements** during training (less stiff)
- Agent will learn to handle a **softer, more flexible** structure
- Better matches real-world buildings (most aren't 800 MN/m!)

### 3. Consistent API Behavior
- REST API `/simulate` endpoint now uses same building as MATLAB
- No more confusion about why results differ
- Model deployment matches training conditions

---

## Building Characteristics Comparison

### Old Python Building (800 MN/m)
- Very stiff structure (like reinforced concrete)
- Small displacements even for large earthquakes
- High natural frequencies
- **Unrealistic** for most structures
- Model learned aggressive control (building can take it)

### New Python Building (20 MN/m) - Matches MATLAB
- Moderate stiffness (like steel frame)
- Larger displacements (more realistic)
- Lower natural frequencies
- **Matches** actual structural engineering practice
- Model learns gentler control (building more flexible)

---

## Time Step Consideration

**MATLAB uses dt = 0.01s (10ms), Python default is dt = 0.02s (20ms)**

### Current Status
- ✅ Time step is read from CSV earthquake files
- ✅ Training CSVs use 0.02s timestep (50 Hz)
- ✅ This is acceptable for Newmark integration (stable)

### Recommendation
- Keep 0.02s for training (faster, stable)
- If needed, can generate 0.01s datasets in future
- Newmark method is unconditionally stable for both

---

## Observation Bounds Impact

The building parameter changes may affect observation bounds:

### Current Bounds (Still Valid)
```python
low=np.array([-1.2, -3.0, -1.2, -3.0, -1.2, -3.0, -1.5, -3.5])
high=np.array([1.2, 3.0, 1.2, 3.0, 1.2, 3.0, 1.5, 3.5])
```

### Expected Behavior with Softer Building
- **Displacements may be larger** (building is 40x softer!)
- Current ±1.2m bounds may be tight for M8.4 earthquakes
- Monitor during training - may need to increase to ±1.5m or ±2.0m

### Action Required
- ⚠️ **Watch training logs** for observation clipping warnings
- If >5% of observations clip, increase bounds
- Re-verify after first training run

---

## Retraining Requirements

### CRITICAL: Old Models are Incompatible
The trained models were optimized for the **old 800 MN/m building**. They will NOT work well on the new 20 MN/m building.

### Why Models Must Be Retrained
1. **Different dynamics**: 40x stiffness change = completely different system
2. **Different displacements**: Old model expects ~30cm, new building may see 50-100cm
3. **Different control strategy**: Old building needed aggressive control, new needs gentler
4. **Observation distribution**: State space statistics completely different

### Action Required
```bash
# 1. MUST delete old models (trained on wrong building)
cd rl/rl_cl
rm -rf rl_cl_robust_models/*

# 2. MUST retrain from scratch
python train_final_robust_rl_cl.py
```

**DO NOT SKIP THIS** - using old models will give terrible results!

---

## Verification

After retraining, verify alignment by comparing Python vs MATLAB:

### Test Scenario: M7.4 PEER Earthquake
```python
# Python (after retraining):
peak_disp: ~50-80cm (expected for 20 MN/m building)
DCR: ~1.5-2.0
```

```matlab
% MATLAB (current):
peak_disp: ~50-80cm (should match Python now!)
DCR: ~1.5-2.0 (should match Python now!)
```

### Success Criteria
✅ Peak displacement within ±20% between MATLAB and Python
✅ DCR values within ±0.3 between platforms
✅ Control force profiles visually similar
✅ No catastrophic failures (>200cm)

---

## Summary Table

| Aspect | Old Status | New Status |
|--------|------------|------------|
| **Building match** | ❌ Different (40x stiffness) | ✅ Identical to MATLAB |
| **Results comparable** | ❌ No | ✅ Yes |
| **Floor mass** | 300,000 kg | 200,000 kg |
| **Stiffness** | 800 MN/m | 20 MN/m |
| **Soft story** | 50% (400 MN/m) | 60% (12 MN/m) |
| **Damping** | 2.0% | 1.5% |
| **Old models valid** | N/A | ❌ No - must retrain |
| **Training time** | N/A | Same (~2-3 hours) |

---

## Combined with Previous Fixes

This building parameter update is **in addition to** the three critical bug fixes:

1. ✅ **Training duration limit** removed (40s → full duration)
2. ✅ **Observation clipping** fixed (4 values → 8 values)
3. ✅ **Baseline drift** corrected (final accel = 0.0)
4. ✅ **Building parameters** aligned (now matches MATLAB)

### All Four Fixes Required
- Fixing bugs alone won't help if building is wrong
- Matching building alone won't help if bugs exist
- **All four must be applied together** for correct results

---

## Date Applied
December 30, 2025

## Author
Siddharth (with Claude)

## Status
✅ **COMPLETE - READY FOR RETRAINING WITH ALIGNED BUILDING**

**Next Step**: Delete old models and retrain from scratch!
