# CRITICAL SAC MODEL FIXES - JANUARY 4, 2026

## Problem Summary
The SAC model showed **catastrophic failures** on extreme earthquakes:
- **PEER_High (M7.4):** 827 cm peak displacement (-382% vs passive!)
- **PEER_Insane (M8.4):** 544 cm peak displacement (-217% vs passive!)
- **Latency test:** "UNSAFE" - baseline completely unstable under 60ms latency

## Root Causes Identified

### 1. **OBSERVATION BOUNDS MISMATCH** (Critical)
**Problem:** 
- RL Controller clipped observations to ±0.5m displacement
- M7.4 earthquakes reach peak displacements of 8.9m
- ALL observations were clipped → model received useless data → random/bad actions

**Impact:**
- Model couldn't see actual system state
- Prediction quality degraded to random noise
- Led to diverging displacements

**Fix Applied:**
```python
# OLD (WRONG):
obs_bounds = {
    'roof_disp': (-0.5, 0.5),    # ±50 cm - TOO SMALL!
    'roof_vel': (-2.0, 2.0),     # ±2 m/s
    'tmd_disp': (-0.6, 0.6),     # ±60 cm
    'tmd_vel': (-2.5, 2.5)       # ±2.5 m/s
}

# NEW (CORRECT):
obs_bounds = {
    'roof_disp': (-5.0, 5.0),    # ±5.0 m (matches training!)
    'roof_vel': (-20.0, 20.0),   # ±20.0 m/s (matches training!)
    'tmd_disp': (-15.0, 15.0),   # ±15.0 m (matches training!)
    'tmd_vel': (-60.0, 60.0)     # ±60.0 m/s (matches training!)
}
```

### 2. **FORCE LIMIT MISMATCH** (Critical)
**Problem:**
- RL Controller set max force to 100 kN
- Training curriculum stages use: 50kN (M4.5) → 100kN (M5.7) → **150kN (M7.4/M8.4)**
- Model trained to output up to 150kN but clamped to 100kN
- Insufficient control authority on extreme earthquakes → divergence

**Impact:**
- Model couldn't apply full trained control
- Forces saturated at 100kN when 150kN+ needed
- Uncontrolled displacement growth

**Fix Applied:**
```python
# OLD (INSUFFICIENT):
self.max_force = 100000.0  # 100 kN

# NEW (MATCHES TRAINING):
self.max_force = 150000.0  # 150 kN (final curriculum stage)
```

### 3. **LATENCY-INDUCED INSTABILITY** (Critical)
**Problem:**
- 60ms latency = 3 timesteps (20ms each) of delay
- Without rate limiting: commands jump ±150kN → jerky motion
- Structure overshoots → control oscillates → divergence
- Explains "UNSAFE under latency" observation

**Impact:**
- Model couldn't handle real-world 60ms latency
- Latency test showed complete failure ("UNSAFE")

**Fix Applied:**
Force rate limiting - max 50kN change per timestep:
```python
# CRITICAL: Force rate limiting for latency robustness
max_rate = 50000.0  # Maximum force change per timestep (N)
delta = force_N - self._last_force
if abs(delta) > max_rate:
    force_N = self._last_force + np.sign(delta) * max_rate

self._last_force = force_N
```

This allows:
- Smooth transitions even with latency
- Prevents overshoot from stale decisions
- Maintains control authority (can go 0→50kN in 1 timestep, 0→150kN in 3 timesteps)

## Files Modified

### 1. `restapi/rl_baseline/rl_controller.py`
- **Line 40-51:** Fixed obs_bounds (±0.5m → ±5.0m)
- **Line 33:** Fixed max_force (100kN → 150kN)
- **Line 98-115:** Added force rate limiting

### 2. `restapi/rl_cl/RLCLController.py`
- **Line 54-64:** Fixed 8-value obs_bounds
- **Line 70-73:** Fixed legacy 4-value obs_bounds
- **Line 75:** Added `_last_force` initialization
- **Line 78-107:** Added force rate limiting to predict_single()
- **Line 109-140:** Added force rate limiting to predict_batch()

## Expected Results After Fix

### M4.5 (Small)
- **Before:** 0.91 cm ✅ (was already working)
- **After:** ~0.91 cm ✅ (unchanged)

### M5.7 (Moderate)
- **Before:** 6.45 cm ✅ (was working)
- **After:** ~6.45 cm ✅ (unchanged)

### M7.4 (High) - **CRITICAL IMPROVEMENT**
- **Before:** 827 cm ❌ (catastrophic!)
- **After:** ~35-45 cm ✅ (85-95% reduction expected)

### M8.4 (Insane) - **CRITICAL IMPROVEMENT**
- **Before:** 544 cm ❌ (catastrophic!)
- **After:** ~45-55 cm ✅ (85-92% reduction expected)

### Latency Test - **CRITICAL IMPROVEMENT**
- **Before:** "UNSAFE" ❌ (complete failure)
- **After:** Robust ✅ (force rate limiting prevents instability)

## Verification Steps

1. **Quick test:** 
   ```bash
   cd restapi
   python -m pytest test_api.py::test_rl_extreme
   ```

2. **Full comparison:**
   ```bash
   cd matlab
   python final_exhaustive_check.py
   ```

3. **Check specific scenarios:**
   - PEER_High (M7.4) should be <50 cm
   - PEER_Insane (M8.4) should be <55 cm
   - Latency test should pass without "UNSAFE" warning

## Why These Fixes Work

### Observation Bounds Fix
- **Principle:** Models must see their entire training distribution at inference
- **Training:** Environment uses ±5.0m / ±20.0m/s bounds
- **Deployment:** Must use same bounds or model is out-of-distribution
- **Result:** Model can see extreme states → make informed decisions

### Force Limit Fix
- **Principle:** Action space must match training exactly
- **Training:** Final stages trained with ±150kN action space
- **Deployment:** Limiting to ±100kN removes 33% of model's learned capability
- **Result:** Model has full trained control authority

### Rate Limiting Fix
- **Principle:** Smoothing stale decisions from latency
- **Physical:** Structure inertia prevents instant force changes anyway
- **Benefits:** 
  - Prevents overshoot from old decisions
  - Allows control with latency (50kN/step = full range in 3 steps)
  - Matches real actuator behavior

## Next Steps

1. ✅ **Apply fixes** (DONE)
2. **Test on extreme earthquakes** (TODO)
3. **Monitor latency robustness** (TODO)
4. **Retrain if necessary** (Only if still unstable)

---

**Status:** FIXES APPLIED - AWAITING TEST RESULTS
**Date:** January 4, 2026
**Author:** AI Assistant
