# Hybrid Model Creation - Status Report

## Summary

Created hybrid model infrastructure to combine two RL models with complementary strengths:
- **Model 1**: `rl_cl_final_robust.zip` (good peak displacement reduction)
- **Model 2**: `rl_cl_dcr_train_5_final.zip` (good DCR performance)

## Problem Identified

**CRITICAL**: All existing models are incompatible with current environment due to observation space mismatch.

### Observation Space Evolution

| Version | Displacement Bounds | Velocity Bounds | Dimensions | Status |
|---------|-------------------|-----------------|------------|---------|
| Original (models trained) | ±0.5m | ±2.0 m/s | 4D: [roof, tmd] | Used in all existing .zip models |
| Current environment | ±5.0m | ±20.0 m/s | 8D: [roof, floor8, floor6, tmd] | Active in codebase |

### Why This Causes Failure

1. **Dimension mismatch**: Models expect 4D observations, environment provides 8D
2. **Bound mismatch**: Models trained on ±0.5m, but M7.4 produces 59cm+ displacement
3. **Out-of-distribution**: Values beyond training range cause catastrophic failure

### Test Results on M7.4 Earthquake

With current (incompatible) models:

| Strategy | Peak Displacement | DCR | Notes |
|----------|------------------|-----|-------|
| Adaptive | 591.23 cm | 1.1451 | Hybrid: gradual transition |
| Weighted Avg | 595.03 cm | 1.1317 | Hybrid: 50-50 blend |
| Max Response | 627.17 cm | 1.5215 | Hybrid: strongest action |
| Model 1 Only | **586.48 cm** | 1.1613 | **Best of all** |
| Model 2 Only | 608.00 cm | 1.2711 | Worse than Model 1 |
| **Expected (with correct models)** | **~40-60 cm** | **~1.2-1.4** | **After retraining** |

**Analysis**: All models fail catastrophically (586-627cm vs target 40-60cm) due to observation space incompatibility.

---

## Solution: Retrain Models First

Before hybrid model can be useful, models must be retrained with:

### Option A: Match Current Environment (8D + Large Bounds)
```python
observation_space = 8D: [roof_disp, roof_vel, floor8_disp, floor8_vel, floor6_disp, floor6_vel, tmd_disp, tmd_vel]
bounds = {
    'disp': 5.0,      # ±5.0m displacement
    'vel': 20.0,      # ±20.0 m/s velocity
    'tmd_disp': 15.0, # ±15.0m TMD displacement
    'tmd_vel': 60.0   # ±60.0 m/s TMD velocity
}
```

**Pros:**
- Additional floor observations (floor 8, floor 6) provide better DCR control
- Very large bounds handle any earthquake (M4.5 to M8.4+)

**Cons:**
- More complex observation space may be harder to train
- Larger bounds can slow convergence

### Option B: Simplified 4D with Correct Bounds
```python
observation_space = 4D: [roof_disp, roof_vel, tmd_disp, tmd_vel]
bounds = {
    'disp': 1.2,      # ±1.2m displacement (2× M7.4 uncontrolled)
    'vel': 3.0,       # ±3.0 m/s velocity
    'tmd_disp': 1.5,  # ±1.5m TMD displacement
    'tmd_vel': 3.5    # ±3.5 m/s TMD velocity
}
```

**Pros:**
- Simpler observation space, easier to train
- Bounds sized appropriately for realistic earthquake range (M4.5-M7.4)

**Cons:**
- No direct observation of critical floors (floor 8 weak story)
- May have slightly worse DCR control

---

## Hybrid Model Infrastructure Created

### Files Created:

**`restapi/rl_cl/create_hybrid_model.py`**
- Main script for creating and testing hybrid models
- Implements 3 combination strategies:
  - **Adaptive**: Weights shift from model1 (early) → model2 (late) based on episode progress
  - **Weighted Average**: Simple 50-50 blend of both models
  - **Max Response**: Uses whichever model suggests stronger action

**Key Components:**

```python
class HybridRLModel:
    """Combines two RL models with complementary strengths"""

    def predict(self, observation, deterministic=True):
        # Get actions from both models
        action1, _ = self.model1.predict(observation, deterministic)
        action2, _ = self.model2.predict(observation, deterministic)

        if self.strategy == 'adaptive':
            # Transition from model1 → model2 during episode
            progress = self.step_count / self.max_steps
            weight1 = 0.8 * (1 - progress) + 0.2  # 80% → 20%
            weight2 = 1 - weight1                  # 20% → 80%
            action = weight1 * action1 + weight2 * action2

        return np.clip(action, -1.0, 1.0), None
```

**`restapi/rl_cl/test_model_obs_space.py`**
- Diagnostic script to check model observation space requirements

**`restapi/models/hybrid_rl_model_best.py`**
- Generated wrapper (currently uses model1_only since it performed best in tests)

---

## Next Steps

### Immediate Actions Required:

1. **Decide on observation space approach** (Option A or B above)

2. **Update environment to match chosen approach**
   - If Option A: Keep current 8D large bounds
   - If Option B: Revert to 4D with ±1.2m bounds

3. **Retrain both models with consistent observation space:**
   ```bash
   cd rl/rl_cl
   python train_final_robust_rl_cl.py  # For peak displacement model
   python train_dcr_focused.py         # For DCR optimization model
   ```

4. **Update hybrid model paths** in `create_hybrid_model.py`:
   ```python
   MODEL1_PATH = Path("../models/NEW_peak_model.zip")
   MODEL2_PATH = Path("../models/NEW_dcr_model.zip")
   ```

5. **Re-run hybrid model comparison:**
   ```bash
   cd restapi/rl_cl
   python create_hybrid_model.py
   ```

6. **Expected results after retraining:**
   - Peak displacement: 40-60 cm (vs current 586+ cm)
   - DCR: 1.2-1.4 (reasonable given soft story)
   - Hybrid may achieve best of both: ~40cm peak + ~1.2 DCR

---

## Technical Details

### Adaptive Weighting Strategy

The adaptive strategy uses time-based weighting:

```python
# Early in earthquake (0-30%):
#   - High displacement expected
#   - Weight 80% towards model1 (peak displacement control)
#   - Weight 20% towards model2

# Mid earthquake (30-70%):
#   - Gradual transition as drift accumulates
#   - Weights shift smoothly

# Late earthquake (70-100%):
#   - DCR becomes critical (drift distribution)
#   - Weight 20% towards model1
#   - Weight 80% towards model2 (DCR control)

progress = step_count / max_steps
weight1 = 0.8 * (1 - progress) + 0.2
weight2 = 1 - weight1
```

### 4D Wrapper Implementation

Created `Simple4DWrapper` to make 8D environment compatible with 4D models:

```python
class Simple4DWrapper(gym.Wrapper):
    """Extract only [roof, tmd] from [roof, floor8, floor6, tmd]"""

    def step(self, action):
        obs_8d, reward, term, trunc, info = self.env.step(action)
        # Extract indices: 0-1 (roof) + 6-7 (tmd)
        obs_4d = np.concatenate([obs_8d[0:2], obs_8d[6:8]])
        return obs_4d, reward, term, trunc, info
```

This allowed testing with current models, but doesn't solve the fundamental observation bounds mismatch.

---

## Recommendations

**Short term:**
1. Revert to 4D observation space (simpler, proven to work)
2. Use ±1.2m bounds (per OBSERVATION_SPACE_FIX.md recommendations)
3. Retrain one model focused on peak displacement reduction
4. Retrain another model focused on DCR optimization
5. Use hybrid model to combine their strengths

**Long term:**
1. Consider 8D observation space once 4D approach is validated
2. Implement ensemble methods beyond simple weighted averaging
3. Add reinforcement learning from human feedback (RLHF) for fine-tuning

---

## Date: December 31, 2025

**Status:** Infrastructure complete, awaiting model retraining with correct observation space

**Priority:** HIGH - Current models fail catastrophically on M7.4+ earthquakes

**Action Required:** User must decide on observation space approach and retrain models before hybrid model can be effective
