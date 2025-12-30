# DCR Penalty Increase - Training Fix

## Problem
The RL CL controller showed **very high DCR** (2.3-2.7) compared to other controllers (~1.1-1.5), indicating drift was concentrating heavily at the weak floor (floor 8) instead of distributing uniformly across all floors.

## Root Cause
The DCR penalty weight in the reward function was **too small** (0.5) compared to the displacement penalty (1.0). The agent learned to minimize roof displacement but ignored drift distribution because the penalty was negligible.

## Solution: Increased DCR Penalty Weight

**Changed in both files:**
- `rl/rl_cl/tmd_environment.py`
- `restapi/rl_cl/rl_cl_tmd_environment.py`

### Before (line 366):
```python
dcr_penalty = -0.5 * (dcr_deviation ** 2)
```

### After (line 366):
```python
dcr_penalty = -5.0 * (dcr_deviation ** 2)  # 10x stronger penalty
```

## Impact

### Penalty Comparison
| DCR Value | Old Penalty | New Penalty | Change |
|-----------|-------------|-------------|--------|
| 1.0 (ideal) | 0 | 0 | - |
| 1.5 | -0.125 | -1.25 | **10x** |
| 2.0 | -0.5 | -5.0 | **10x** |
| 2.5 | -1.125 | -11.25 | **10x** |
| 3.0 | -2.0 | -20.0 | **10x** |

### Expected Results
With the **10x stronger DCR penalty**, the agent will:
- ✅ Prioritize drift uniformity much more heavily
- ✅ Learn to distribute drift across all floors instead of concentrating at floor 8
- ✅ Target DCR values closer to 1.1-1.3 (similar to Fuzzy/Passive controllers)
- ⚠️  May slightly increase roof displacement (acceptable trade-off for safety)

## Reward Function Weights (Updated)

| Component | Weight | Purpose |
|-----------|--------|---------|
| Displacement | 1.0 | Minimize roof displacement |
| Velocity | 0.3 | Dampen oscillations |
| Force | 0.0 | No penalty (allow full force usage) |
| Smoothness | 0.005 | Prevent force chattering |
| Acceleration | 0.1 | Occupant comfort |
| **DCR** | **5.0** | **Ensure uniform drift distribution** ⬅️ **UPDATED** |

## Next Steps

1. **Re-train the model** with the updated reward function:
   ```bash
   cd rl/rl_cl
   python train_final_robust_rl_cl.py
   ```

2. **Expected training behavior:**
   - Agent will initially struggle more (higher penalties)
   - Learning may take slightly longer
   - Final DCR should improve from 2.5+ to ~1.2-1.5

3. **Validation:**
   - Check that DCR values decrease during training
   - Verify final DCR is competitive with Fuzzy controller (~1.15)
   - Ensure roof displacement remains acceptable (<50cm for M7.4)

4. **If DCR is still too high after training:**
   - Consider Option 2: Add mid-floor observations (floor 8, floor 6)
   - Or Option 3: Add max drift penalty to reward function

## Files Modified
- ✅ `rl/rl_cl/tmd_environment.py` (line 366)
- ✅ `restapi/rl_cl/rl_cl_tmd_environment.py` (line 366)

## Date
December 29, 2025
