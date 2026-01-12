# V13 Critical Reward Function Bug Fix

## Date
2026-01-12

## The Problem

V13 trained model produced **catastrophically bad results**:
- **Displacement: 208.69 cm** (vs 14 cm target, vs 17 cm uncontrolled)
- **ISDR: 9.718%** (vs 0.4% target, vs 0.97% uncontrolled)
- **Performance: 10× WORSE than doing nothing!**

The trained agent learned to **amplify the building's motion** instead of controlling it.

## Root Cause Analysis

### The Bug

The original reward function used **only negative penalties**:

```python
# OLD (BROKEN):
P_disp = -((roof_disp / d_roof_target) ** 2)      # Always ≤ 0
P_isdr = -((max_isdr / ISDR_target) ** 2)          # Always ≤ 0
P_dcr = -((DCR / DCR_target) ** 2)                 # Always ≤ 0
P_force = -(force_utilization ** 2)                # Always ≤ 0

r_t = w_disp * P_disp + w_DCR * P_dcr + w_ISDR * P_isdr + w_force * P_force
# Maximum possible reward = 0 (when everything is perfect)
# Typical reward = -2 to -20 (always negative)
```

### Why This Broke Training

At the start of each episode when everything is near zero:
- `P_disp ≈ 0` (no displacement yet)
- `P_isdr ≈ 0` (no drift yet)
- `P_force ≈ 0` (no force applied yet)
- `P_dcr = -(1.0/1.15)² = -0.756` (baseline DCR penalty even when DCR is perfect!)

**Baseline reward = 3.0 × (-0.756) = -2.268**

This created a **perverse incentive**:

1. Agent starts with baseline reward of -2.268 per step
2. Applying any control force makes reward MORE negative (adds -0.3 × P_force)
3. The agent learns: "I can't get positive reward, so minimize cumulative negative reward"
4. **Solution the agent found: End the episode as fast as possible!**
5. To end episodes faster, agent amplifies building motion to trigger large displacements
6. Result: Agent learned to **destroy** the building instead of control it

### Mathematical Proof

At zero state with zero control:
```
P_disp = -(0/0.14)² = 0
P_isdr = -(0/0.004)² = 0
P_dcr = -(1.0/1.15)² = -0.756
P_force = -(0/300000)² = 0

r_t = 3.0×0 + 3.0×(-0.756) + 5.0×0 + 0.3×0 = -2.268
```

**This matches the -2.268 reward we observed in diagnostics!**

## The Fix

Restructured reward function to provide **positive rewards for good performance**:

```python
# NEW (FIXED):
R_disp = 1.0 - (disp_ratio ** 2)           # Reward for small displacement
R_isdr = 1.0 - (isdr_ratio ** 2)           # Reward for small drift
R_dcr = 1.0 - dcr_deviation                # Reward for uniform drift
P_force = -(force_utilization ** 2)        # Penalty for force (unchanged)

r_t = w_disp * R_disp + w_DCR * R_dcr + w_ISDR * R_isdr + w_force * P_force
# Baseline reward (perfect control): ~9.5
# Typical reward (good control): ~5 to 8
# Typical reward (poor control): -5 to -15
```

### Key Changes

1. **Positive baseline**: Perfect control now gives ~9.5 reward instead of 0
2. **Rewards vs penalties**: Three components now reward good performance
3. **Proper incentives**: Agent is motivated to maximize reward, not minimize negative reward
4. **Force still penalized**: Efficiency incentive maintained

### Verification

Test results confirm fix:

```
Zero state:             reward = 9.739  ✓ (positive baseline)
Good control (7cm):     reward = 8.941  ✓ (positive reward)
Poor control (28cm):    reward = -4.705 ✓ (negative penalty)
Force gradient:         0kN > 50kN > 300kN  ✓ (force penalty works)
```

## Weight Adjustments

Updated weights for better balance:

```python
# OLD:
w_disp = 3.0
w_DCR = 3.0
w_ISDR = 5.0
w_force = 0.3

# NEW:
w_disp = 3.0      # Unchanged
w_DCR = 2.0       # REDUCED (less critical than ISDR)
w_ISDR = 5.0      # Unchanged (highest - safety critical)
w_force = 0.5     # INCREASED (encourage efficiency)
```

## Impact

### Before Fix (Catastrophic Failure)
- Baseline reward: -2.268 (always negative)
- Agent learned to amplify motion to end episodes faster
- Results: 208cm displacement, 9.7% ISDR (10× worse than uncontrolled)

### After Fix (Expected Performance)
- Baseline reward: +9.739 (positive)
- Agent motivated to maximize reward by good control
- Expected results: ~10cm displacement, ~0.5% ISDR (2× better than uncontrolled)

## Lessons Learned

1. **Never use all-negative reward functions** - Always provide positive baseline for good behavior
2. **Test reward function in isolation** - Verify it produces expected values before training
3. **Check baseline reward** - At zero state, reward should be positive or neutral
4. **Perverse incentives are subtle** - Agent will exploit any loophole to maximize cumulative reward
5. **Reward shaping matters more than hyperparameters** - Wrong incentive structure can't be fixed by tuning learning rates

## Next Steps

1. **Re-train V13 with fixed environment**
2. **Test on held-out dataset**
3. **Verify results match expected performance**
4. **Document training metrics and convergence behavior**

## Files Modified

- [tmd_environment_v13_rooftop.py](restapi/rl_cl/tmd_environment_v13_rooftop.py) - Fixed reward function (lines 439-520)
- [test_v13_reward.py](rl/rl_cl/test_v13_reward.py) - Verification test script

## Technical Details

### Reward Function Comparison

| Component | OLD (Broken) | NEW (Fixed) | Change |
|-----------|--------------|-------------|--------|
| Displacement | `-(d/target)²` | `1 - (d/target)²` | Reward for small values |
| ISDR | `-(ISDR/target)²` | `1 - (ISDR/target)²` | Reward for small values |
| DCR | `-(DCR/target)²` | `1 - \|DCR-target\|/target` | Reward for target proximity |
| Force | `-(f/fmax)²` | `-(f/fmax)²` | Unchanged (penalty) |
| **Baseline** | **-2.268** | **+9.739** | **+12.0 shift** |

### Expected Reward Ranges

| Scenario | OLD | NEW | Notes |
|----------|-----|-----|-------|
| Perfect zero state | -2.268 | +9.739 | Starting condition |
| Good control (7cm, 0.3% ISDR) | -1.5 to -3.0 | +7.0 to +9.0 | Desired behavior |
| Target performance (14cm, 0.4% ISDR) | -3.0 to -5.0 | +5.0 to +7.0 | Acceptable |
| Poor control (28cm, 1% ISDR) | -10.0 to -15.0 | -5.0 to -10.0 | Discouraged |
| Catastrophic (>50cm, >2% ISDR) | -20.0 to -50.0 | -15.0 to -30.0 | Strongly discouraged |

## Verification Protocol

Before training any new model, always:

1. Run `python test_v13_reward.py` to verify reward function
2. Check baseline reward at zero state (should be positive)
3. Verify force gradient (more force = lower reward)
4. Confirm reward decreases with worse performance
5. Test edge cases (large displacement, large ISDR)
