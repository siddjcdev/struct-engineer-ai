# Shaped Reward Implementation - DCR Penalty Removal

## Summary

Removed DCR penalty from shaped reward environment to eliminate conflicting objectives between displacement minimization and drift distribution optimization.

## Problem Identified

The shaped reward function had **contradictory objectives**:

1. **Displacement/Velocity penalties** (-10.0 × |disp|, -3.0 × |vel|):
   - Applied every timestep
   - Told agent: "Minimize ALL displacement everywhere"
   - Cumulative signal strength: ~-3000 per episode

2. **DCR penalty** (-2.0 × (dcr-1)²):
   - Applied once at episode end
   - Told agent: "Allow other floors to drift more to balance concentration"
   - Signal strength: ~-4.5 per episode

**Result**: DCR signal was 666× weaker than displacement signal, creating noise without benefit.

## Changes Made

### 1. tmd_environment_shaped_reward.py (lines 408-426)

**Before:**
```python
# Calculate DCR and apply moderate penalty
if percentile_75 > 0.001:
    current_dcr = max_peak / percentile_75
    dcr_deviation = max(0, current_dcr - 1.0)
    dcr_penalty = -2.0 * (dcr_deviation ** 2)
else:
    dcr_penalty = 0.0

reward = (
    displacement_penalty +
    velocity_penalty +
    force_direction_bonus +
    dcr_penalty  # ← Conflicts with displacement penalty
)
```

**After:**
```python
# Track DCR for metrics only, NOT used in reward
floor_drifts = self._compute_interstory_drifts(self.displacement[:self.n_floors])
self.peak_drift_per_floor = np.maximum(self.peak_drift_per_floor, floor_drifts)

reward = (
    displacement_penalty +      # -10.0 * |disp|
    velocity_penalty +          # -3.0 * |vel|
    force_direction_bonus       # +5.0 or -2.0
    # NO dcr_penalty - let it emerge naturally
)
```

### 2. train_shaped_rewards.py (lines 110-112, 138, 162-169)

Added DCR tracking and analysis to verify hypothesis that good control naturally produces good DCR:

```python
# Get DCR from environment
episode_metrics = test_env.get_episode_metrics()
dcr = episode_metrics.get('dcr', 0.0)

# Report DCR
print(f"   DCR (Drift Concentration Ratio): {dcr:.2f}")

# Analyze DCR hypothesis
if dcr <= 1.5:
    print(f"   ✅ DCR is good ({dcr:.2f} ≤ 1.5) - uniform drift distribution")
    print(f"   → Hypothesis CONFIRMED: Good control naturally produces good DCR")
```

## Hypothesis

**Good vibration control naturally produces good DCR** because:

1. Correct damping (opposing velocity) dissipates energy uniformly
2. Energy doesn't concentrate on a single floor when properly controlled
3. All floors experience proportional damping
4. No conflicting signals to confuse the agent

## Final Reward Structure (v3 - Pure Direction Learning)

**CRITICAL UPDATE**: Removed displacement/velocity penalties - they create noise that drowns out force direction signal!

```python
reward = force_direction_bonus  # ONLY reward signal!

where:
  # Due to Newton's 3rd law: F_eq[roof] -= control_force
  # Same signs = correct damping!
  if (velocity > 0 and force > 0) or (velocity < 0 and force < 0):
      force_direction_bonus = +5.0 * |force| / max_force  # Correct!
  else:
      force_direction_bonus = -2.0 * |force| / max_force  # Wrong!
```

**Why this works**:
- Agent learns the **action** (force direction), not the **outcome** (displacement)
- No noise from random displacement changes during exploration
- Clear, immediate feedback on every step
- Once correct direction is learned, displacement reduction emerges naturally

**Previous attempts failed** because:
1. v1: Displacement penalties (10x stronger) drowned out direction bonus
2. v2: Wrong force direction (opposite signs) - taught agent to amplify vibrations
3. v3: Pure direction learning - clean signal, correct physics

## Expected Outcomes

1. **Peak displacement**: <19 cm on M4.5 (15-28% improvement vs 21.02 cm uncontrolled)
2. **Force direction correctness**: >70% (agent learns correct physics)
3. **DCR**: ≤1.5 (natural consequence of good control)
4. **Faster learning**: Clearer reward signal, no conflicting objectives

## Testing

Run training:
```bash
cd rl/rl_cl
python train_shaped_rewards.py
```

Expected training time: ~30-60 minutes for 200K timesteps on CPU.

## Rollback Plan

If DCR remains high (>2.0) after training, we can:
1. Re-enable DCR penalty but reduce displacement/velocity weights
2. Adjust force direction bonus to be less dominant
3. Consider multi-objective reward with balanced weights

---

**Created**: 2025-12-31
**Hypothesis**: Good control → good DCR naturally
**Status**: Ready for testing
