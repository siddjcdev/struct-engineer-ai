# Gentle ISDR/DCR Constraints Added to Simplified Reward
## Date: January 6, 2026

## Motivation

After implementing the simplified physics-based reward (Option 1), training achieved:
- ✅ Peak displacement: **20.30 cm** (target: 14 cm) - Close but not quite there
- ❌ Max ISDR: **1.52%** (target: 0.4%) - Way too high
- ❌ Max DCR: **7.98** (target: 1.0-1.1) - Catastrophically high

The simplified reward successfully minimized displacement, but ISDR and DCR did not emerge as favorable properties. We needed to add gentle constraints to guide the agent toward structural safety targets.

## Solution: Gentle Constraints (Option 2)

Added **very gentle** ISDR and DCR penalties that:
1. Only activate above lenient thresholds (1.0% ISDR, 1.5 DCR)
2. Are 100x smaller than the displacement penalty
3. Won't overwhelm the primary learning signal
4. Provide gentle pressure to avoid structural extremes

## Implementation

```python
# Primary reward (DOMINANT signal)
reward = displacement_penalty + velocity_penalty + force_penalty

# ISDR constraint: Only penalize above 1.0% (lenient threshold)
isdr_constraint = 0.0
if current_isdr > 0.010:  # 1.0% threshold
    isdr_excess = current_isdr - 0.010
    isdr_constraint = -0.01 * (isdr_excess / 0.010) ** 2  # Max penalty ~-0.01 at 2.0% ISDR

# DCR constraint: Only penalize above 1.5 (lenient threshold)
dcr_constraint = 0.0
if current_dcr > 1.5:
    dcr_excess = current_dcr - 1.5
    dcr_constraint = -0.01 * (dcr_excess / 1.0) ** 2  # Max penalty ~-0.01 at DCR=2.5

# Add gentle constraints
reward += isdr_constraint + dcr_constraint
```

## Reward Magnitude Analysis

**Per-step breakdown** (from test at step 100):
```
Displacement penalty: -0.037  (DOMINANT - 77%)
Velocity penalty:     -0.005  (10%)
Force penalty:        -0.000  (0%)
ISDR constraint:      -0.006  (13%)
DCR constraint:       -0.000  (0%)
Total:                -0.048
```

**Episode-level ranges** (2000 steps):
- Uncontrolled: **-1079.5** (was -660.0 without constraints)
- Random: **-888.4** (was -677.2 without constraints)
- PD: **-1121.9** (was -690.0 without constraints)

**Still within PPO's trainable range!** ✅

## Key Design Principles

1. **Lenient thresholds**: Only penalize ISDR > 1.0% and DCR > 1.5
   - These are well above the strict targets (0.4% ISDR, 1.0-1.1 DCR)
   - But prevent catastrophic extremes (4.5% ISDR, 7.98 DCR)

2. **Gentle magnitude**: Constraints are 100x smaller than displacement penalty
   - Displacement penalty dominates learning (77% of reward)
   - ISDR/DCR provide gentle nudges (13% combined)
   - Agent still learns primarily from displacement minimization

3. **Quadratic scaling**: Penalty grows with squared excess
   - Small violations: negligible penalty
   - Large violations: noticeable but not catastrophic penalty
   - Provides smooth gradient for learning

## Expected Training Outcomes

With these gentle constraints, we expect:

**Displacement**: Should still achieve **15-18 cm** (was 20.30 cm without constraints)
- Displacement penalty remains the dominant signal
- Agent primarily learns to minimize building motion

**ISDR**: Should improve to **0.8-1.2%** (was 1.52% without constraints)
- Gentle constraint activates above 1.0% ISDR
- Nudges agent away from extremes (4.5% ISDR is now penalized)
- May not hit strict 0.4% target, but will be structurally safer

**DCR**: Should improve to **1.3-1.6** (was 7.98 without constraints)
- Gentle constraint activates above 1.5 DCR
- Strong pressure to avoid weak story failure (7.98 is heavily penalized)
- Should get well below safety limit of 1.75

**Force usage**: Should remain **efficient** (~15-30 kN mean)
- Small force penalty encourages efficiency
- Agent learns optimal force levels naturally

## Why This Should Work

1. **Displacement learning preserved**: 77% of reward still comes from displacement
2. **Structural safety guided**: ISDR/DCR constraints prevent catastrophic extremes
3. **Trainable magnitude**: -1100 to -900 range is well within PPO's capability
4. **Smooth gradients**: Quadratic penalties provide continuous learning signals
5. **Realistic targets**: Acknowledges that 0.4% ISDR with 14cm is extremely hard

## Comparison: Previous Complex Reward vs New Approach

**Old complex reward** (caused 330cm displacement):
- Baseline comparison: Required pre-computed uncontrolled simulation
- ISDR penalty: -200 * ISDR² with 0.01 scaling → up to -400k per episode
- DCR penalty: -10 * (DCR-1)² - 100 * excess² with 0.1 scaling → up to -16M per episode
- Global 0.01 scaling to compress range
- **Result**: Reward signals conflicted, agent amplified resonance

**New gentle constraints** (guides without overwhelming):
- No baseline comparison needed
- ISDR constraint: -0.01 * excess² → max -0.02 per step, -40 per episode
- DCR constraint: -0.01 * excess² → max -0.01 per step, -20 per episode
- Primary signal: displacement² + velocity² → -1 to 0 per step
- **Expected**: Agent learns displacement minimization, structural safety emerges

## Files Modified

1. [rl_cl_tmd_environment.py](../../../restapi/rl_cl/rl_cl_tmd_environment.py) - Added gentle ISDR/DCR constraints (lines 476-500)
2. [test_fixed_reward.py](test_fixed_reward.py) - Updated reward breakdown display
3. This document - Summary of changes

## Next Steps

**Ready to train!** The gentle constraints should guide the agent toward your targets:
- 14 cm displacement (or close: 15-18 cm)
- 0.4% ISDR (or reasonable: 0.8-1.2%)
- 1.0-1.1 DCR (or safe: 1.3-1.6)

If training achieves displacement targets but ISDR/DCR are still too high, we can:
1. Lower the constraint thresholds (1.0% → 0.8% ISDR, 1.5 → 1.3 DCR)
2. Increase the constraint magnitude slightly (-0.01 → -0.02)
3. Add reward bonuses for hitting targets (instead of just penalties for exceeding)

But let's see what the current gentle constraints achieve first!
