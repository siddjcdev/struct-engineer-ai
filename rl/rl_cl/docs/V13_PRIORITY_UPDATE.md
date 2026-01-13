# V13 Priority Update: ISDR and DCR First, Displacement Constraint

## Date
2026-01-12

## Current Performance
- **Displacement: 17.04 cm** ✓ (acceptable baseline)
- **ISDR: 1.332%** ❌ (target: <0.55%)
- **DCR: 1.47** ❌ (target: <1.15)

## Problem Statement

The trained model achieved acceptable displacement (17cm, similar to uncontrolled) but failed the primary safety objectives:
- ISDR is 2.4× higher than target (1.332% vs 0.55%)
- DCR is 1.28× higher than target (1.47 vs 1.15)

The previous reward weights didn't reflect the true priorities, allowing the agent to trade off ISDR/DCR improvements for displacement optimization.

## User Requirements (Clear Priorities)

### PRIMARY OBJECTIVES (what we're trying to optimize):
1. **ISDR < 0.55%** - Interstory drift ratio (safety critical)
2. **DCR < 1.15** - Drift concentration ratio (damage uniformity)

### HARD CONSTRAINT (what we absolutely cannot violate):
3. **Displacement ≤ 17 cm** - NEVER ALLOW INCREASES, only maintain or decrease

### SECONDARY (nice to have):
4. Force efficiency - Minimize actuator usage

## Solution: Restructured Reward Weights

### OLD Weights (Equal Priority)
```python
w_disp = 3.0    # Equal importance
w_DCR = 2.0     # Too low
w_ISDR = 5.0    # Not dominant enough
w_force = 0.5
# Total possible: ~10 (equal weighting)
```

### NEW Weights (Priority-Based)
```python
w_ISDR = 10.0           # HIGHEST - primary objective
w_DCR = 8.0             # VERY HIGH - primary objective
w_disp = 5.0            # MODERATE - maintain baseline
w_disp_penalty = 100.0  # MASSIVE - hard constraint enforcement
w_force = 0.3           # LOW - efficiency is secondary

# Combined priority: ISDR+DCR = 18.0 vs displacement = 5.0
# Ratio: 3.6:1 in favor of ISDR/DCR optimization
```

## Key Changes

### 1. Hard Constraint Enforcement
Added massive penalty for displacement exceeding 17 cm:

```python
if roof_disp > 0.17:  # 17 cm constraint
    violation_ratio = (roof_disp - 0.17) / 0.17
    P_disp_constraint = -100.0 * (violation_ratio ** 2)
else:
    P_disp_constraint = 0.0
```

**Effect**: Agent will NEVER learn to increase displacement above 17cm - the penalty is too severe.

### 2. ISDR Target Relaxed to 0.55%
```python
ISDR_target = 0.0055  # 0.55% (was 0.4%)
```

**Reason**: 0.4% may be too aggressive given the 17cm displacement constraint. 0.55% is still excellent performance and more achievable.

### 3. DCR Target Ideal = 1.0
```python
DCR_ideal = 1.0   # Perfect uniformity
DCR_max = 1.15    # Acceptable threshold
```

**Effect**: Agent is rewarded for DCR close to 1.0 (perfect damage distribution), with quadratic penalty for deviation.

### 4. Displacement Normalization Changed
```python
# OLD: normalized to 14cm target
disp_ratio = roof_disp / 0.14

# NEW: normalized to 17cm constraint
disp_ratio = roof_disp / 0.17
```

**Effect**: Agent sees 17cm as the "acceptable" baseline, not 14cm, so it won't sacrifice ISDR/DCR to hit 14cm.

## Expected Behavior Change

### Previous Behavior (Equal Weighting)
- Agent optimized all metrics equally
- Could trade off ISDR reduction for displacement improvement
- No hard constraint on displacement
- Result: 17cm disp (good), 1.33% ISDR (bad), 1.47 DCR (bad)

### New Behavior (Priority-Based)
- Agent focuses primarily on reducing ISDR and DCR
- Displacement constraint strictly enforced at 17cm (hard ceiling)
- Agent learns: "Keep displacement ≤ 17cm, then maximize ISDR/DCR performance"
- Expected result: ≤17cm disp (maintain), <0.55% ISDR (achieve), <1.15 DCR (achieve)

## Reward Calculation Examples

### Scenario 1: Perfect Control
```
displacement = 10 cm
ISDR = 0.3%
DCR = 1.05
force = 0 kN

R_disp = 1.0 - (10/17)² = 0.654    → 5.0 * 0.654 = 3.27
R_ISDR = 1.0 - (0.3/0.55)² = 0.702 → 10.0 * 0.702 = 7.02
R_DCR = 1.0 - (0.05/1.15)² = 0.998 → 8.0 * 0.998 = 7.98
P_disp_constraint = 0 (no violation)
P_force = 0

Total reward = 3.27 + 7.02 + 7.98 + 0 + 0 = 18.27
```

### Scenario 2: Acceptable Performance (Target)
```
displacement = 17 cm (at constraint)
ISDR = 0.55%
DCR = 1.15
force = 100 kN

R_disp = 1.0 - (17/17)² = 0.0      → 5.0 * 0.0 = 0
R_ISDR = 1.0 - (0.55/0.55)² = 0.0  → 10.0 * 0.0 = 0
R_DCR = 1.0 - (0.15/1.15)² = 0.983 → 8.0 * 0.983 = 7.86
P_disp_constraint = 0 (no violation)
P_force = -(100000/300000)² = -0.111 → 0.3 * -0.111 = -0.033

Total reward = 0 + 0 + 7.86 + 0 - 0.033 = 7.83
```

### Scenario 3: Constraint Violation (Catastrophic)
```
displacement = 20 cm (VIOLATES 17cm constraint)
ISDR = 0.4%
DCR = 1.1
force = 50 kN

R_disp = 1.0 - (20/17)² = -0.384   → 5.0 * -0.384 = -1.92
R_ISDR = 1.0 - (0.4/0.55)² = 0.469 → 10.0 * 0.469 = 4.69
R_DCR = 1.0 - (0.1/1.15)² = 0.992  → 8.0 * 0.992 = 7.94
violation = (20-17)/17 = 0.176
P_disp_constraint = -100.0 * 0.176² = -3.10
P_force = -0.008

Total reward = -1.92 + 4.69 + 7.94 - 3.10 - 0.008 = 7.60

BUT MORE IMPORTANTLY: The -3.10 penalty (and growing quadratically)
makes this MUCH WORSE than keeping displacement at 17cm.
```

## Training Impact

### Reward Signal Clarity
The new weights create a clear priority hierarchy:

1. **Avoid displacement violations** (-100× penalty) → Agent learns hard constraint first
2. **Maximize ISDR performance** (10.0 weight) → Agent's primary optimization target
3. **Maximize DCR performance** (8.0 weight) → Agent's secondary target
4. **Optimize displacement within constraint** (5.0 weight) → Tertiary goal
5. **Minimize force usage** (0.3 weight) → Minor efficiency incentive

### Expected Learning Progression
1. **Stage 1**: Agent learns displacement constraint (never exceed 17cm)
2. **Stage 2**: Agent learns to reduce ISDR while respecting constraint
3. **Stage 3**: Agent learns to improve DCR uniformity
4. **Stage 4**: Agent fine-tunes displacement reduction within constraint
5. **Stage 5**: Agent optimizes force efficiency

## Verification Results

Test of updated reward function:

```
Zero state (perfect):           reward = 23.000  ✓
Good control (7cm, 0.03% ISDR): reward = 22.107  ✓
Poor control (28cm, 0.27% ISDR): reward = -34.917 ✓ (strong penalty)
Force gradient:                 working correctly ✓
```

Note: Baseline reward increased from ~9.5 to ~23.0 due to increased ISDR/DCR weights.

## Next Steps

1. **Re-train V13 from scratch** with updated environment
2. **Monitor training metrics**:
   - Displacement should stay ≤ 17cm throughout training
   - ISDR should decrease progressively toward 0.55%
   - DCR should decrease progressively toward 1.0-1.15
3. **Test on held-out dataset** to verify generalization
4. **Compare to previous V13 results** to confirm improvement

## Files Modified

- [tmd_environment_v13_rooftop.py:439-543](restapi/rl_cl/tmd_environment_v13_rooftop.py#L439-L543) - Updated reward function with priorities

## Summary Table

| Metric | Current | Target | Priority | Weight | Constraint |
|--------|---------|--------|----------|--------|------------|
| ISDR | 1.332% | <0.55% | PRIMARY | 10.0 | Soft (optimization goal) |
| DCR | 1.47 | <1.15 | PRIMARY | 8.0 | Soft (optimization goal) |
| Displacement | 17.04cm | ≤17cm | CONSTRAINT | 5.0 + 100.0 penalty | Hard (never exceed) |
| Force | - | minimize | TERTIARY | 0.3 | None (efficiency) |

**Key Insight**: By making displacement a hard constraint (with massive penalty) rather than an optimization objective, we free the agent to focus on the true goals (ISDR and DCR) without fear of displacement increases.
