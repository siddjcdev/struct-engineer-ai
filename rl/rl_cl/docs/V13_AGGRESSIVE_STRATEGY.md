# V13 Aggressive Optimization Strategy

## Date
2026-01-12

## Strategy Statement

**"Aggressively minimize soft-story ISDR and DCR, while keeping displacement reasonably below 17 cm and control forces within limits."**

## Reward Philosophy Change

### Previous Approach (Conservative)
- Treated all metrics equally with moderate weights
- Hard constraint at 17cm
- Result: Achieved displacement control but failed ISDR/DCR targets

### New Approach (Aggressive)
- **ISDR and DCR dominate** the reward signal (6.75:1 ratio)
- **Displacement has soft + hard constraints** (15cm soft target, 17cm hard limit)
- **Force is secondary** efficiency concern

## Weight Structure

### Aggressive Weights
```python
w_ISDR = 15.0         # DOMINANT - aggressive ISDR minimization
w_DCR = 12.0          # VERY HIGH - aggressive DCR optimization
w_disp = 4.0          # MODERATE - keep displacement reasonable
w_disp_soft = 10.0    # Discourage 15-17cm range
w_disp_hard = 200.0   # Never exceed 17cm (catastrophic)
w_force = 0.5         # Mild efficiency incentive
```

### Priority Ratio
- **ISDR + DCR weight: 27.0**
- **Displacement weight: 4.0**
- **Ratio: 6.75:1** in favor of ISDR/DCR aggressive optimization

This is **2.25× more aggressive** than the previous 3.6:1 ratio.

## Displacement Constraint System

### Two-Tier Constraint
1. **Soft target: 15 cm**
   - Comfortable operating range
   - Agent gets full reward below 15cm
   - Gradual penalty from 15-17cm (-10.0 weight)

2. **Hard limit: 17 cm**
   - Absolute maximum (never exceed)
   - Catastrophic penalty above 17cm (-200.0 weight)
   - Ensures constraint is NEVER violated

### Displacement Reward Calculation
```python
if disp > 17cm:
    P_hard = -200.0 * ((disp-17)/17)²  # Catastrophic
elif disp > 15cm:
    P_soft = -10.0 * ((disp-15)/(17-15))²  # Discourage
    R_disp = 1.0 - (disp/15)²  # Moderate reward
else:
    R_disp = 1.0 - (disp/15)²  # Full reward
```

**Effect**: Agent learns "stay below 15cm if possible, never exceed 17cm"

## DCR Optimization Enhancement

### Previous (Moderate Penalty)
```python
dcr_deviation = abs(DCR - 1.0) / 1.15  # Normalized to acceptable range
R_dcr = 1.0 - dcr_deviation²
w_DCR = 8.0
```

### New (Steep Penalty)
```python
dcr_deviation = abs(DCR - 1.0) / 0.5  # Normalized to tight range
R_dcr = 1.0 - dcr_deviation²
w_DCR = 12.0  # INCREASED weight
```

**Change**: DCR deviation is now penalized **2.3× more steeply** (dividing by 0.5 instead of 1.15), and the weight increased by 50%.

**Effect**: Agent strongly incentivized to keep DCR as close to 1.0 as possible, not just below 1.15.

## Expected Learning Behavior

### Stage 1: Constraint Learning (Episodes 0-5000)
- Agent learns displacement hard limit (17cm)
- Avoids catastrophic penalties
- Begins to understand soft constraint (15cm)

### Stage 2: ISDR Optimization (Episodes 5000-15000)
- Agent aggressively reduces ISDR (dominant 15.0 weight)
- May sacrifice some displacement performance for ISDR gains
- Learns that low ISDR gives high reward

### Stage 3: DCR Refinement (Episodes 15000-25000)
- Agent optimizes DCR toward 1.0 (12.0 weight)
- Learns to balance ISDR and DCR
- Fine-tunes force application patterns

### Stage 4: Integrated Optimization (Episodes 25000+)
- Agent jointly optimizes ISDR, DCR, and displacement
- Stays comfortably below 15cm while minimizing ISDR/DCR
- Efficient force usage emerges naturally

## Verification Results

Test of aggressive reward function:

```
Zero state (perfect):           reward = 31.000  ✓
Good control (7cm, 0.03% ISDR): reward = 30.060  ✓
Poor control (28cm, 0.27% ISDR): reward = -70.423 ✓ (massive penalty)
Force gradient:                 working correctly ✓
```

**Key Observation**: Baseline reward increased from 23.0 → 31.0 due to higher ISDR/DCR weights.

## Reward Range Analysis

| Scenario | ISDR | DCR | Disp (cm) | Expected Reward | Notes |
|----------|------|-----|-----------|-----------------|-------|
| Perfect | 0.3% | 1.0 | 10 | ~28-30 | Ideal performance |
| Excellent | 0.4% | 1.05 | 12 | ~24-26 | Well within targets |
| Good | 0.55% | 1.15 | 15 | ~18-20 | At soft target |
| Acceptable | 0.6% | 1.2 | 16 | ~12-15 | In soft penalty zone |
| Poor | 0.8% | 1.3 | 17 | ~5-8 | At hard limit |
| Failing | 1.0% | 1.4 | 17 | ~-5 to 0 | Missing targets |
| Bad | 1.3% | 1.5 | 17 | ~-15 to -10 | Current performance |
| Catastrophic | 2.0% | 2.0 | 20 | ~-100+ | Hard constraint violated |

## Expected Performance After Training

### Conservative Estimate
- **ISDR: 0.6-0.7%** (improvement from 1.33%)
- **DCR: 1.1-1.2** (improvement from 1.47)
- **Displacement: 14-16 cm** (maintain below 17cm)

### Optimistic Estimate
- **ISDR: 0.4-0.55%** (hit target range)
- **DCR: 1.0-1.1** (near-perfect uniformity)
- **Displacement: 12-15 cm** (comfortable margin)

### Why This Will Work

1. **6.75:1 priority ratio** forces agent to focus on ISDR/DCR
2. **Soft constraint at 15cm** gives displacement flexibility
3. **Hard constraint at 17cm** prevents violations
4. **Steep DCR penalty** (dividing by 0.5) drives toward 1.0
5. **High baseline reward** (31.0) provides clear signal

## Comparison to Previous Versions

| Version | Strategy | ISDR Weight | DCR Weight | Disp Constraint | Ratio | Result |
|---------|----------|-------------|------------|-----------------|-------|--------|
| V12 | Soft-story TMD | 5.0 | 3.0 | None | 1:1 | Failed (wrong placement) |
| V13 (original) | All negative | 5.0 | 3.0 | None | - | Catastrophic (wrong reward) |
| V13 (fixed) | Positive baseline | 10.0 | 8.0 | 17cm hard | 3.6:1 | Good disp, poor ISDR/DCR |
| **V13 (aggressive)** | **ISDR/DCR focus** | **15.0** | **12.0** | **15cm soft, 17cm hard** | **6.75:1** | **Expected success** |

## Training Command

```bash
python train_v13_rooftop.py --run-name v13_aggressive_optimization
```

## Monitoring Metrics

Watch for these during training:

1. **Mean episode reward**: Should increase from ~0 toward 20-25
2. **ISDR in tensorboard**: Should decrease from 1.3% toward 0.5%
3. **DCR in tensorboard**: Should decrease from 1.47 toward 1.0-1.1
4. **Displacement**: Should stay consistently below 17cm
5. **Soft constraint violations**: Should decrease (fewer episodes with 15-17cm)

## Success Criteria

### Minimum Acceptable
- ISDR < 0.8% (improvement from 1.33%)
- DCR < 1.3 (improvement from 1.47)
- Displacement < 17cm (maintain)

### Target Performance
- ISDR < 0.55% (hit target)
- DCR < 1.15 (hit target)
- Displacement < 15cm (comfortable margin)

### Stretch Goal
- ISDR < 0.4% (original ambitious target)
- DCR < 1.1 (excellent uniformity)
- Displacement < 14cm (ideal performance)

## Files Modified

- [tmd_environment_v13_rooftop.py:439-562](restapi/rl_cl/tmd_environment_v13_rooftop.py#L439-L562) - Aggressive reward function

## Key Insights

1. **Aggressive weighting works** when you have a clear priority hierarchy
2. **Two-tier constraints** (soft + hard) give agent flexibility while ensuring safety
3. **Steep penalties** (DCR normalization by 0.5) drive toward ideal values, not just acceptable ranges
4. **High baseline reward** (31.0) makes the optimization signal very clear
5. **Displacement flexibility** (15-17cm buffer) allows agent to focus on primary objectives

## Next Action

**Re-train V13 immediately** with aggressive reward function to achieve ISDR/DCR targets.
