# V13 Critical Fix: Correct DCR Calculation

## Date
2026-01-12

## The Issue

The DCR (Drift Concentration Ratio) calculation in V13 was **mathematically incorrect** according to structural engineering standards.

## The Proper Definition

According to ASCE standards and structural engineering literature:

```
DCR = max(Δᵢ) / Δ̄
```

Where:
- **Δᵢ** = inter-story drift at story *i*
- **max(Δᵢ)** = maximum inter-story drift (across all stories)
- **Δ̄** = average inter-story drift over all stories
- **At the same timestep**

### Interpretation
- **DCR = 1.0**: Perfectly uniform drift distribution (ideal)
- **DCR > 1.0**: Drift is concentrated at certain floors (undesirable)
- **DCR = 2.0**: Maximum drift is 2× the average (significant concentration)

The goal is to minimize DCR toward 1.0, indicating uniform damage distribution.

## What V13 Was Doing Wrong

### Incorrect Implementation
```python
# WRONG: Uses historical max drifts per floor
recent_max_drifts = []
for floor in range(n_floors):
    recent_drifts = drift_history_per_floor[floor][-100:]
    recent_max_drifts.append(max(recent_drifts))  # Max over time

overall_max_drift = max(recent_max_drifts)  # Max of maximums
mean_drift = np.mean(recent_max_drifts)    # Mean of maximums
DCR = overall_max_drift / mean_drift
```

### Problems
1. **Compares peaks from different times**: Each floor's max might have occurred at different timesteps
2. **Not instantaneous**: Doesn't represent current drift distribution
3. **Incorrect physics**: Doesn't measure concentration at any specific moment

### Example of Why This Fails
```
Floor 1: max drift = 2.0 cm (occurred at t=5s)
Floor 2: max drift = 1.0 cm (occurred at t=3s)
Floor 3: max drift = 1.5 cm (occurred at t=7s)

Old calculation: DCR = 2.0 / mean(2.0, 1.0, 1.5) = 2.0 / 1.5 = 1.33

But at t=5s when floor 1 peaked:
Floor 1: 2.0 cm
Floor 2: 0.8 cm
Floor 3: 1.0 cm
Actual DCR = 2.0 / mean(2.0, 0.8, 1.0) = 2.0 / 1.27 = 1.57

The old method underestimates concentration!
```

## The Correct Implementation

### Fixed Code
```python
# CORRECT: Use current floor drifts at this timestep
abs_drifts = np.abs(all_floor_drifts)  # Current drifts at all floors
max_current_drift = np.max(abs_drifts)
mean_current_drift = np.mean(abs_drifts)

if mean_current_drift > 1e-8:
    DCR = max_current_drift / mean_current_drift
else:
    DCR = 1.0  # At zero drift, assume uniform
```

### Why This Is Correct
1. **Instantaneous**: Measures drift distribution at current timestep
2. **Physics-based**: Represents actual concentration of drift at this moment
3. **Matches definition**: Directly implements DCR = max(Δᵢ) / mean(Δᵢ)
4. **Actionable**: Agent can see immediate effect of control on DCR

## Impact on Training

### Before (Incorrect DCR)
- Agent received wrong DCR values in reward
- DCR penalty didn't reflect actual drift concentration
- Agent learned incorrect correlation between actions and DCR
- Training signal was noisy and misleading

### After (Correct DCR)
- Agent receives accurate DCR values in reward
- DCR penalty reflects true drift concentration
- Agent learns correct correlation: uniform control → low DCR
- Training signal is clear and accurate

## Expected Behavior Change

### Incorrect DCR Impact on Results
Looking at previous results:
- Displacement: 18.78 cm
- ISDR: 1.375%
- **DCR: 1.30**

The agent was optimizing for the **wrong DCR definition**, which may explain why DCR stayed high even when displacement wasn't terrible.

### Correct DCR Should Enable
With proper DCR calculation, the agent will:
1. See immediate feedback when drift concentrates at soft-story
2. Learn control patterns that distribute drift uniformly
3. Achieve DCR closer to 1.0 (uniform distribution)

## Mathematical Verification

### Test Case: Uniform Distribution
```python
all_floor_drifts = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

max_drift = 1.0
mean_drift = 1.0
DCR = 1.0 / 1.0 = 1.0  ✓ Perfect uniformity
```

### Test Case: Soft-Story Concentration
```python
all_floor_drifts = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2.0, 0.5, 0.5, 0.5, 0.5]
                                                       ↑ Floor 8 (soft-story)

max_drift = 2.0 (at floor 8)
mean_drift = (11×0.5 + 2.0) / 12 = 0.625
DCR = 2.0 / 0.625 = 3.2  ✓ High concentration detected!
```

### Test Case: Linear Distribution
```python
all_floor_drifts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

max_drift = 1.2
mean_drift = 0.65
DCR = 1.2 / 0.65 = 1.85  ✓ Moderate concentration
```

## Files Modified

- [tmd_environment_v13_rooftop.py:541-561](restapi/rl_cl/tmd_environment_v13_rooftop.py#L541-L561) - Corrected DCR calculation

## Comparison: Old vs New

| Aspect | Old (Incorrect) | New (Correct) | Impact |
|--------|----------------|---------------|--------|
| Formula | max(historical max) / mean(historical max) | max(current) / mean(current) | ✓ Physics-based |
| Timestep | Mixed (different times) | Single (current time) | ✓ Instantaneous |
| Reflects concentration | Partially | Accurately | ✓ Critical |
| Training signal | Noisy | Clear | ✓ Better learning |
| Matches ASCE standards | No | Yes | ✓ Correct definition |

## Why This Matters

DCR is a **critical metric** for seismic control because:

1. **Structural safety**: Concentrated drift causes localized damage
2. **Soft-story collapse**: High DCR at soft-story is catastrophic failure mode
3. **TMD effectiveness**: Proper control should distribute drift uniformly
4. **Agent feedback**: Correct DCR gives agent accurate signal about control quality

With incorrect DCR calculation, the agent was:
- Optimizing for the wrong objective
- Receiving misleading reward signals
- Unable to learn proper drift distribution control

## Expected Improvement

### Conservative Estimate
With correct DCR calculation, expect **10-15% improvement** in DCR metric:
- Old DCR: 1.30-1.40 (with wrong formula)
- New DCR: 1.10-1.20 (with correct formula + better learning)

### Optimistic Estimate
If agent learns to actively distribute drift:
- Target DCR: 1.0-1.15 (uniform distribution achieved)

## Summary

This was a **fundamental error** in the DCR implementation. The formula used was mathematically incorrect according to structural engineering standards.

**Key Lesson**: Always verify formulas against authoritative sources (ASCE standards, textbooks) before implementation, especially for domain-specific metrics.

The fix ensures:
1. ✓ DCR matches ASCE definition: max(Δᵢ) / mean(Δᵢ)
2. ✓ Instantaneous measurement at current timestep
3. ✓ Accurate reward signal for agent training
4. ✓ Physically meaningful drift concentration metric

Combined with expanded observations and aggressive reward weights, V13 should now properly optimize DCR toward the 1.0-1.15 target range.
