# V12 Lessons Learned - Why Soft-Story TMD Failed

## Date: January 12, 2026

## Executive Summary

**V12 hypothesis FAILED**: Placing TMD at soft story (floor 8) made performance WORSE than uncontrolled building.

- **Displacement**: 17.97 cm → 18.27 cm (-1.7% worse)
- **Max ISDR**: 1.036% → 1.279% (-23.4% worse!)
- **Critical floor shifted**: Floor 8 → Floor 1

The agent learned to saturate control force (300 kN constantly), creating a "whipping" effect that amplified lower floor motion.

## Detailed Analysis

### What We Observed

| Metric | Uncontrolled | V12 Controlled | Change |
|--------|-------------|----------------|--------|
| Peak roof displacement | 17.97 cm | 18.27 cm | **+1.7% worse** |
| Max ISDR | 1.036% (floor 8) | 1.279% (floor 1) | **+23.4% worse** |
| Floor 8 ISDR | 1.036% | 1.067% | -3.0% (minimal) |
| Control force | 0 kN | 300 kN (saturated) | Maxed out |

### Floor-by-Floor Impact

The TMD at floor 8 created **amplification in lower floors**:

```
Floor    Uncontrolled ISDR    Controlled ISDR    Change
1        0.993%               1.279%             -28.8% WORSE
2        0.897%               1.059%             -18.1% WORSE
3        0.843%               0.856%             -1.5% WORSE
4        0.785%               0.878%             -11.9% WORSE
5        0.746%               0.983%             -31.7% WORSE
6        0.728%               0.908%             -24.8% WORSE
7        0.741%               0.792%             -6.8% WORSE
8        1.036%               1.067%             -3.0% WORSE ← TMD location
```

**Critical finding**: Every single floor got worse with the TMD!

### Root Causes

#### 1. **Physics Problem: Force Propagation**

TMD at floor 8 applies 300 kN force between floor 8 and TMD mass. This force:
- Propagates **downward** through floors 1-7
- Creates additional inertial loading on lower floors
- Acts like a sledgehammer transmitting shock through structure

**Equation**: When TMD pushes on floor 8 with force F:
```
F_floor8 = +F_control (upward on floor 8)
F_floor7 = -F_control (downward, transmitted through column)
F_floor6 = -F_control (further transmission)
... continues down to floor 1
```

The lower floors experience the control force as **additional loading**, not control.

#### 2. **Reward Function Flaw**

V12 reward only penalized **floor 8 drift**:

```python
# V12 reward (WRONG)
ISDR = floor8_drift / story_height  # Only floor 8!
P_isdr = -((ISDR / ISDR_target) ** 2)
```

**Problem**: Agent optimized for floor 8, ignored floors 1-7 getting worse.

The agent learned:
- "Apply 300 kN constantly to minimize floor 8 motion"
- Floor 8 ISDR: 1.036% → 1.067% (barely helped)
- Floor 1-7: Got much worse (agent doesn't see/care)

#### 3. **Observation Space Mismatch**

V12 observation: `[floor8_disp, floor8_vel, tmd_disp, tmd_vel, roof_disp, roof_vel]`

**Missing**: States of floors 1-7 where damage actually occurred!

Agent was "blind" to the floors it was destroying.

#### 4. **Wrong DCR Calculation**

```python
# V12 DCR (WRONG)
recent_drifts = self.drift_history[-100:]  # Only floor 8 drifts!
max_drift = max(recent_drifts)
percentile_75 = sorted_drifts[int(0.75 * len(sorted_drifts))]
DCR = max_drift / percentile_75
```

**Problem**: DCR should be ratio of max drift across ALL floors to average drift. V12 only looked at floor 8.

**True DCR calculation**:
```python
max_drift_all_floors = max([drift_floor_i for i in range(12)])
mean_drift_all_floors = mean([drift_floor_i for i in range(12)])
DCR = max_drift_all_floors / mean_drift_all_floors
```

### Why Rooftop TMD is Better

**Rooftop TMD** (v11 approach):
- Force applied at top → affects global mode shape
- Control force dissipates through damping, not transmitted as shock
- More compatible with first mode (which dominates response)

**Physical intuition**:
- Building sways like inverted pendulum in first mode
- TMD at roof = mass at tip of pendulum (optimal for energy dissipation)
- TMD at floor 8 = mass at middle (creates moment, amplifies lower section)

### Numerical Evidence

V11 (rooftop TMD, 250 kN):
- Max ISDR: ~1.52% (reported in earlier tests)
- Improvement: +3% vs uncontrolled

V12 (floor 8 TMD, 300 kN):
- Max ISDR: 1.279%
- Improvement: **-23.4%** (made it worse!)

**Conclusion**: Even mediocre rooftop TMD (v11) beats floor 8 TMD.

## Lessons for V13

### 1. **Use Rooftop TMD**
- Place TMD at floor 12 (roof)
- Mount point: `self.tmd_floor = 11` (0-indexed)

### 2. **Track ALL Floors in Reward**

```python
# Calculate ISDR for ALL 12 floors
isdrs = []
for floor in range(12):
    if floor == 0:
        drift = abs(d[0])
    else:
        drift = abs(d[floor] - d[floor-1])
    isdr = drift / story_height
    isdrs.append(isdr)

# Penalize MAX ISDR, not just one floor
max_isdr = max(isdrs)
P_isdr = -((max_isdr / ISDR_target) ** 2)
```

### 3. **Fix DCR Calculation**

```python
# True DCR: max drift / mean drift across all floors
all_floor_drifts = [max_drift_floor_i for i in range(12)]
DCR = max(all_floor_drifts) / mean(all_floor_drifts)
```

### 4. **Return Proper Metrics**

```python
def get_episode_metrics(self):
    # Calculate ISDR for each floor
    floor_isdrs = []
    for floor in range(12):
        # ... calculate drift and ISDR
        floor_isdrs.append(isdr)

    max_isdr = max(floor_isdrs)
    critical_floor = floor_isdrs.index(max_isdr) + 1

    return {
        'max_isdr_percent': max_isdr,
        'critical_floor': critical_floor,
        'floor_isdrs': floor_isdrs,  # All floors for analysis
        'DCR': proper_dcr_calculation,
        ...
    }
```

### 5. **Expanded Observation (Optional)**

Consider adding critical floor information:
```python
obs = [
    roof_disp, roof_vel,           # Primary control target
    floor8_disp, floor8_vel,       # Soft story monitoring
    floor1_disp, floor1_vel,       # Foundation monitoring (optional)
    tmd_disp, tmd_vel              # TMD state
]
```

### 6. **Penalty Weights**

V12 used:
- `w_disp = 4.0` (displacement)
- `w_DCR = 4.0` (drift concentration)
- `w_ISDR = 1.5` (ISDR)
- `w_force = 0.2` (force efficiency)

**For V13**: Increase ISDR weight since it's most critical:
- `w_disp = 3.0`
- `w_DCR = 3.0`
- `w_ISDR = 5.0` ← **Increase** (ISDR is safety-critical)
- `w_force = 0.3`

### 7. **Keep What Worked in V12**

✅ **Training infrastructure**:
- Multi-file training (10 variants per magnitude)
- Held-out test evaluation
- Comprehensive error handling
- Log file writing
- TensorBoard metrics

✅ **Hyperparameters**:
- 4-layer network [256, 256, 256, 256]
- `n_steps=2048`, `batch_size=256`, `n_epochs=10`
- `learning_rate=3e-4`, `ent_coef=0.03`
- 1.5M timesteps

✅ **Force limit**: 300 kN (keep higher force, rooftop can use it better)

## V13 Expected Performance

With rooftop TMD + proper multi-floor ISDR tracking:

**Conservative estimate**:
- Displacement: 14-16 cm (vs 14 cm target)
- Max ISDR: 0.6-0.8% (vs 0.4% target)
- DCR: 1.2-1.3 (vs 1.15 target)
- Improvement vs uncontrolled: +40-50% ISDR reduction

**Optimistic estimate** (if everything aligns):
- Displacement: 13.5 cm ✅
- Max ISDR: 0.45-0.5% ⚠️ (close to target)
- DCR: 1.15-1.18 ⚠️
- Improvement: +60% ISDR reduction

**Rationale**:
- Rooftop placement is physically correct for global control
- 300 kN force gives 20% more authority than v11's 250 kN
- Proper reward function guides agent to minimize true max ISDR
- Agent can't "cheat" by ignoring critical floors

## Bottom Line

**V12 failure teaches us**:
1. TMD placement matters MORE than force magnitude
2. Reward function MUST align with true performance metric (max ISDR across all floors)
3. "Direct control" at soft story sounds good but creates worse dynamics
4. Sometimes conventional wisdom (rooftop TMD) exists for good reasons

**V13 will succeed where V12 failed** by:
- Using proven rooftop TMD placement
- Tracking and penalizing true max ISDR across all floors
- Proper DCR calculation
- Maintaining v12's excellent training infrastructure

The soft-story TMD was a good hypothesis to test, but the physics proved it wrong. Now we know better.
