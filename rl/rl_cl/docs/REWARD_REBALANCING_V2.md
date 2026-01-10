# Reward Rebalancing V2: Making ISDR the Dominant Objective

## Date: January 9, 2026

## Problem: ISDR Got Worse Despite Aggressive Penalties

After implementing aggressive ISDR/DCR constraints, training results showed:

```
Previous (weak constraints):
  Displacement: 20.63 cm
  ISDR: 1.42%
  DCR: 1.30

Current (aggressive constraints):
  Displacement: 20.39 cm (improved by 0.24 cm) ✓
  ISDR: 1.76% (WORSE by 0.34%!) ✗
  DCR: 1.29 (slightly better) ✓
```

**The agent is trading ISDR for displacement** - it's willing to accept 0.34% more ISDR to get 0.24 cm less displacement.

## Root Cause Analysis

### Reward Component Breakdown (at 20cm displacement, 1.76% ISDR)

**Before rebalancing**:
```
Displacement penalty: -(0.4)² = -0.16 per step (33% of signal)
ISDR penalty: -0.32 per step (67% of signal)
Total structural signal: -0.48 per step
```

**Diagnosis**: ISDR penalty (-0.32) is already 2× stronger than displacement (-0.16), but the agent **still prioritizes displacement**.

**This reveals**: Displacement and ISDR are **physically coupled** with your TMD configuration. Reducing displacement to 20cm inherently causes ~1.7% ISDR.

## The Solution: Invert the Priority

Make ISDR the **dominant objective** and displacement **secondary**.

### Changes Made

#### 1. Reduced Displacement/Velocity Penalties (50% reduction)

**Before**:
```python
displacement_penalty = -(disp_normalized ** 2)
velocity_penalty = -(vel_normalized ** 2)
```

**After**:
```python
displacement_penalty = -0.5 * (disp_normalized ** 2)  # 50% of original
velocity_penalty = -0.5 * (vel_normalized ** 2)       # 50% of original
```

**Effect**: At 20cm displacement:
- Before: -0.16 penalty
- After: **-0.08 penalty** (halved)

#### 2. Doubled ISDR/DCR Bonuses

**Before**:
```python
isdr_bonus = 0.05 * (0.005 - current_isdr) / 0.005  # Max +0.05
dcr_bonus = 0.03 * (1.1 - current_dcr) / 0.1        # Max +0.03
```

**After**:
```python
isdr_bonus = 0.10 * (0.005 - current_isdr) / 0.005  # Max +0.10 (2× stronger)
dcr_bonus = 0.06 * (1.1 - current_dcr) / 0.1        # Max +0.06 (2× stronger)
```

**Effect**: Stronger positive reinforcement for meeting targets.

## New Reward Balance

### At Current Performance (20cm, 1.76% ISDR, 1.29 DCR)

```
Displacement penalty: -0.08  (17% of signal)
Velocity penalty:     -0.04  (9%)
ISDR penalty:         -0.32  (70% - DOMINANT!)
DCR penalty:          -0.02  (4%)
Total:                -0.46
```

**ISDR is now 4× more important than displacement!**

### At Target Performance (14cm, 0.4% ISDR, 1.05 DCR)

```
Displacement penalty: -0.04  (36%)
Velocity penalty:     -0.02  (18%)
ISDR bonus:           +0.08  (73% - POSITIVE!)
DCR bonus:            +0.03  (27% - POSITIVE!)
Total:                +0.05  (POSITIVE REWARD!)
```

**Improvement gradient**: From -0.46 to +0.05 = **0.51 reward units** (massive incentive!)

### The Key Insight

With this rebalancing:
- Agent gets **-0.32 ISDR penalty** for being at 1.76%
- Agent gets **+0.08 ISDR bonus** for reaching 0.4%
- **Net improvement**: 0.40 reward units just from ISDR!

Compare to displacement:
- Agent gets **-0.08 penalty** for 20cm
- Agent gets **-0.04 penalty** for 14cm
- **Net improvement**: 0.04 reward units (10× less than ISDR!)

**The agent now has clear incentive to prioritize ISDR over displacement.**

## Expected Outcomes

### Scenario 1: Agent Finds ISDR-Optimal Control

```
M4.5 Results (optimistic):
  Displacement: 16-22 cm (may increase slightly from 20.39 cm)
  ISDR: 0.5-0.8% (dramatic improvement from 1.76%)
  DCR: 1.0-1.15 (near-elastic)
```

**Trade-off**: Accept 2-4 cm more displacement to achieve 1.0% less ISDR.

This is **exactly what you want** - "almost no structural damage" is the priority.

### Scenario 2: Physical Limits Reached

```
M4.5 Results (realistic):
  Displacement: 14-18 cm (meets target)
  ISDR: 0.6-0.9% (significant improvement but not quite 0.4%)
  DCR: 1.05-1.2 (nearly elastic)
```

**If agent converges here**, it means 0.4% ISDR is **physically impossible** with:
- 2% TMD mass ratio
- 150 kN active force limit
- Current building parameters

**This would still be excellent** - 0.7% ISDR is "minimal structural damage".

## What to Do If Scenario 2 Occurs

If training converges at ~0.7% ISDR despite the aggressive rebalancing:

### Option A: Increase TMD Mass Ratio

```python
# In rl_cl_tmd_environment.py
self.tmd_mass = 0.03 * self.floor_mass  # 3% instead of 2%
```

**Effect**: More passive damping, lower ISDR achievable.

### Option B: Increase Active Force Limit

```python
# In ppo_config_v9_advanced.py
'force_limit': 200_000,  # 200 kN instead of 150 kN
```

**Effect**: More control authority, finer force profiles possible.

### Option C: Relax Targets

Accept that 0.6-0.8% ISDR is "almost no structural damage" for your system:
- 0.5%: Immediate occupancy (building fully functional)
- 0.7%: Life safety easily met, minimal damage
- 1.5%: Code-compliant life safety threshold

**Your 0.7% result would be excellent performance.**

## Comparison: V1 vs V2 Rebalancing

| Aspect | V1 (Aggressive) | V2 (ISDR-Dominant) |
|--------|----------------|-------------------|
| **Displacement Penalty** | -1.0 × norm² | -0.5 × norm² (50% weaker) |
| **ISDR Penalty** | -0.20 max | -0.20 max (unchanged) |
| **ISDR Bonus** | +0.05 max | +0.10 max (2× stronger) |
| **DCR Bonus** | +0.03 max | +0.06 max (2× stronger) |
| **ISDR Signal Weight** | 67% of total | 70-80% of total |
| **Displacement Weight** | 33% of total | 17-20% of total |
| **Priority** | Balanced | **ISDR > Displacement** |
| **Expected M4.5 ISDR** | 1.76% (actual) | 0.5-0.8% |
| **Expected M4.5 Disp** | 20.39 cm (actual) | 16-22 cm |

## Training Instructions

**Retrain from scratch** with the rebalanced reward:

```bash
cd c:\Dev\dAmpIng26\git\struct-engineer-ai\rl\rl_cl
python train_v11.py --run-name isdr_dominant_v2
```

**Monitor these metrics in TensorBoard**:
- `metrics/max_isdr_percent` - should **decrease dramatically** below 1.0% within 200k steps
- `metrics/avg_peak_displacement_cm` - may increase slightly (acceptable trade-off)
- `rollout/ep_rew_mean` - should improve (become less negative or positive!)

### Success Criteria

**Excellent Result** (Scenario 1):
- ISDR < 0.8%
- Displacement < 25 cm
- DCR < 1.2

**Good Result** (Scenario 2):
- ISDR: 0.8-1.0%
- Displacement: 14-20 cm
- DCR: 1.1-1.3

**Poor Result** (need to investigate):
- ISDR > 1.2% (rebalancing didn't work)
- Displacement > 30 cm (sacrificed too much)

## Files Modified

1. **rl_cl_tmd_environment.py** (lines 432-433):
   - Reduced displacement penalty: -1.0 → -0.5
   - Reduced velocity penalty: -1.0 → -0.5

2. **rl_cl_tmd_environment.py** (line 489):
   - Increased ISDR bonus: +0.05 → +0.10

3. **rl_cl_tmd_environment.py** (line 502):
   - Increased DCR bonus: +0.03 → +0.06

4. **rl_cl_tmd_environment.py** (lines 507-519):
   - Updated reward magnitude documentation

## Key Takeaway

**The problem wasn't that ISDR penalties were too weak** - they were already 2× stronger than displacement.

**The problem was that displacement was still the implicit priority.** By halving displacement penalties and doubling structural bonuses, we've **inverted the objective hierarchy**:

**Before**: "Minimize displacement, avoid extreme ISDR"
**After**: "Minimize ISDR, keep displacement reasonable"

This aligns with your stated goal: **"almost no structural damage"** (structural safety is the priority).
