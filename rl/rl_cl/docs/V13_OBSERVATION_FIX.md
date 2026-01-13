# V13 Critical Fix: Expanded Observation Space

## Date
2026-01-12

## The Problem (Root Cause Identified!)

**Results continued to fail despite reward function changes:**
- Displacement: 18.78 cm ❌ (worse than before!)
- ISDR: 1.375% ❌ (2.5× target)
- DCR: 1.30 ❌ (above 1.15 target)

**Root Cause**: The agent was trying to optimize ISDR and DCR across **12 floors** while only observing **2 floors** (roof + floor 8).

## The Fundamental Issue

### What We Asked the Agent to Do
- Minimize max_ISDR across ALL 12 floors
- Optimize DCR (drift concentration across ALL 12 floors)
- Keep displacement below 17 cm

### What the Agent Could See (OLD)
```python
Observation = [
    roof_displacement,    # Floor 12
    roof_velocity,
    tmd_displacement,
    tmd_velocity,
    floor8_displacement,  # Floor 8
    floor8_velocity
]
# Total: 6 values, representing 2 out of 12 floors
```

**The Problem**: The agent was **blind to 10 out of 12 floors**. It couldn't optimize what it couldn't observe!

### Analogy
Imagine asking someone to:
- Balance 12 spinning plates
- Keep the wobbliest plate from falling
- BUT they can only see 2 of the 12 plates

**Result**: They can't succeed because they lack the information needed to accomplish the task.

## The Solution

### Expanded Observation Space (NEW)
```python
Observation = [
    floor1_displacement,   # Floor 1 (bottom)
    floor1_velocity,
    floor4_displacement,   # Floor 4 (lower-mid)
    floor4_velocity,
    floor8_displacement,   # Floor 8 (soft-story - CRITICAL)
    floor8_velocity,
    floor11_displacement,  # Floor 11 (near-top)
    floor11_velocity,
    roof_displacement,     # Floor 12 (roof)
    roof_velocity,
    tmd_displacement,
    tmd_velocity
]
# Total: 12 values, representing 5 key floors spanning full building height
```

### Floor Selection Strategy

**Selected floors provide strategic coverage:**

1. **Floor 1 (bottom)**: Base shear and foundation response
2. **Floor 4 (lower-mid)**: Lower section behavior
3. **Floor 8 (soft-story)**: CRITICAL - this is where ISDR concentrates
4. **Floor 11 (near-top)**: Upper section behavior
5. **Floor 12 (roof)**: TMD attachment point and global response

**Coverage**: These 5 floors span the entire building height and capture:
- Bottom section (floors 1-4)
- Middle/soft-story section (floor 8)
- Top section (floors 11-12)

This gives the agent a representative "view" of the drift profile across the full structure.

## What This Enables

### Before (Blind Optimization)
- Agent sees roof moving → applies TMD force
- Agent sees floor 8 moving → adjusts strategy
- Agent is BLIND to floors 2, 3, 4, 5, 6, 7, 9, 10, 11
- Cannot detect if drift is concentrating at unseen floors
- Cannot optimize DCR (needs to see drift distribution)
- Cannot minimize max ISDR (might be at an unseen floor)

### After (Informed Optimization)
- Agent sees floor 1 → knows base response
- Agent sees floor 4 → knows lower section behavior
- Agent sees floor 8 → knows soft-story is critical
- Agent sees floor 11 → knows upper section response
- Agent sees roof → knows global displacement
- Can detect drift patterns and concentration
- Can optimize DCR by seeing drift distribution
- Can minimize max ISDR by observing key floors

## Why This Will Work

### 1. Direct Observability
The agent can now "see" the drift patterns it's being rewarded for optimizing.

### 2. Soft-Story Detection
Floor 8 (soft-story) is directly observed, so agent can detect and respond to drift concentration there.

### 3. DCR Optimization
With 5 floors observed, agent can estimate drift distribution and work toward uniform distribution (DCR → 1.0).

### 4. ISDR Minimization
Agent can observe multiple floors simultaneously and learn which control patterns minimize ISDR across the structure.

### 5. Neural Network Capacity
The observation space increased from 6 → 12 values, which is well within PPO's capacity (we've successfully trained with larger observation spaces before).

## Technical Changes

### Observation Space Expansion
```python
# OLD: 6-dimensional observation
Shape: (6,)
Low: [-5, -20, -15, -60, -5, -20]
High: [5, 20, 15, 60, 5, 20]

# NEW: 12-dimensional observation
Shape: (12,)
Low: [-5, -20, -5, -20, -5, -20, -5, -20, -5, -20, -15, -60]
High: [5, 20, 5, 20, 5, 20, 5, 20, 5, 20, 15, 60]
```

### Observation Function Update
```python
def _get_observation(self) -> np.ndarray:
    """
    EXPANDED OBSERVATION for ISDR/DCR optimization.

    Observing multiple floors allows the agent to:
    - Detect drift patterns across the building height
    - Identify which floors have high ISDR
    - Optimize DCR by seeing drift distribution
    - Apply control to minimize max ISDR across all floors
    """
    # Returns 12 normalized values from 5 strategic floors + TMD
```

## Impact on Training

### Network Architecture
PPO's default MLP policy will automatically adapt to the 12-dimensional input:
- Input layer: 12 neurons (was 6)
- Hidden layers: [64, 64] (unchanged)
- Output layer: 1 neuron for control force (unchanged)

The network has **sufficient capacity** to process 12 inputs - this is not a bottleneck.

### Training Time
- **No significant impact** expected
- Slightly more computation per forward pass (6 → 12 inputs)
- But the agent can learn MUCH faster because it has the right information

### Sample Efficiency
**EXPECTED IMPROVEMENT**: The agent should learn faster because:
- It can directly observe what it's being rewarded for
- No need to infer unseen floor states from limited observations
- Clear signal between observations and rewards

## Expected Results After Re-training

### With Previous Observation Space (6 values)
- Agent was guessing about 10 unseen floors
- Could not effectively optimize ISDR/DCR
- Result: 18.78cm disp, 1.375% ISDR, 1.30 DCR (all failing)

### With New Observation Space (12 values)
**Conservative Estimate:**
- ISDR: 0.7-0.9% (significant improvement)
- DCR: 1.15-1.25 (improvement toward target)
- Displacement: 15-17 cm (respects constraints)

**Optimistic Estimate:**
- ISDR: 0.5-0.6% (hits target range!)
- DCR: 1.05-1.15 (excellent uniformity)
- Displacement: 12-15 cm (comfortable margin)

## Why Previous Attempts Failed

Looking back at the progression:

1. **V13 original**: Wrong reward function (all negative) → catastrophic failure
2. **V13 fixed reward**: Positive baseline but agent couldn't see what to optimize → poor ISDR/DCR
3. **V13 aggressive weights**: 6.75:1 priority but still blind to most floors → still poor
4. **V13 expanded observation (NOW)**: Agent can finally see and optimize what we're asking for!

**Key Insight**: Reward function tuning can't overcome missing observations. If the agent can't observe it, it can't optimize it.

## Comparison Table

| Aspect | V13 Original (6 obs) | V13 New (12 obs) | Impact |
|--------|---------------------|------------------|--------|
| Floors observed | 2 (roof, floor8) | 5 (1, 4, 8, 11, 12) | 2.5× more coverage |
| Can detect soft-story drift? | Partially (floor 8 only) | Yes (direct observation) | ✓ Improved |
| Can optimize DCR? | No (blind to distribution) | Yes (sees distribution) | ✓ Critical |
| Can minimize max ISDR? | No (might be at unseen floor) | Yes (key floors observed) | ✓ Critical |
| Can detect drift concentration? | No | Yes | ✓ Critical |
| Network capacity needed | Low (6 inputs) | Low (12 inputs) | ✓ No issue |
| Training complexity | N/A | Slightly higher | Minimal |
| **Expected ISDR** | **1.3-1.4%** | **0.5-0.7%** | **2× improvement** |
| **Expected DCR** | **1.3-1.5** | **1.0-1.15** | **Hit target** |

## Files Modified

- [tmd_environment_v13_rooftop.py:144-165](restapi/rl_cl/tmd_environment_v13_rooftop.py#L144-L165) - Expanded observation space definition
- [tmd_environment_v13_rooftop.py:350-393](restapi/rl_cl/tmd_environment_v13_rooftop.py#L350-L393) - Updated observation function

## Critical Lesson

**Observability is fundamental to optimization.**

You cannot optimize what you cannot observe. Before tuning rewards or hyperparameters, ensure the agent has access to the information needed to achieve the objective.

In this case:
- **Objective**: Minimize max ISDR across 12 floors
- **Required**: Observe enough floors to detect where ISDR is high
- **Previous**: Observed 2 floors (insufficient)
- **Now**: Observe 5 strategic floors (sufficient)

## Next Action

**Re-train V13 with expanded observation space:**

```bash
python train_v13_rooftop.py --run-name v13_expanded_observations
```

This combines:
1. ✓ Aggressive ISDR/DCR optimization (6.75:1 weight ratio)
2. ✓ Proper displacement constraints (15cm soft, 17cm hard)
3. ✓ **NEW: Expanded observations (5 floors + TMD)**

The agent now has the tools, incentives, AND information to succeed.
