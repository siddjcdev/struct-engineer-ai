# Training Failure Analysis - Hybrid Control Attempt
## Date: January 4, 2026

## Stage 1 Results (M4.5, 500 kN force limit)

**Final Performance**:
- Peak displacement: 20.66 cm (target: 14 cm) ✗ 48% worse
- Max ISDR: 1.57% (target: 0.4%) ✗ **293% worse**
- Max DCR: 28.93 (target: 1.0-1.1) ✗ **2,630% worse** (CATASTROPHIC)
- Mean force: 1.4 kN (out of 500 kN) ✗ **0.28% utilization**

**Training Metrics**:
- Episode reward: -135k to -176k
- Value loss: 73-114 million (catastrophic)
- Explained variance: 0.008-0.029 (should be > 0.8)
- FPS: 363-368 steps/second
- Total timesteps: 200,000 (completed)

## Root Cause: Reward Magnitude Explosion

### The DCR Penalty Problem

With DCR = 28.93 and the current penalty formula:

```python
dcr_penalty = -10.0 * ((28.93 - 1.0) ** 2)  # Base penalty
dcr_penalty += -100.0 * (28.93 - 1.75) ** 2) # Excess penalty above 1.75

# Calculation:
base = -10.0 * (27.93)² = -7,801
excess = -100.0 * (27.18)² = -73,857
total_per_step = -81,658
scaled_by_0.1 = -8,166 per step
over_2000_steps = -16,332,000 cumulative!
```

This **-16.3 million reward** from DCR alone completely dominates all other signals:
- Displacement reward: ~+6,000 (0.04% of DCR penalty magnitude)
- ISDR penalty: ~-2,000 (0.01% of DCR penalty magnitude)
- Velocity penalty: ~-200 (0.001% of DCR penalty magnitude)

### Why the Value Function Collapsed

The value network tries to predict cumulative returns, which vary from:
- Best case (with good control): -50,000
- Worst case (with bad control): -16,000,000

This **320x range** causes:
1. Value loss of 73-114 million (gradient explosion)
2. Explained variance of 0.008-0.029 (value function learns nothing)
3. Policy becomes risk-averse: "any action might make DCR worse, so do nothing"

## Why Hybrid Control Failed

**Configuration**:
- Weak passive TMD: k=10 kN/m, c=500 N·s/m
- Strong active control: 500 kN force limit

**Problem**: With k=10 kN/m, the passive TMD provides almost no damping, leading to:
- Uncontrolled baseline: 54.07 cm displacement
- **Uncontrolled DCR: ~28-30 (catastrophic weak story failure)**

The building with weak passive TMD exhibits catastrophic weak story behavior. The active control has no chance to learn because:
1. Any action that doesn't perfectly solve the DCR problem gets -16M reward
2. Value network cannot learn with such extreme targets
3. Agent learns "do nothing" is safest (minimizes variance in terrible rewards)

## Previous Attempts Recap

### Attempt 1: Pure Passive TMD (k=50 kN/m, c=2 kN·s/m)
- Baseline: 21.02 cm
- Agent result: 21.58 cm (made it WORSE)
- **Issue**: Passive system already near-optimal, no room to improve

### Attempt 2: Pure Active Control (k=0, c=0)
- Baseline: 20.90 cm
- Agent result: 20.93 cm, DCR=25.61
- Force usage: 2.3 kN out of 500 kN
- **Issue**: Agent learned "do nothing" strategy

### Attempt 3: Hybrid Control (k=10 kN/m, c=500 N·s/m, 500 kN active)
- Baseline: 54.07 cm
- Agent result: 20.66 cm (GOOD displacement!)
- DCR: 28.93 (CATASTROPHIC)
- Force usage: 1.4 kN out of 500 kN
- **Issue**: DCR penalty so massive it prevented value function from learning

## Fundamental Problem: The Weak Story

The building has a weak 8th floor (60% stiffness of other floors). This creates:
- Natural tendency for drift concentration
- DCR naturally exceeds 1.5-2.0 even with good displacement control
- Without proper TMD tuning, DCR can reach 25-30

**The contradiction**:
- To reduce displacement: need strong active control forces
- To reduce DCR: need to distribute drift evenly across all floors
- These goals conflict when the building has a structural weak point

## Why Standard PPO Cannot Solve This

PPO requires:
1. Reward magnitudes in range -100 to +100 per step (max ±200k per 2000-step episode)
2. Explained variance > 0.8 (value function must fit returns)
3. Stable gradients (value loss < 100, policy gradient loss < 10)

Our problem violates all three:
1. Rewards range from -16M to +10k (160,000x range)
2. Explained variance: 0.008-0.029 (value function learns nothing)
3. Value loss: 73-114 million (catastrophic instability)

## Possible Solutions

### Option 1: Accept Higher DCR Targets (Easiest)
- Change M4.5 target from DCR=1.0-1.1 to DCR=1.8-2.2
- This acknowledges the weak floor reality
- Reduce DCR penalty magnitude by 100x:
  ```python
  dcr_penalty = -0.1 * ((current_dcr - 1.0) ** 2)
  if current_dcr > 2.5:  # Higher threshold
      dcr_penalty += -1.0 * (current_dcr - 2.5) ** 2
  # Scale by 0.01 instead of 0.1
  reward += 0.01 * dcr_penalty
  ```

### Option 2: Use Better Passive TMD Tuning (Recommended)
- Go back to k=40-50 kN/m (moderately tuned passive TMD)
- Use moderate active control (200-300 kN)
- Accept that improvement will be 10-20%, not 50%+
- This gives reasonable baseline DCR=1.5-1.8 instead of 28.93

### Option 3: Clipped DCR Penalty (Quick Fix)
- Cap the DCR penalty to prevent explosion:
  ```python
  dcr_penalty = np.clip(dcr_penalty, -100, 0)  # Cap at -100 per step
  ```
- This prevents -16M rewards but loses gradient information

### Option 4: Logarithmic Scaling (Advanced)
- Use log-scale rewards to compress range:
  ```python
  if current_dcr > 1.0:
      dcr_penalty = -10 * np.log(1 + (current_dcr - 1.0))
  ```
- Maintains gradient but compresses extreme values

### Option 5: Separate Displacement and DCR Training (Multi-Objective)
- Stage 1a: Train only for displacement (ignore DCR)
- Stage 1b: Fine-tune for DCR using learned displacement policy
- Requires curriculum within curriculum

## Recommended Next Step

**Option 2 is recommended**: Return to moderately tuned passive TMD (k=40-45 kN/m, c=1500-2000 N·s/m) combined with moderate active control (200-300 kN).

**Why this will work**:
1. Baseline DCR will be ~1.4-1.6 instead of 28.93
2. DCR penalty will be ~-50 per step instead of -8,000
3. Value function can learn (targets in -100k to +10k range)
4. Agent has room to improve (15-25% displacement reduction)
5. Realistic targets: 18-20 cm (vs 21 cm baseline) with DCR=1.3-1.5

**Revised realistic targets for M4.5**:
- Displacement: 18-20 cm (15-20% improvement from 21 cm baseline)
- ISDR: 0.8-1.2% (achievable with good control)
- DCR: 1.3-1.5 (acknowledging weak floor constraint)

This accepts the physical reality that a building with a weak 8th floor will have some drift concentration, and that's OK as long as it's below the safety limit of 1.75.
