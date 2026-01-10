# Final Recommendation - TMD Active Control Training
## Date: January 4, 2026

## Executive Summary

After extensive testing with pure passive, pure active, weak passive + strong active, and moderate passive + moderate active configurations, we've identified that **reward magnitudes are still too large for stable PPO training**, regardless of TMD tuning.

The root cause is that the DCR and ISDR penalties create rewards in the range of -400k to +10k per episode, which causes value function collapse (explained variance < 0.03, value loss > 70M).

## Current Configuration Status

**TMD Parameters**: k=50 kN/m, c=2 kN·s/m (near-optimal passive)
**Active Control**: 150 kN (light, for fine-tuning)
**Test Results**:
- Uncontrolled: 53.71 cm, DCR=1.63, reward=-438k
- Random control: 54.23 cm, DCR=1.58, reward=-201k (+237k improvement)
- PD control: 639.82 cm (unstable), DCR=1.75, reward=-523k

**Key Observation**: Uncontrolled baseline is consistently ~54 cm across all k values tested (10, 42, 48, 50 kN/m). This suggests either:
1. The synthetic test earthquake differs significantly from training earthquakes
2. The TMD passive dynamics aren't being applied correctly in baseline computation
3. The passive TMD has minimal effect on this specific test earthquake

## The Core Problem: Reward Scaling

The reward function creates per-step rewards ranging from -200 to +5, leading to cumulative rewards of -400k to +10k over 2000 steps. This causes:

1. **Value Loss Explosion**: 73-634 million (should be < 100)
2. **Explained Variance Collapse**: 0.008-0.029 (should be > 0.8)
3. **Training Instability**: Agent learns "do nothing" or makes things worse

**Example Calculation** (from hybrid control training):
- DCR = 28.93
- DCR penalty = -10*(28.93-1)² - 100*(28.93-1.75)² = -81,658 per step
- Scaled by 0.1 = -8,166 per step
- Over 2000 steps = -16.3 million cumulative reward

Even with better DCR (1.63 from k=50kN/m), the rewards are still -400k cumulative, which is 4x larger than PPO can handle.

## Recommended Solutions (In Order of Preference)

### Option 1: Reward Normalization (Quickest Fix)

**What to do**: Scale ALL reward components by 0.01 to bring episode rewards into -4k to +100 range.

**Implementation**:
```python
# In rl_cl_tmd_environment.py, line ~476
reward = 0.01 * (
    displacement_reward +
    velocity_penalty +
    force_penalty +
    smoothness_penalty +
    acceleration_penalty +
    0.01 * isdr_penalty +           # Already scaled, will become 0.0001x
    0.1 * dcr_penalty +             # Already scaled, will become 0.001x
    underutilization_penalty
)
```

**Pros**:
- Quick one-line fix
- Preserves relative reward weights
- Should allow value function to learn

**Cons**:
- Gradient magnitudes will be smaller (may need higher learning rate)
- Doesn't address fundamental reward design issues

---

### Option 2: Remove Uncontrolled Baseline Comparison (Fundamental Redesign)

**What to do**: Instead of rewarding improvement over baseline, reward absolute performance on normalized metrics.

**Implementation**:
```python
# Target-based reward (simpler, more stable)
displacement_reward = -abs(roof_disp) / 0.5  # Normalize by 50cm target
isdr_reward = -current_isdr / 0.02           # Normalize by 2% target
dcr_reward = -max(0, current_dcr - 1.0)      # Only penalize above 1.0

reward = displacement_reward + isdr_reward + dcr_reward
# Range: -3 to 0 per step, -6000 to 0 per episode (trainable!)
```

**Pros**:
- Eliminates baseline computation complexity
- Simpler reward function
- Guaranteed stable magnitudes

**Cons**:
- Loses relative improvement signal
- May not generalize across earthquake magnitudes
- Requires careful target tuning

---

### Option 3: Clipped Reward Components (Middle Ground)

**What to do**: Cap each reward component to prevent extreme values.

**Implementation**:
```python
# Clip each component before combining
displacement_reward = np.clip(displacement_reward, -5, +5)
isdr_penalty_scaled = np.clip(0.01 * isdr_penalty, -2, 0)
dcr_penalty_scaled = np.clip(0.1 * dcr_penalty, -10, 0)  # Cap at -10 even for catastrophic DCR

reward = displacement_reward + ... + dcr_penalty_scaled
# Range: -20 to +5 per step, -40k to +10k per episode
```

**Pros**:
- Prevents extreme penalties from dominating
- Maintains gradient up to clip threshold
- Moderate implementation effort

**Cons**:
- Loses information about truly catastrophic failures
- Arbitrary clip thresholds need tuning
- May not fully solve value function issues

---

### Option 4: Accept Current Targets as Unrealistic

**What to do**: Revise targets upward to match physical reality.

**New Targets**:
- M4.5: 19-21 cm (vs 14 cm original), 1.0-1.5% ISDR, 1.4-1.6 DCR
- M5.7: 25-28 cm (vs 22 cm original), 1.3-1.8% ISDR, 1.5-1.7 DCR
- M7.4: 32-38 cm (vs 30 cm original), 1.8-2.2% ISDR, 1.6-1.75 DCR
- M8.4: 42-48 cm (vs 40 cm original), 2.0-2.5% ISDR, 1.7-1.75 DCR

This acknowledges:
- Building has inherent weak floor (8th floor at 60% stiffness)
- TMD can only improve so much (15-25% is realistic, not 50%+)
- Some drift concentration is unavoidable with weak floor

---

## My Recommendation

**Implement Option 1 immediately** (0.01 global scaling):
1. It's a one-line change
2. Should fix value function learning
3. Can be tested within 30 minutes

**If Option 1 fails, try Option 2** (target-based rewards):
1. Cleaner reward design
2. Eliminates baseline computation issues
3. More predictable behavior

**If both fail, accept Option 4** (revised targets):
1. Acknowledges physical constraints
2. Reduces pressure on DCR/ISDR penalties
3. Allows successful training even if performance is modest

---

## Why Previous Approaches Failed

1. **Pure Passive (k=50 kN/m)**: Baseline already ~21 cm, no room for improvement
2. **Pure Active (k=0)**: Catastrophic DCR=25.61, agent learned "do nothing"
3. **Weak Passive (k=10 kN/m)**: Catastrophic DCR=28.93, -16M reward explosion
4. **Moderate Passive (k=42-48 kN/m)**: Baseline still ~54 cm (TMD dynamics issue?)
5. **Well-Tuned Passive (k=50 kN/m)**: Baseline still ~54 cm in test (synthetic earthquake?)

**Common thread**: All configurations had cumulative rewards outside PPO's stable range (-100k to +100k).

---

## Immediate Next Steps

1. Implement 0.01 global reward scaling in [rl_cl_tmd_environment.py:476](../../../restapi/rl_cl/rl_cl_tmd_environment.py#L476)
2. Update test script expected range to -4k to +100 for 2000-step episode
3. Re-run test_fixed_reward.py to verify reward magnitudes
4. If test passes, clean up old checkpoints and start training
5. Monitor TensorBoard for:
   - value_loss < 100 (not 70M+)
   - explained_variance > 0.7 (not 0.03)
   - episode_reward in -4k to +100 range (not -400k)

If training still fails with Option 1, we'll pivot to Option 2 (target-based rewards).
