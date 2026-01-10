# Hybrid Control Configuration Test - January 4, 2026

## Problem Summary
After 150+ fixes, the agent was still producing results close to uncontrolled baseline:
- **Pure passive TMD (k=50 kN/m)**: 21.02 cm baseline, agent achieved 21.58 cm (WORSE!)
- **Pure active control (k=0)**: 20.90 cm baseline, agent achieved 20.93 cm with catastrophic DCR of 25.61

Root cause: Agent learned "do nothing" strategy to avoid making displacement worse.

## Solution: Hybrid Control

### Configuration
**Weak Passive TMD:**
- Stiffness: k = 10 kN/m (down from 50 kN/m)
- Damping: c = 500 N·s/m (down from 2 kN·s/m)
- Provides minimal baseline damping

**Strong Active Control:**
- Stage 1 (M4.5): 500 kN (up from 110 kN)
- Stage 2 (M5.7): 700 kN
- Stage 3 (M7.4): 900 kN
- Stage 4 (M8.4): 1000 kN

### Expected Baseline (M4.5)
From test_fixed_reward.py:
- Uncontrolled displacement: 54.07 cm
- Uncontrolled ISDR: 4.55%
- Uncontrolled DCR: 1.58
- Uncontrolled reward: -398,304

### Realistic Targets (M4.5)
With 500 kN active control:
- **Displacement**: 15-20 cm (65-72% improvement from 54 cm)
- **ISDR**: 1.5-2.5% (50-67% improvement from 4.55%)
- **DCR**: 1.2-1.4 (acceptable, below 1.75 safety limit)
- **Reward**: -50k to +10k range

### Reward Function Features

1. **Displacement Reward (Primary)**:
   - Based on improvement ratio vs uncontrolled baseline
   - 5.0 * improvement_ratio per step
   - Dominant signal to drive learning

2. **ISDR Penalty (Continuous)**:
   - Base: -200 * ISDR²
   - Above 1.2%: additional -500 * excess²
   - Scaled by 0.01 in final reward

3. **DCR Penalty (Continuous)**:
   - Base: -10 * (DCR - 1.0)²
   - Above 1.75: additional -100 * excess²
   - Scaled by 0.1 in final reward

4. **Underutilization Penalty (NEW)**:
   - Triggers when DCR > 5.0 AND force usage < 10%
   - Penalty: -50 * (0.10 - utilization)
   - Prevents "do nothing" strategy

### Test Results

**Uncontrolled (zero force):**
- Displacement: 54.07 cm
- ISDR: 4.55%
- DCR: 1.58
- Reward: -398,304

**Random Control (±50% of 500 kN):**
- Displacement: 53.88 cm (0.35% improvement)
- ISDR: 4.79%
- DCR: 1.70
- Mean force: 126.9 kN (25% utilization)
- Reward: -51,699 (+346k improvement!)

**Conclusion**: Reward function correctly incentivizes control. Random control gets 7.7x better reward despite minimal displacement improvement, showing the function rewards force utilization when beneficial.

## Training Plan

### Stage 1: M4.5 Earthquakes
- Timesteps: 200,000
- Force limit: 500 kN
- n_steps: 2048
- batch_size: 64
- n_epochs: 10
- ent_coef: 0.05
- clip_range: 0.2
- clip_range_vf: None

### Success Criteria

**Stage 1 Pass Criteria:**
- Episode reward > -100k (vs -398k uncontrolled)
- Peak displacement < 25 cm (vs 54 cm uncontrolled)
- Max ISDR < 2.5% (vs 4.55% uncontrolled)
- DCR < 1.75 (safety limit)
- Force utilization > 10% (avoid "do nothing")

**TensorBoard Metrics to Watch:**
- `rollout/ep_rew_mean` should increase from ~-400k toward -50k to +10k
- `ppo/value_loss` should be stable < 100
- `ppo/explained_variance` should reach > 0.8
- `ppo/entropy_loss` should decay slowly
- Custom metrics: `peak_displacement`, `max_isdr_percent`, `DCR`, `mean_force`

### What Could Still Go Wrong

1. **Force saturation at 500 kN**: If agent hits force limits frequently, may need to increase to 700-800 kN
2. **DCR penalty dominates too much**: May need to reduce 0.1 scaling to 0.05
3. **Displacement reward too weak**: May need to increase from 5.0 to 7.0-10.0
4. **Building resonance**: 0.196 Hz natural frequency may still resonate with earthquake frequencies

### Next Steps After Stage 1

If Stage 1 succeeds:
1. Continue to Stage 2 (M5.7) with 700 kN
2. Monitor for force saturation
3. Adjust ISDR penalty scaling if needed (0.01 → 0.02 for stronger safety focus)

If Stage 1 fails:
1. Check TensorBoard for gradient issues (value loss, explained variance)
2. Verify force utilization is > 10%
3. Consider increasing displacement reward weight
4. Consider reducing DCR penalty scaling
5. May need to accept that 54 cm → 15 cm (72% improvement) is unrealistic
