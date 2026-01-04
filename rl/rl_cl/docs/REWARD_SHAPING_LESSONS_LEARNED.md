# Reward Shaping Lessons Learned

## Summary

After extensive experimentation with reward shaping for TMD earthquake control, we discovered that **simpler is better**. The original reward function was correct, and our attempts to "improve" it with shaped rewards actually made learning worse.

## Journey Through Failed Approaches

### v1: Strong Displacement/Velocity Penalties + Force Direction Bonus
**Hypothesis**: Amplify displacement penalties and add force direction bonus to guide learning.

**Configuration**:
```python
reward = -10.0 * |disp| + -3.0 * |vel| + force_direction_bonus
```

**Results**:
- Peak displacement: 29.33 cm (39.5% WORSE than uncontrolled!)
- Force direction correctness: 99.2%
- **Problem**: Strong displacement penalties drowned out the force direction signal

### v2: Fixed Force Direction Logic
**Hypothesis**: Previous failure was due to wrong force direction (opposite vs same signs).

**Configuration**:
```python
# Changed from opposite signs to same signs due to Newton's 3rd law
if (vel > 0 and force > 0) or (vel < 0 and force < 0):
    bonus = +5.0  # Correct
```

**Results**:
- Peak displacement: 26.27 cm (25% WORSE)
- Force direction correctness: 0.6% (essentially random)
- **Problem**: Same issue - displacement penalties masked direction signal

### v3: Pure Force Direction Learning
**Hypothesis**: Remove ALL outcome penalties, learn only the action (force direction).

**Configuration**:
```python
reward = force_direction_bonus  # ONLY signal!
# Displacement/velocity penalties = 0
```

**Results**:
- Peak displacement: 24.94 cm (18.7% WORSE)
- Force direction correctness: 99.5% (agent learned perfectly!)
- **Critical Discovery**: Agent learned the "correct" direction but made things WORSE!

**Key Insight**:
- Constant +1.0 force → 18.38 cm (12.6% improvement) ✓
- Velocity-based switching → 24.94 cm (worse than uncontrolled) ✗

**This means velocity-based force direction switching actually PUMPS ENERGY into the system instead of dissipating it!** The timing and phase matter more than instantaneous direction.

### v4: Back to Basics (FINAL)
**Hypothesis**: Original simple reward was correct. Use it with curriculum learning and more timesteps.

**Configuration**:
```python
reward = -1.0 * |disp| + -0.3 * |vel|
# NO force direction bonus
# NO DCR penalty
# 4-stage curriculum learning
# 700K total timesteps
```

**Initial Results** (200K timesteps, single stage):
- Peak displacement: 20.28 cm (3.5% improvement) ✓
- Force direction correctness: 40.4%
- Mean force: 27,199 N
- **First actual improvement!**

## Key Lessons

### 1. Don't Encode Incorrect Physics Assumptions
We assumed "oppose velocity → reduce displacement" but this is oversimplified. Vibration damping involves:
- Phase relationships
- Timing
- Energy transfer dynamics
- TMD natural frequency interactions

Explicit reward shaping based on wrong assumptions **actively teaches harmful behavior**.

### 2. Reward Signal Strength Matters
When combining multiple reward components, their relative magnitudes determine what the agent actually learns:

| Component | Weight | Per-step | Over 1000 steps |
|-----------|--------|----------|-----------------|
| Displacement | -10.0 | -1.5 | -1500 |
| Velocity | -3.0 | -1.5 | -1500 |
| Force direction | +5.0 | +4.5 | +4500 |
| DCR | -2.0 | -4.5 | -4.5 (end only) |

The agent learns to optimize the **dominant** signal. Weak signals get ignored or create noise.

### 3. Outcome-Based vs Action-Based Rewards
- **Outcome-based** (displacement, velocity): High variance during exploration, hard to attribute to specific actions
- **Action-based** (force direction): Low variance, clear credit assignment, **but only if we know the correct action!**

We don't actually know the optimal force direction a priori (our assumptions were wrong), so action-based shaping failed.

### 4. DCR Penalty Conflicts with Displacement Minimization
- DCR penalty says: "Let other floors drift more to balance concentration"
- Displacement penalty says: "Minimize ALL displacement everywhere"

These are contradictory. DCR penalty is 666× weaker and gets ignored, but creates noise.

**Solution**: Remove DCR penalty. Good vibration control naturally produces good DCR (hypothesis confirmed: DCR=0.0 in all tests).

### 5. Curriculum Learning > Reward Shaping
Instead of trying to guide learning through complex rewards, use:
- **Progressive task difficulty** (M4.5 → M5.7 → M7.4 → M8.4)
- **Increasing force limits** (50kN → 100kN → 150kN)
- **More training time** (700K vs 200K timesteps)

Let the agent discover the right strategy through structured exploration.

## Final Recommendation

**Use the simplest reward that captures the objective**:

```python
reward = -1.0 * abs(roof_displacement) - 0.3 * abs(roof_velocity)
```

**Combined with**:
- 4-stage curriculum learning
- 700K total timesteps (150K + 150K + 200K + 200K)
- Standard SAC hyperparameters
- No observation bound tricks
- No complex reward shaping

This allows the agent to:
1. Learn basic control on easy earthquakes (M4.5)
2. Refine policy on moderate earthquakes (M5.7)
3. Handle extreme cases (M7.4, M8.4)
4. Discover the right control strategy through experience

## What We Thought vs What We Learned

| We Thought | We Learned |
|------------|-----------|
| Stronger signals → better learning | Stronger ≠ better; can drown out important signals |
| Guide with force direction bonus | Our physics intuition was wrong; hurt performance |
| Observation bounds were the issue | They weren't; reward design was |
| More complex reward → faster learning | Simpler reward + more time → better results |
| DCR needs explicit penalty | Good control → good DCR naturally |

## Files

- **Environment**: `tmd_environment_shaped_reward.py` (v4: simple reward, no DCR penalty)
- **Training**: `train_shaped_rewards.py` (v4: curriculum learning, 700K timesteps)
- **Documentation**: This file

## Status

Ready for final training run with v4 configuration.

Expected results:
- M4.5: 15-19 cm (10-25% improvement)
- M5.7: 35-45 cm
- M7.4: Better than current catastrophic failures
- M8.4: Better than current catastrophic failures

The agent will discover the optimal control through exploration rather than being told incorrect physics rules.
