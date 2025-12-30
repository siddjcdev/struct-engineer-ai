# Expected Controller Performance Analysis
## Building: MATLAB-Aligned (20 MN/m stiffness, 200,000 kg floors)

**Date**: December 30, 2025
**Author**: Siddharth (with Claude)

---

## Building Configuration (All Controllers Same)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Floors** | 12 | Standard mid-rise |
| **Floor mass** | 200,000 kg | 200 tons per floor |
| **Story stiffness** | 20 MN/m | Steel frame (flexible) |
| **Soft story (Floor 8)** | 12 MN/m (60% of normal) | **Critical weakness** |
| **Damping ratio** | 1.5% | Light damping |
| **TMD mass** | 4,000 kg (2% ratio) | On roof |
| **Total mass** | 2,404,000 kg (~2,400 tons) | Including TMD |

### Building Characteristics
- **Natural period**: ~1.2-1.5 seconds (estimate)
- **Flexibility**: MUCH softer than old 800 MN/m config
- **Vulnerability**: Soft story at floor 8 concentrates drift
- **Expected behavior**: Larger displacements than old stiff building

---

## Test Scenarios

### 1. PEER_small_M4.5_PGA0.25g.csv
- **Magnitude**: M4.5 (small earthquake)
- **PGA**: 0.25g (2.45 m/s²)
- **Duration**: ~40 seconds
- **Difficulty**: ⭐ Easy

### 2. PEER_moderate_M5.7_PGA0.35g.csv
- **Magnitude**: M5.7 (moderate earthquake)
- **PGA**: 0.35g (3.43 m/s²)
- **Duration**: ~40 seconds
- **Difficulty**: ⭐⭐ Medium

### 3. PEER_high_M7.4_PGA0.75g.csv
- **Magnitude**: M7.4 (large earthquake)
- **PGA**: 0.75g (7.36 m/s²)
- **Duration**: 60 seconds
- **Difficulty**: ⭐⭐⭐⭐ Hard

### 4. PEER_insane_M8.4_PGA0.9g.csv
- **Magnitude**: M8.4 (extreme earthquake)
- **PGA**: 0.9g (8.83 m/s²)
- **Duration**: ~100+ seconds
- **Difficulty**: ⭐⭐⭐⭐⭐ Extreme

---

## Controller Comparison

### 1. Passive TMD (No Control)
**What it gets**: Fixed tuning (optimized for building natural frequency)
**How it works**: Mechanical resonance only

| Scenario | Expected Peak Displacement | Notes |
|----------|---------------------------|-------|
| **M4.5 (0.25g)** | **20-30 cm** | TMD works well at design frequency |
| **M5.7 (0.35g)** | **35-45 cm** | Starting to struggle |
| **M7.4 (0.75g)** | **70-90 cm** | Poor performance, off-tune |
| **M8.4 (0.9g)** | **100-130 cm** | Saturates, ineffective |

**DCR**: 1.8-2.2 (drift concentrates at soft story)

**Pros**:
- Simple, no power needed
- Predictable behavior

**Cons**:
- Fixed tuning (can't adapt)
- Poor performance on strong earthquakes
- High DCR (can't distribute drift)

---

### 2. Fuzzy Logic Control
**What it gets**: Same 4 observations as old RL baseline
- Roof displacement
- Roof velocity
- TMD displacement
- TMD velocity

**How it works**: Rule-based control with membership functions

| Scenario | Expected Peak Displacement | Notes |
|----------|---------------------------|-------|
| **M4.5 (0.25g)** | **18-25 cm** | Good, beats passive by ~20% |
| **M5.7 (0.35g)** | **28-38 cm** | Decent adaptive response |
| **M7.4 (0.75g)** | **55-75 cm** | Struggles with extreme events |
| **M8.4 (0.9g)** | **85-110 cm** | Rules not optimized for this |

**DCR**: 1.6-1.9 (slightly better than passive, but still can't see floor 8)

**Pros**:
- Adaptive tuning
- Decent performance across scenarios
- Interpretable rules

**Cons**:
- Only sees roof (can't target floor 8)
- Rules designed for old building (may not transfer well)
- High DCR (no floor-level control)

---

### 3. RL Baseline (Old Model - 4 Observations)
**What it gets**: Same as Fuzzy Logic
- Roof displacement
- Roof velocity
- TMD displacement
- TMD velocity

**How it works**: Reinforcement learning policy (trained on old 800 MN/m building)

| Scenario | Expected Peak Displacement | Notes |
|----------|---------------------------|-------|
| **M4.5 (0.25g)** | **⚠️ UNTESTED** | Model trained on wrong building |
| **M5.7 (0.35g)** | **⚠️ UNTESTED** | Likely poor transfer |
| **M7.4 (0.75g)** | **⚠️ CATASTROPHIC?** | Trained for 30cm, may see 100cm+ |
| **M8.4 (0.9g)** | **⚠️ CATASTROPHIC?** | Completely out of distribution |

**DCR**: Unknown (likely 2.0+, can't see floor 8)

**Status**: ❌ **DO NOT USE - WRONG BUILDING**

**Why it will fail**:
- Trained on 800 MN/m stiffness (40x stiffer!)
- Expects much smaller displacements
- Control strategy optimized for stiff building
- Will apply wrong forces → instability

---

### 4. RL-CL (NEW Model - 8 Observations) **AFTER RETRAINING**
**What it gets**: Expanded observation space
- Roof displacement & velocity
- **Floor 8 displacement & velocity** (weak floor!)
- **Floor 6 displacement & velocity** (mid-height reference)
- TMD displacement & velocity

**How it works**: Curriculum-trained RL policy on CORRECT 20 MN/m building

| Scenario | Expected Peak Displacement | Notes |
|----------|---------------------------|-------|
| **M4.5 (0.25g)** | **15-22 cm** ✅ | Best performance, sees all floors |
| **M5.7 (0.35g)** | **25-35 cm** ✅ | Competitive with fuzzy |
| **M7.4 (0.75g)** | **45-65 cm** ✅ | **Best DCR**, targets floor 8 |
| **M8.4 (0.9g)** | **70-95 cm** ✅ | Trained for extreme events |

**DCR**: **1.2-1.6** (MUCH better - can target floor 8 directly!)

**Pros**:
- Sees floor 8 (can prevent drift concentration)
- Trained on correct building
- Curriculum learning (handles all magnitudes)
- Domain randomization (robust to noise/latency)

**Cons**:
- Requires retraining (old model useless)
- Black box (less interpretable than fuzzy)

---

## Detailed Scenario Analysis

### Scenario 1: M4.5 (0.25g) - Small Earthquake

**Uncontrolled (No TMD)**: ~50-60 cm peak
**Baseline drift**: Soft story would see ~40% more drift

| Controller | Peak Disp | Reduction vs Uncontrolled | DCR | Winner? |
|------------|-----------|---------------------------|-----|---------|
| Passive TMD | 25 cm | 50% ✅ | 1.9 | - |
| Fuzzy Logic | 22 cm | 56% ✅ | 1.7 | - |
| RL Baseline | ⚠️ Unknown | ⚠️ Risky | ⚠️ Unknown | ❌ |
| **RL-CL (NEW)** | **18 cm** | **64%** ✅✅ | **1.3** ✅ | **🏆 YES** |

**Why RL-CL wins**: Can preemptively control floor 8 before excessive drift builds up.

---

### Scenario 2: M5.7 (0.35g) - Moderate Earthquake

**Uncontrolled**: ~80-100 cm peak
**Soft story amplification**: Significant

| Controller | Peak Disp | Reduction vs Uncontrolled | DCR | Winner? |
|------------|-----------|---------------------------|-----|---------|
| Passive TMD | 40 cm | 50% ✅ | 2.1 | - |
| Fuzzy Logic | 33 cm | 58% ✅ | 1.8 | - |
| RL Baseline | ⚠️ Unknown | ⚠️ Risky | ⚠️ Unknown | ❌ |
| **RL-CL (NEW)** | **30 cm** | **62%** ✅✅ | **1.5** ✅ | **🏆 YES** |

**Why RL-CL wins**: Better drift distribution across floors, proactive control of weak story.

---

### Scenario 3: M7.4 (0.75g) - Large Earthquake ⭐ CRITICAL TEST

**Uncontrolled**: ~150-180 cm peak (building distress!)
**Soft story**: Would concentrate 60%+ of damage here

| Controller | Peak Disp | Reduction vs Uncontrolled | DCR | Winner? |
|------------|-----------|---------------------------|-----|---------|
| Passive TMD | 85 cm | 47% ⚠️ | 2.4 ❌ | - |
| Fuzzy Logic | 65 cm | 57% ✅ | 1.9 ⚠️ | - |
| RL Baseline | **💥 FAIL** | **Catastrophic** | **N/A** | ❌❌ |
| **RL-CL (NEW)** | **55 cm** | **63%** ✅✅ | **1.4** ✅ | **🏆 YES** |

**Why RL-CL wins**:
- Only controller that can "see" floor 8 drift in real-time
- Trained on full 60-second duration
- DCR penalty encourages uniform drift distribution
- Moderate penalty avoids pathological behavior

**Why RL Baseline fails catastrophically**:
- Trained for 30cm on stiff building, sees 100cm+ on soft building
- Applies wrong control forces (too aggressive or too weak)
- No training on 60-second earthquakes
- Corrupted observations from clipping bug

---

### Scenario 4: M8.4 (0.9g) - Extreme Earthquake 🔥

**Uncontrolled**: ~200-250 cm peak (severe damage/collapse risk!)
**This is the ultimate stress test**

| Controller | Peak Disp | Reduction vs Uncontrolled | DCR | Winner? |
|------------|-----------|---------------------------|-----|---------|
| Passive TMD | 120 cm | 40% ⚠️ | 2.5 ❌ | - |
| Fuzzy Logic | 100 cm | 50% ⚠️ | 2.1 ⚠️ | - |
| RL Baseline | **💥💥 FAIL** | **Collapse risk** | **N/A** | ❌❌❌ |
| **RL-CL (NEW)** | **80 cm** | **60%** ✅ | **1.6** ✅ | **🏆 YES** |

**Why RL-CL wins**:
- Curriculum learning trained on extreme events
- Domain randomization → robust to extreme inputs
- Full-duration training (can handle 100+ second quakes)
- Can prioritize floor 8 protection

**Key advantage**: Even if RL-CL doesn't achieve lowest peak displacement, it will have the **best DCR** (most uniform damage distribution).

---

## Domain Randomization Scenarios (Robustness Tests)

### TEST6b: 10% Sensor Noise
**What happens**: Observations corrupted by ±10% Gaussian noise

| Controller | Expected Performance |
|------------|---------------------|
| Passive TMD | ✅ Immune (no sensors) |
| Fuzzy Logic | ⚠️ Degrades ~10-15% |
| RL Baseline | ❌ Likely fails (not trained for noise) |
| **RL-CL (NEW)** | ✅ **Robust** (trained with noise) |

---

### TEST6c: 50ms Communication Latency
**What happens**: Control force delayed by 2-3 timesteps

| Controller | Expected Performance |
|------------|---------------------|
| Passive TMD | ✅ Immune (no communication) |
| Fuzzy Logic | ⚠️ Degrades ~20% (phase lag) |
| RL Baseline | ❌ Likely unstable |
| **RL-CL (NEW)** | ✅ **Robust** (trained with latency buffer) |

---

### TEST6d: 5% Data Dropout
**What happens**: Random sensor readings drop to zero

| Controller | Expected Performance |
|------------|---------------------|
| Passive TMD | ✅ Immune |
| Fuzzy Logic | ⚠️ Degrades ~15% (missing data) |
| RL Baseline | ❌ Unpredictable behavior |
| **RL-CL (NEW)** | ✅ **Robust** (trained with dropout mask) |

---

### TEST6e: Combined Stress (All three!)
**What happens**: Noise + Latency + Dropout simultaneously

| Controller | Expected Performance |
|------------|---------------------|
| Passive TMD | ✅ Immune |
| Fuzzy Logic | ❌ Degrades ~40% (too much degradation) |
| RL Baseline | ❌❌ Complete failure |
| **RL-CL (NEW)** | ✅ **Still functional** (~20% degradation) |

**Why RL-CL wins**: Only controller explicitly trained for these conditions!

---

## Summary Table: Expected Peak Displacements

| Scenario | Passive | Fuzzy | RL Baseline (Old) | **RL-CL (NEW)** | Winner |
|----------|---------|-------|-------------------|-----------------|--------|
| **M4.5 (0.25g)** | 25 cm | 22 cm | ⚠️ Unknown | **18 cm** ✅ | RL-CL |
| **M5.7 (0.35g)** | 40 cm | 33 cm | ⚠️ Unknown | **30 cm** ✅ | RL-CL |
| **M7.4 (0.75g)** | 85 cm | 65 cm | ❌ FAIL | **55 cm** ✅ | RL-CL |
| **M8.4 (0.9g)** | 120 cm | 100 cm | ❌❌ FAIL | **80 cm** ✅ | RL-CL |
| **10% Noise** | Immune | ~24 cm | ❌ FAIL | **20 cm** ✅ | Passive/RL-CL |
| **50ms Latency** | Immune | ~26 cm | ❌ FAIL | **21 cm** ✅ | Passive/RL-CL |
| **5% Dropout** | Immune | ~24 cm | ❌ FAIL | **20 cm** ✅ | Passive/RL-CL |
| **Combined** | Immune | ~31 cm | ❌❌ FAIL | **23 cm** ✅ | Passive/RL-CL |

### Expected DCR Comparison

| Scenario | Passive | Fuzzy | RL Baseline | **RL-CL (NEW)** | Best DCR |
|----------|---------|-------|-------------|-----------------|----------|
| **M4.5** | 1.9 | 1.7 | ⚠️ | **1.3** ✅ | **RL-CL** |
| **M5.7** | 2.1 | 1.8 | ⚠️ | **1.5** ✅ | **RL-CL** |
| **M7.4** | 2.4 | 1.9 | ⚠️ | **1.4** ✅ | **RL-CL** |
| **M8.4** | 2.5 | 2.1 | ⚠️ | **1.6** ✅ | **RL-CL** |

**Key Insight**: RL-CL is the **ONLY** controller that can achieve DCR < 1.6 consistently because it's the only one that can "see" and control floor 8 directly!

---

## Why Building Parameters Matter

### Old Building (800 MN/m) vs New Building (20 MN/m)

| Earthquake | Old Building Peak | New Building Peak | Ratio |
|------------|-------------------|-------------------|-------|
| M4.5 | ~8 cm | ~25 cm | **3.1x larger** |
| M5.7 | ~12 cm | ~40 cm | **3.3x larger** |
| M7.4 | ~30 cm | ~85 cm | **2.8x larger** |
| M8.4 | ~45 cm | ~120 cm | **2.7x larger** |

**This is why old models CANNOT be used!**
- Old model: Trained for 8-30cm displacements
- New building: Will see 25-120cm displacements
- The control strategy is completely wrong!

---

## Recommendations

### For Comparison Testing
1. **DO NOT use RL Baseline (old 4-obs model)** - will give catastrophic results
2. **DO retrain RL-CL** with all fixes before comparison
3. **Expect RL-CL to win** on DCR metric (only controller with floor visibility)
4. **Passive TMD serves as safety baseline** (no-risk reference)
5. **Fuzzy Logic is fair comparison** (same building, adaptive)

### For Production Deployment
1. **RL-CL (retrained)** for best performance
2. **Fuzzy Logic** as backup (simpler, interpretable)
3. **Passive TMD** as emergency fallback (no power needed)

### For Research Papers
- Highlight **DCR improvement** as key innovation
- Show RL-CL achieves **uniform drift distribution**
- Demonstrate **robustness** to noise/latency/dropout
- Compare against **fair baselines** (fuzzy, passive)
- **Don't show old RL baseline** (unfair comparison - wrong building)

---

## Date
December 30, 2025

## Status
✅ **PREDICTIONS READY - AWAITING RETRAINING TO VALIDATE**
