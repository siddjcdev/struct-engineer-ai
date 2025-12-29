# Robust RL Training Workflow - Proper Train/Test Split

## The Problem We're Solving

**Previous Issue**: Training and testing on the same CSV files
- Even with domain randomization, model could memorize earthquake waveforms
- Training on `PEER_small.csv` → Testing on `PEER_small.csv` (same file!)
- Result: Overfitting to specific time series

**New Approach**: Proper train/test split at the dataset level
- Training: NEW synthetic earthquakes (same magnitude, different waveforms)
- Testing: ORIGINAL PEER earthquakes (held-out, never seen during training)
- Domain randomization: Added noise/latency/dropout on top

## Workflow

### Step 1: Generate Synthetic Training Data

```bash
cd /home/user/struct-engineer-ai/matlab/datasets
python generate_training_earthquakes.py
```

**What this does:**
- Generates 3 variants each of M4.5, M5.7, M7.4, M8.4 earthquakes
- Uses Clough-Penzien stochastic ground motion model
- Matches target PGA but creates different waveforms
- Saves to `training_set/TRAIN_*.csv`
- Creates comparison plots to verify quality

**Output:**
```
training_set/
├── TRAIN_M4.5_PGA0.25g_variant1.csv
├── TRAIN_M4.5_PGA0.25g_variant2.csv
├── TRAIN_M4.5_PGA0.25g_variant3.csv
├── TRAIN_M5.7_PGA0.35g_variant1.csv
├── TRAIN_M5.7_PGA0.35g_variant2.csv
├── TRAIN_M5.7_PGA0.35g_variant3.csv
├── TRAIN_M7.4_PGA0.75g_variant1.csv
├── TRAIN_M7.4_PGA0.75g_variant2.csv
├── TRAIN_M7.4_PGA0.75g_variant3.csv
├── TRAIN_M8.4_PGA0.9g_variant1.csv
├── TRAIN_M8.4_PGA0.9g_variant2.csv
└── TRAIN_M8.4_PGA0.9g_variant3.csv
```

**Verification:** Check the generated comparison plots:
- `training_earthquakes_M4.5_comparison.png`
- `training_earthquakes_M5.7_comparison.png`
- `training_earthquakes_M7.4_comparison.png`
- `training_earthquakes_M8.4_comparison.png`

Ensure:
- ✓ PGA matches target (within ±5%)
- ✓ Frequency content similar to original
- ✓ Duration matches (~40s)
- ✓ Waveforms DIFFERENT from original (not memorizable)

### Step 2: Train with Proper Split + Domain Randomization

```bash
cd /home/user/struct-engineer-ai
python rl/rl_cl/train_final_robust_rl_cl.py
```

**What this does:**
- **Training data**: Randomly samples from synthetic variants
- **Each episode**: Random augmentation (noise/latency/dropout)
- **Curriculum**: M4.5@50kN → M5.7@100kN → M7.4@150kN → M8.4@150kN
- **After each stage**: Tests on held-out PEER data to verify generalization
- **Total training**: 700k timesteps (~3-5 hours)

**Training strategy:**
```python
Each episode:
1. Randomly pick: TRAIN_M4.5_variant1.csv OR variant2.csv OR variant3.csv
2. Add random sensor noise: 0-10%
3. Add random actuator noise: 0-5%
4. Add random latency: 0-40ms
5. Add random dropout: 0-8%
→ Model CANNOT memorize any specific pattern!
```

**Output:** `simple_rl_models/perfect_rl_final_robust.zip`

### Step 3: Deploy and Test

```bash
# Copy trained model to API
cp rl/rl_cl/simple_rl_models/perfect_rl_final_robust.zip \
   restapi/rl_cl/simple_rl_models/

# Update API to use new model
# (Restart API server if running)

# Run comprehensive tests on HELD-OUT PEER data
# This tests on earthquakes the model has NEVER seen!
```

## Why This Works

### Old Approach (Failed)
```
Training: PEER_small.csv + augmentation
Testing:  PEER_small.csv + noise/latency
Problem:  Same earthquake waveform → memorization possible
Result:   -237% on latency test
```

### New Approach (Robust)
```
Training: TRAIN_M4.5_variant1/2/3.csv + augmentation
Testing:  PEER_small.csv (completely different waveform)
Benefit:  Model CANNOT memorize test data
Result:   Expected >0% on all stress tests
```

## Expected Results

| Scenario | Old Model | Expected New Model |
|----------|-----------|-------------------|
| PEER_small | Good | Good ✓ |
| PEER_moderate | Good | Good ✓ |
| PEER_high | Good | Good ✓ |
| PEER_insane | Good | Good ✓ |
| **Mod-Latency** | **-237%** ✗ | **>0%** ✓ |
| 10% Noise | Degrades | >50% ✓ |
| 8% Dropout | <10% | >20% ✓ |
| Combined | ~20% | >30% ✓ |
| **DCR (PEER_high)** | 2.1 | 1.8-2.0 ✓ |

## Key Differences from Previous Training

| Aspect | Previous | New |
|--------|----------|-----|
| **Training data** | PEER_*.csv (4 files) | TRAIN_*.csv (12 files) |
| **Test data** | Same PEER_*.csv ✗ | Original PEER_*.csv ✓ |
| **Augmentation** | None → Added later | Yes (from start) |
| **Earthquake variants** | 1 per magnitude | 3 per magnitude |
| **Train = Test?** | YES ✗ | NO ✓ |
| **Can memorize?** | YES ✗ | NO ✓ |

## Scientific Validity

This approach follows proper machine learning practices:

1. ✓ **Train/test split**: Different datasets entirely
2. ✓ **Data augmentation**: Domain randomization during training
3. ✓ **Distribution matching**: Synthetic data has same statistical properties
4. ✓ **Held-out validation**: Test on data never seen during training
5. ✓ **Stress testing**: Evaluate robustness under adverse conditions

## Files Modified

- `rl/rl_cl/tmd_environment.py` - Added augmentation parameters
- `rl/rl_cl/train_robust_rl_cl.py` - Basic domain randomization (old)
- `rl/rl_cl/train_final_robust_rl_cl.py` - **Final training script (use this!)**
- `matlab/datasets/generate_training_earthquakes.py` - Synthetic data generation

## Troubleshooting

**Q: Training fails - "No training files found"**
A: Run `generate_training_earthquakes.py` first to create synthetic data

**Q: PGA doesn't match target in synthetic data**
A: Check comparison plots - should be within ±5%. Adjust if needed.

**Q: Model still fails stress tests**
A: Increase augmentation range or training timesteps

**Q: DCR still too high**
A: This is expected - DCR penalty is intentionally low (0.1) to prioritize robustness

## Next Steps After Training

1. Copy model to API directory
2. Restart API server
3. Run full comparison tests
4. Verify:
   - No catastrophic failures (>-100%)
   - Robust to latency/noise/dropout
   - DCR reasonable (<2.5)
   - Overall performance competitive with baseline

The model should now generalize to:
- Unseen earthquake waveforms ✓
- Real-world sensor/actuator imperfections ✓
- Communication delays ✓
- Signal dropouts ✓
