# Test v11 Model - Usage Guide

## Quick Start

### Basic Usage

Test your trained model on all earthquakes:

```bash
cd c:\Dev\dAmpIng26\git\struct-engineer-ai\rl\rl_cl
python test_v11_model.py --model-path models/rl_v11_advanced/final_1M_fixed_scale/stage1_150kN.zip
```

### Custom Test Directory

```bash
python test_v11_model.py \
  --model-path models/rl_v11_advanced/final_1M_fixed_scale/stage1_150kN.zip \
  --test-dir ../../matlab/datasets
```

### Custom Force Limit

```bash
python test_v11_model.py \
  --model-path models/rl_v11_advanced/final_1M_fixed_scale/stage1_150kN.zip \
  --force-limit 150000
```

## What You'll See

### Detailed Results Per Earthquake

```
======================================================================
  M4.5 TEST RESULTS
======================================================================

  Peak Roof Displacement:
    v11-Advanced: 14.23 cm
    v8-Baseline:  20.72 cm
    Δ from v8:    -6.49 cm 🏆 IMPROVED
    Uncontrolled: 21.02 cm
    Improvement:  +32.3% ✓

  Structural Safety:
    Max ISDR:     0.45%
                  ✅ (target: <0.5%, limit: <1.5%)
    DCR:          1.08
                  ✅ (target: ~1.0, limit: <1.75)

  Control Effort:
    Peak force:   142.3 kN
    Mean force:   68.5 kN
    RMS force:    74.2 kN

  Additional Metrics:
    RMS displacement: 4.82 cm
    Max drift:        5.12 cm
```

### Summary Table

```
======================================================================
  SUMMARY: v11 Advanced PPO Performance
======================================================================

Magnitude    v11 (cm)     v8 (cm)      Uncont (cm)    Improve    ISDR%      DCR
----------------------------------------------------------------------
M4.5         14.23        20.72        21.02          +32.3%     0.45       1.08
M5.7         28.54        46.45        46.02          +38.0%     0.68       1.12
M7.4         98.32        219.30       235.55         +58.3%     1.24       1.18
M8.4         187.45       363.36       357.06         +47.5%     2.15       1.42
----------------------------------------------------------------------

Average improvement over uncontrolled: 44.0%
```

### Target Achievement Analysis

```
======================================================================
  TARGET ACHIEVEMENT ANALYSIS
======================================================================

M4.5 Targets ("Almost No Structural Damage"):
  Target: 10-18 cm displacement, 0.3-0.5% ISDR, DCR ~1.0

  Displacement: 14.23 cm ✅ MET
  ISDR:         0.45% ✅ MET
  DCR:          1.08 ✅ MET

  Overall Assessment:
  ✅ ALL TARGETS MET - Excellent performance!
     'Almost no structural damage' achieved
```

## Output Interpretation

### Status Indicators

**Displacement Improvement**:
- `✓` - Better than uncontrolled
- `✗` - Worse than uncontrolled

**ISDR Status**:
- `✅` - < 0.5% (target met - "almost no damage")
- `⚠️` - 0.5-1.5% (acceptable - "minimal damage")
- `❌` - > 1.5% (excessive - needs improvement)

**DCR Status**:
- `✅` - < 1.1 (target met - nearly elastic)
- `⚠️` - 1.1-1.75 (acceptable - safe)
- `❌` - > 1.75 (safety limit exceeded)

### Target Achievement

**✅ ALL TARGETS MET**:
- Displacement: 10-18 cm ✓
- ISDR: ≤ 0.5% ✓
- DCR: ≤ 1.1 ✓
- **Action**: Celebrate! Training succeeded.

**⚠️ CLOSE TO TARGETS**:
- Displacement: ≤ 22 cm
- ISDR: ≤ 0.8%
- DCR: ≤ 1.3
- **Action**: Excellent performance. May indicate physical limits. Consider hardware upgrades if stricter targets needed.

**❌ TARGETS NOT MET**:
- Metrics significantly above targets
- **Action**: Check TensorBoard for convergence issues. Ensure training completed full 1M steps. May need more training time or hyperparameter tuning.

## Comparing Different Models

### Test Multiple Checkpoints

```bash
# Test different training stages
python test_v11_model.py --model-path models/rl_v11_advanced/run1/stage1_150kN.zip > results_run1.txt
python test_v11_model.py --model-path models/rl_v11_advanced/run2/stage1_150kN.zip > results_run2.txt
python test_v11_model.py --model-path models/rl_v11_advanced/run3/stage1_150kN.zip > results_run3.txt

# Compare results
diff results_run1.txt results_run2.txt
```

### Test at Different Training Milestones

If you saved intermediate checkpoints:

```bash
python test_v11_model.py --model-path models/rl_v11_advanced/checkpoint_200k.zip
python test_v11_model.py --model-path models/rl_v11_advanced/checkpoint_500k.zip
python test_v11_model.py --model-path models/rl_v11_advanced/checkpoint_1M.zip
```

This shows learning progression over time.

## Troubleshooting

### Error: Model file not found

```
❌ Error: Model file not found!
   Expected: models/rl_v11_advanced/final_1M_fixed_scale/stage1_150kN.zip
```

**Solution**: Check model path is correct. Use `ls` to find your model:

```bash
ls models/rl_v11_advanced/*/stage*.zip
```

### Error: Test file not found

```
⚠️  Skipping M4.5: File not found
   Expected: ../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv
```

**Solution**: Specify correct test directory:

```bash
python test_v11_model.py \
  --model-path <your-model> \
  --test-dir /path/to/your/datasets
```

### Unexpected Poor Performance

If results show worse than uncontrolled:

1. **Check reward_scale**: Model should have been trained with `reward_scale=1.0`
2. **Check force_limit**: Use same value as training (default: 150000)
3. **Verify training completed**: Check if training ran for full duration
4. **Check TensorBoard**: Look for convergence issues during training

## Advanced Usage

### Update Baseline Values

If you have different baseline performance, edit the script:

```python
# In test_v11_model.py, lines 22-35
UNCONTROLLED_BASELINES = {
    'M4.5': 21.02,   # Update with your values
    'M5.7': 46.02,
    'M7.4': 235.55,
    'M8.4': 357.06,
}

V8_BASELINES = {
    'M4.5': 20.72,   # Update with your v8 results
    # ...
}
```

### Test on Custom Earthquakes

Add to test_files dict:

```python
test_files = {
    'M4.5': 'path/to/earthquake1.csv',
    'Custom1': 'path/to/custom_earthquake.csv',
    # ...
}
```

## Example Workflow

### 1. Train Model

```bash
python train_v11.py --run-name final_1M_fixed_scale
# Wait for training to complete (1M steps)
```

### 2. Test Model

```bash
python test_v11_model.py --model-path models/rl_v11_advanced/final_1M_fixed_scale/stage1_150kN.zip
```

### 3. Analyze Results

Check output for:
- ✅ All targets met → Success!
- ⚠️ Close to targets → Excellent, consider hardware limits
- ❌ Targets not met → Check TensorBoard, may need more training

### 4. If Targets Not Met

**Option A**: Continue training
```bash
# Training will resume from checkpoint
python train_v11.py --run-name final_1M_fixed_scale
```

**Option B**: Try different hyperparameters
```python
# Edit ppo_config_v9_advanced.py
'timesteps': 2_000_000,  # Increase to 2M
```

**Option C**: Increase hardware limits
```python
# Edit ppo_config_v9_advanced.py
'force_limit': 200_000,  # Increase to 200 kN
```

## Success Criteria Reminder

**M4.5 Targets** ("Almost No Structural Damage"):
- Peak displacement: 10-18 cm
- Max ISDR: 0.3-0.5%
- DCR: ~1.0 (< 1.1)

**Acceptable Performance**:
- Peak displacement: < 22 cm
- Max ISDR: < 0.8%
- DCR: < 1.3

**Good Performance** (vs uncontrolled):
- Any improvement > 10%
- ISDR < 1.5%
- DCR < 1.75
