# V12 Training Script Updates

## Date: January 11, 2026

## Summary

Updated `train_v12_soft_story.py` with proper dataset handling, comprehensive error handling, and held-out test set evaluation after each training stage.

## Key Changes

### 1. Dataset Path Handling

**Old Approach** (Hardcoded single files):
```python
train_files_map = {
    'M4.5': os.path.join(data_dir, 'PEER_small_M4.5_PGA0.25g.csv'),
    # ...
}
```

**New Approach** (Discovers training variants):
```python
def find_training_files(data_dir: str, magnitude: str) -> List[str]:
    """Find all training files for a given magnitude"""
    training_dir = os.path.join(data_dir, 'training', 'training_set_v2')
    # Finds TRAIN_M4.5_PGA0.25g_RMS0.073g_variant*.csv
    # Returns 10 files per magnitude for diverse training
```

**Benefits**:
- Uses proper training directory: `matlab/datasets/training/training_set_v2/`
- Discovers all 10 training variants automatically
- More diverse training data = better generalization

### 2. Held-Out Test Set Evaluation

**New Function**: `evaluate_on_test_set()`

After each training stage completes, the model is automatically evaluated on the held-out test file:

```python
# Example output:
==================================================================
  HELD-OUT TEST EVALUATION - M4.5
==================================================================
  Test file: PEER_small_M4.5_PGA0.25g.csv

  Test Results:
    Peak Displacement: 15.23 cm
    Max ISDR:          0.523%
    DCR:               1.18
    Peak Force:        287.4 kN
    Episode Reward:    -8.45

  Target Achievement:
    Displacement: 15.23 cm (target: 14 cm) ⚠️
    ISDR:         0.523% (target: 0.4%) ⚠️
    DCR:          1.18 (target: 1.15) ⚠️

  ⚠️  Close to targets - Very good performance
==================================================================
```

**Benefits**:
- Immediate feedback on generalization performance
- Detects overfitting early
- Shows if targets are achievable
- No need to run separate test script after training

### 3. Comprehensive Error Handling

**Added try-except blocks for**:

1. **Dataset Discovery**:
```python
try:
    train_files = find_training_files(data_dir, magnitude)
    test_file = find_test_file(data_dir, magnitude)
except FileNotFoundError as e:
    print(f"❌ ERROR: {e}")
    # Skip this stage and continue with next
```

2. **Environment Creation**:
```python
try:
    env = create_parallel_envs(train_files, force_limit, reward_scale, n_envs)
except Exception as e:
    print(f"❌ ERROR creating environments: {e}")
    traceback.print_exc()
```

3. **Training Loop**:
```python
try:
    model.learn(total_timesteps=timesteps, ...)
    print(f"✅ Training completed successfully!")
except Exception as e:
    print(f"❌ ERROR during training: {e}")
    traceback.print_exc()
```

4. **Model Saving**:
```python
try:
    model.save(final_path)
    print(f"✅ Model saved: {final_path}")
except Exception as e:
    print(f"❌ ERROR saving model: {e}")
```

5. **Test Evaluation**:
```python
try:
    # Run test episode
    metrics = env.get_episode_metrics()
    return metrics
except Exception as e:
    print(f"❌ ERROR during test evaluation: {e}")
    traceback.print_exc()
    return None
```

**Benefits**:
- Training continues even if one stage fails
- Clear error messages with full stack traces
- Graceful degradation (skip failed stages, continue with next)
- Easier debugging

### 4. Training Summary with Test Results

**Final Output**:
```
================================================================================
  TRAINING COMPLETE!
================================================================================

✅ Final model saved: models/v12_soft_story_tmd/final_model.zip

================================================================================
  HELD-OUT TEST SET SUMMARY
================================================================================

Magnitude    Disp (cm)    ISDR (%)     DCR        Status
----------------------------------------------------------------------
M4.5         15.23        0.523        1.18       ⚠️  CLOSE
----------------------------------------------------------------------

Targets: Displacement ≤ 14 cm, ISDR ≤ 0.4%, DCR ≤ 1.15

================================================================================
  NEXT STEPS
================================================================================

To test the model on all earthquakes:
  python test_v12_model.py --model-path models/v12_soft_story_tmd/final_model.zip

To monitor training metrics:
  tensorboard --logdir logs

Model checkpoints saved in: models/v12_soft_story_tmd
================================================================================
```

### 5. Improved Logging and Status Messages

**Throughout training**:
```
✅ Found 10 training files
✅ Found test file: PEER_small_M4.5_PGA0.25g.csv
✅ Training completed successfully!
✅ Model saved: models/v12_soft_story_tmd/stage1_M4.5_final.zip
✅ Test evaluation complete
```

**Error cases**:
```
❌ ERROR: Training directory not found
❌ ERROR creating environments: [detailed traceback]
⚠️  Test evaluation failed
⚠️  No model was trained (all stages failed or were skipped)
```

## New Utility Functions

### `find_training_files(data_dir, magnitude)`
- **Purpose**: Discover all training variants for a magnitude
- **Returns**: List of 10 CSV file paths
- **Example**: `['TRAIN_M4.5_...variant1.csv', ..., 'TRAIN_M4.5_...variant10.csv']`

### `find_test_file(data_dir, magnitude)`
- **Purpose**: Locate held-out test file for a magnitude
- **Returns**: Single CSV file path
- **Example**: `'matlab/datasets/test/PEER_small_M4.5_PGA0.25g.csv'`

### `evaluate_on_test_set(model, test_file, force_limit, magnitude)`
- **Purpose**: Evaluate trained model on held-out test earthquake
- **Returns**: Dict with metrics (displacement, ISDR, DCR, forces, reward)
- **Features**:
  - Deterministic policy evaluation
  - Target achievement analysis
  - Status indicators (✅/⚠️/❌)

## Usage

### Basic Training
```bash
cd c:\Dev\dAmpIng26\git\struct-engineer-ai\rl\rl_cl
python train_v12_soft_story.py
```

### Custom Configuration
```bash
python train_v12_soft_story.py \
  --run-name v12_breakthrough_attempt2 \
  --data-dir ../../matlab/datasets \
  --models-dir models \
  --logs-dir logs
```

### Resume Training
```bash
python train_v12_soft_story.py \
  --resume-from models/v12_soft_story_tmd/stage1_M4.5_500000_steps.zip
```

## Expected Training Flow

1. **Initialize**
   - Create output directories
   - Display configuration

2. **For Each Stage** (currently 1 stage: M4.5 @ 300kN):
   - Discover 10 training files
   - Find held-out test file
   - Create 4 parallel environments
   - Initialize or load PPO model
   - Train for 1.5M timesteps
   - Save final model
   - **Evaluate on held-out test set** ← NEW
   - Display test results
   - Store test metrics

3. **Finalize**
   - Save final model
   - Display test results summary ← NEW
   - Show next steps

## Error Recovery

The script is now resilient to various failures:

**Scenario 1: Missing training files**
```
❌ ERROR: No training files found for M4.5
   Searched: ../../matlab/datasets/training/training_set_v2/TRAIN_M4.5_*.csv
   Skipping stage 1
```
→ Script continues (doesn't crash)

**Scenario 2: Environment creation fails**
```
❌ ERROR creating environments: [detailed error]
[Full stack trace]
   Skipping stage 1
```
→ Script continues to next stage

**Scenario 3: Training crashes mid-way**
```
✅ Training completed successfully! (checkpoint at 500k steps)
❌ ERROR saving model: [error]
```
→ Can resume from checkpoint

**Scenario 4: Test evaluation fails**
```
⚠️  Test evaluation failed
```
→ Training results still saved, can test manually later

## Performance Monitoring

### During Training
```bash
tensorboard --logdir logs
```

**Key metrics to watch**:
- `metrics/max_isdr_percent` - Should drop below 0.5%
- `metrics/avg_peak_displacement_cm` - Should drop below 16 cm
- `rollout/ep_rew_mean` - Should converge to -5 to -10

### After Training

**Test Results** appear automatically in console:
- Peak displacement vs 14 cm target
- Max ISDR vs 0.4% target
- DCR vs 1.15 target
- Status indicators for quick assessment

**Comprehensive Testing**:
```bash
python test_v12_model.py --model-path models/v12_soft_story_tmd/final_model.zip
```

## File Structure

```
rl/rl_cl/
├── train_v12_soft_story.py       # Updated training script
├── test_v12_model.py             # Comprehensive test script
├── V12_SOFT_STORY_TMD.md         # Technical documentation
└── models/
    └── v12_soft_story_tmd/
        ├── stage1_M4.5_50000_steps.zip      # Checkpoint every 50k
        ├── stage1_M4.5_100000_steps.zip
        ├── ...
        ├── stage1_M4.5_final.zip            # Final stage model
        └── final_model.zip                   # Overall final model

matlab/datasets/
├── training/
│   └── training_set_v2/
│       ├── TRAIN_M4.5_PGA0.25g_RMS0.073g_variant1.csv
│       ├── TRAIN_M4.5_PGA0.25g_RMS0.073g_variant2.csv
│       ├── ... (10 total)
│       └── TRAIN_M4.5_PGA0.25g_RMS0.073g_variant10.csv
└── test/
    ├── PEER_small_M4.5_PGA0.25g.csv         # Held-out test
    ├── PEER_moderate_M5.7_PGA0.35g.csv
    ├── PEER_high_M7.4_PGA0.75g.csv
    └── PEER_insane_M8.4_PGA0.9g.csv
```

## Benefits Summary

1. **Better Training**: 10 diverse earthquake variants instead of 1
2. **Immediate Feedback**: Test results after each stage, no waiting
3. **Robustness**: Comprehensive error handling prevents crashes
4. **Transparency**: Clear status messages and error reporting
5. **Convenience**: Automatic test evaluation, no separate scripts needed
6. **Debugging**: Full stack traces on errors, easier to diagnose issues
7. **Monitoring**: Summary table of test results at the end

## Ready to Train!

The v12 training script is now production-ready with proper error handling, diverse training data, and automatic test evaluation.

```bash
python train_v12_soft_story.py --run-name v12_breakthrough
```

Expected training time: 12-24 hours for 1.5M timesteps @ 4 parallel environments.
