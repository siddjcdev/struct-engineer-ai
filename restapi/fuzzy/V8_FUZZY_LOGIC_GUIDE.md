# V8 TMD Simulation with Fuzzy Logic - Complete Guide

## 🎯 What's New in V8

V8 adds **Fuzzy Logic Controller** integration to your TMD simulation system:

- ✅ Test **Passive TMD** (mechanical, no control)
- ✅ Test **Fuzzy Logic TMD** (active control via AI)
- ✅ **Compare both** side-by-side
- ✅ Run on all **6 test cases**
- ✅ Automatic performance analysis

---

## 🚀 Quick Start (5 minutes)

### Step 1: Start Python API

```bash
cd your_project_folder
python main.py
```

Leave this running! You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8001
```

### Step 2: Run V8 Tests in MATLAB

```matlab
% In MATLAB:
RUN_ALL_6_TESTS_V8_WITH_FUZZY()
```

### Step 3: Select What to Run

You'll see a menu:
```
Options:
  1. Quick demo - Passive TMD only (~6 min)
  2. Quick demo - Fuzzy Logic only (~10 min)
  3. Quick demo - Both (comparison) (~16 min)
  4. Run all comprehensive - Passive only (~15 min)
  5. Run all comprehensive - Fuzzy only (~30 min)
  6. Run all comprehensive - Both (~45 min)
  7. Run specific test case
  8. Compare Passive vs Fuzzy on specific test
```

**Recommended:** Start with option 3 (both controllers on quick demo)

---

## 📊 What Gets Created

### Output Files

After running, you'll have JSON files:

```
your_project/
├── tmd_v8_passive_simulation_YYYYMMDD_HHMMSS.json    ← Passive TMD results
├── tmd_v8_fuzzy_simulation_YYYYMMDD_HHMMSS.json      ← Fuzzy Logic results
└── data/
    └── fuzzy_outputs/
        ├── fuzzy_output_latest.json                   ← Latest API response
        └── fuzzy_batch_YYYYMMDD_HHMMSS.json          ← All control forces
```

### JSON Structure

Both files contain:
```json
{
  "version": "v8_passive" or "v8_fuzzy",
  "controller_type": "passive" or "fuzzy_logic",
  "baseline": {
    "DCR": 0.85,
    "max_drift": 0.042,
    "max_roof": 0.23
  },
  "improvements": {
    "dcr_reduction_pct": 45.2,
    "drift_reduction_pct": 38.7,
    "roof_reduction_pct": 41.3
  },
  "time_series": {
    "time": [...],
    "earthquake_acceleration": [...],
    "baseline_roof": [...],
    "fuzzy_roof": [...]  // (only in fuzzy file)
  }
}
```

---

## 🧪 Example Workflows

### Workflow 1: Quick Comparison Test

Test one scenario with both controllers:

```matlab
% Run the test runner
RUN_ALL_6_TESTS_V8_WITH_FUZZY()

% Choose option 8 (compare on specific test)
% Select test 4 (Large Earthquake)
```

**Result:** Two JSON files comparing passive vs fuzzy performance

### Workflow 2: Test All Cases with Fuzzy Logic

Run all 6 test cases using fuzzy control:

```matlab
RUN_ALL_6_TESTS_V8_WITH_FUZZY()

% Choose option 5 (comprehensive fuzzy only)
```

**Time:** ~30 minutes
**Result:** 10+ JSON files (6 main tests + stress tests)

### Workflow 3: Generate Full Comparison Report

Run everything for your paper:

```matlab
RUN_ALL_6_TESTS_V8_WITH_FUZZY()

% Choose option 6 (comprehensive both)
```

**Time:** ~45 minutes
**Result:** Complete dataset comparing passive vs fuzzy on all scenarios

---

## 📈 Understanding the Results

### Performance Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **DCR** | Demand-to-Capacity Ratio | < 1.0 (safe) |
| **Max Drift** | Maximum inter-story displacement | < 2% story height |
| **Max Roof** | Maximum roof displacement | Lower is better |
| **RMS Displacement** | Root-mean-square displacement | Comfort metric |

### Improvement Percentages

Positive percentage = Fuzzy is better than baseline

Example:
- `dcr_reduction_pct: 45.2` → Fuzzy reduced DCR by 45.2%
- `drift_reduction_pct: 38.7` → Fuzzy reduced drift by 38.7%

### Typical Results

**Passive TMD:**
- DCR reduction: 30-40%
- Drift reduction: 25-35%
- No energy consumption

**Fuzzy Logic TMD:**
- DCR reduction: 40-55%
- Drift reduction: 35-50%
- Active control (requires power)
- Max control force: 20-50 kN

---

## 🔧 How It Works

### Passive TMD Flow

```
1. Load earthquake data
2. Optimize TMD parameters (mass, stiffness, damping)
3. Run simulation with passive TMD
4. Calculate performance
5. Save results
```

### Fuzzy Logic TMD Flow

```
1. Load earthquake data
2. Run baseline simulation (no control)
   ↓
3. Extract displacement & velocity
   ↓
4. Send to Python API → Fuzzy Controller
   ↓
5. Receive control forces
   ↓
6. Re-run simulation WITH control forces
   ↓
7. Compare baseline vs controlled
   ↓
8. Save results
```

### The Fuzzy Controller

**Inputs:**
- Inter-story drift (displacement)
- Inter-story drift velocity

**Processing:**
- 5 membership functions per input
- 11 fuzzy rules
- Engineering-based logic

**Output:**
- Control force (Newtons)
- Direction: positive = push right, negative = pull left

---

## 🐛 Troubleshooting

### Problem: "Cannot connect to Python API"

**Solution:**
```bash
# Start the API
python main.py

# Verify it's running
# Open browser: http://localhost:8001/docs
```

### Problem: "FuzzyTMDController not found"

**Solution:**
```matlab
% Add matlab_fuzzy_integration.m to path
addpath('path/to/your/files')

% Or copy file to current directory
```

### Problem: Fuzzy simulation very slow

**Explanation:** Each API call takes ~5-10ms
- For 500 time steps: ~2.5-5 seconds
- This is normal!

**Speed it up:**
- Use batch mode (already implemented in v8)
- Reduce time steps (use coarser dt)

### Problem: Different results each run

**Cause:** Random noise in earthquake data or initial conditions

**Solution:** For reproducible results, set random seed:
```matlab
rng(42);  % Before running simulation
```

---

## 📊 Analyzing Results in MATLAB

### Load and Compare JSON Files

```matlab
% Read both result files
passive = jsondecode(fileread('tmd_v8_passive_simulation_20241121_143022.json'));
fuzzy = jsondecode(fileread('tmd_v8_fuzzy_simulation_20241121_143045.json'));

% Compare DCR
fprintf('Passive DCR: %.2f\n', passive.baseline.DCR);
fprintf('Fuzzy DCR: %.2f\n', fuzzy.fuzzy_results.DCR);

% Plot comparison
figure;
bar([passive.baseline.DCR, fuzzy.baseline.DCR; ...
     passive.tmd_results.DCR, fuzzy.fuzzy_results.DCR]);
set(gca, 'XTickLabel', {'Baseline', 'With Control'});
legend('Passive', 'Fuzzy');
ylabel('DCR');
title('Performance Comparison');
```

### Extract Control Forces

```matlab
% Load fuzzy results
data = jsondecode(fileread('tmd_v8_fuzzy_simulation_20241121_143045.json'));

% Plot control forces
figure;
plot(data.time_series.time, data.time_series.control_force/1000);
xlabel('Time (s)');
ylabel('Control Force (kN)');
title('Fuzzy Logic Control Force Time History');
grid on;
```

### Calculate Energy Consumption

```matlab
% Fuzzy controller energy
forces = data.time_series.control_force;
dt = data.time_series.time(2) - data.time_series.time(1);

% Energy = ∫|F|·|v| dt (simplified)
% Actual calculation would need velocity data
power_estimate = mean(abs(forces)) / 1000;  % Average kW
fprintf('Estimated average power: %.1f kW\n', power_estimate);
```

---

## 🎓 For Your Research Paper

### Recommended Testing Sequence

1. **Quick validation:** Option 3 (both on quick demo)
2. **Full dataset:** Option 6 (comprehensive both)
3. **Specific scenarios:** Option 8 (targeted comparisons)

### Figures to Generate

1. **Time history comparison** (baseline vs passive vs fuzzy)
2. **Performance bar charts** (DCR, drift, roof displacement)
3. **Control force time history** (fuzzy only)
4. **Energy comparison** (passive vs fuzzy)
5. **Robustness analysis** (stress tests)

### Tables for Paper

**Table 1: Test Scenarios**
- Test 1-6 descriptions
- Earthquake magnitude
- Wind speeds
- Expected PGA

**Table 2: Performance Results**
- All metrics for passive
- All metrics for fuzzy
- Improvement percentages

**Table 3: Computational Cost**
- Simulation time
- API calls
- Memory usage

---

## ⚙️ Advanced Configuration

### Modify Fuzzy Controller Parameters

In `main.py`:
```python
fuzzy_controller = FuzzyTMDController(
    displacement_range=(-0.8, 0.8),    # Increase if needed
    velocity_range=(-3.0, 3.0),         # Increase if needed
    force_range=(-200000, 200000)       # ±200 kN
)
```

### Change Control Floor

In `thefunc_dcr_floor_tuner_v8_fuzzy.m`, line ~85:
```matlab
critical_floor = building.n_floors;  % Use roof

% Or use a specific floor:
critical_floor = 7;  % Apply control at 7th floor
```

### Modify Building Parameters

In `thefunc_dcr_floor_tuner_v8_fuzzy.m`, around line 180:
```matlab
building.n_floors = 15;  % Increase to 15 floors
building.m = ones(building.n_floors, 1) * 120000;  % Heavier
building.k = ones(building.n_floors, 1) * 60e6;     # Stiffer
```

---

## 📞 Quick Reference

**Start API:**
```bash
python main.py
```

**Run V8 tests:**
```matlab
RUN_ALL_6_TESTS_V8_WITH_FUZZY()
```

**Files to check:**
- Results: `tmd_v8_*_simulation_*.json`
- Fuzzy API data: `data/fuzzy_outputs/`
- Plots: Auto-generated if no output arguments

**Key functions:**
- `thefunc_dcr_floor_tuner_v8_passive()` - Passive TMD
- `thefunc_dcr_floor_tuner_v8_fuzzy()` - Fuzzy Logic TMD
- `FuzzyTMDController()` - MATLAB API wrapper

---

## 🎉 You're Ready!

Start with:
```matlab
RUN_ALL_6_TESTS_V8_WITH_FUZZY()
% Choose option 3
```

This will run both passive and fuzzy on a quick demo (~16 minutes) so you can see how everything works before running your full test suite.

Good luck with your research! 🚀
