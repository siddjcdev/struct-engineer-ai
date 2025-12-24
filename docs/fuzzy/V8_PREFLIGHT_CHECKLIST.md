# ✅ V8 System Pre-Flight Checklist

Use this checklist before running your v8 simulations to ensure everything is properly configured.

---

## 🔧 Installation & Setup

### Python Environment

- [ ] Python 3.7+ installed
- [ ] Required packages installed:
  ```bash
  pip install fastapi uvicorn scikit-fuzzy numpy pydantic
  ```
- [ ] All Python files in project folder:
  - [ ] `main.py`
  - [ ] `models.py` (your existing file)
  - [ ] `test_fuzzy_controller.py` (optional)

### MATLAB Environment

- [ ] MATLAB R2019b or later
- [ ] All MATLAB files in project folder or path:
  - [ ] `RUN_ALL_6_TESTS_V8_WITH_FUZZY.m`
  - [ ] `thefunc_dcr_floor_tuner_v8_passive.m`
  - [ ] `thefunc_dcr_floor_tuner_v8_fuzzy.m`
  - [ ] `matlab_fuzzy_integration.m`
  - [ ] `thefunc_dcr_floor_tuner_v7.m` (your existing v7 function)

### Test Data

- [ ] `datasets/` folder exists
- [ ] All 6 test case CSV files exist:
  - [ ] `TEST1_stationary_wind_12ms.csv`
  - [ ] `TEST2_turbulent_wind_25ms.csv`
  - [ ] `TEST3_small_earthquake_M4.5.csv`
  - [ ] `TEST4_large_earthquake_M6.9.csv`
  - [ ] `TEST5_earthquake_M6.7.csv`
  - [ ] `TEST5_hurricane_wind_50ms.csv`
  - [ ] `TEST6a_baseline_clean.csv`
  - [ ] `TEST6b_with_10pct_noise.csv`

**If datasets missing:** Run `create_all_6_test_datasets.m`

---

## 🚀 Before Each Run

### 1. Start Python API

```bash
cd your_project_folder
python main.py
```

**Verify:**
- [ ] Terminal shows: "Uvicorn running on http://0.0.0.0:8001"
- [ ] No error messages
- [ ] Browser test: Open `http://localhost:8001/docs` (should show API documentation)

### 2. Test API Connection from MATLAB

```matlab
% In MATLAB:
fuzzy = FuzzyTMDController();
connected = fuzzy.check_connection();

if connected
    disp('✅ API connected!');
else
    disp('❌ API not responding');
end
```

**Result:**
- [ ] Shows "✅ Connected to Fuzzy Logic API"
- [ ] No errors

### 3. Verify Directories

```matlab
% Check that output directories exist
if ~exist('data', 'dir')
    mkdir('data');
end
if ~exist('data/fuzzy_outputs', 'dir')
    mkdir('data/fuzzy_outputs');
end
```

**Result:**
- [ ] Directories created with no errors

---

## 🧪 Quick System Test (2 minutes)

Run this minimal test to verify everything works:

```matlab
% Test passive TMD
fprintf('Testing passive TMD...\n');
thefunc_dcr_floor_tuner_v8_passive('el_centro', false);

% Test fuzzy TMD
fprintf('\nTesting fuzzy TMD...\n');
thefunc_dcr_floor_tuner_v8_fuzzy('el_centro', false);

% Check outputs
passive_files = dir('tmd_v8_passive_simulation_*.json');
fuzzy_files = dir('tmd_v8_fuzzy_simulation_*.json');

if ~isempty(passive_files)
    fprintf('✅ Passive TMD test passed\n');
else
    fprintf('❌ Passive TMD test failed\n');
end

if ~isempty(fuzzy_files)
    fprintf('✅ Fuzzy TMD test passed\n');
else
    fprintf('❌ Fuzzy TMD test failed\n');
end
```

**Expected result:**
- [ ] Both simulations complete without errors
- [ ] Two JSON files created
- [ ] Both tests show ✅

---

## 🐛 Common Issues & Solutions

### Issue: "Cannot connect to Python API"

**Checklist:**
- [ ] Python API is running (`python main.py`)
- [ ] No firewall blocking port 8001
- [ ] No other application using port 8001
- [ ] `matlab_fuzzy_integration.m` is in MATLAB path

**Fix:**
```bash
# Check if port is in use (Windows)
netstat -ano | findstr :8001

# Check if port is in use (Mac/Linux)
lsof -i :8001

# If occupied, kill the process or change port
```

### Issue: "FuzzyTMDController not found"

**Fix:**
```matlab
% Add to MATLAB path
addpath('path/to/matlab_fuzzy_integration.m');

% Or copy file to current directory
copyfile('path/to/matlab_fuzzy_integration.m', pwd);
```

### Issue: "Dataset files not found"

**Fix:**
```matlab
% Create datasets
create_all_6_test_datasets();

% Or manually check folder
cd datasets
ls  % List files
```

### Issue: JSON files not saving

**Checklist:**
- [ ] Write permissions in current directory
- [ ] Disk space available
- [ ] No file name conflicts

**Fix:**
```matlab
% Check write permissions
[status, msg] = fileattrib(pwd);
if status && msg.UserWrite
    disp('✅ Write permission OK');
else
    disp('❌ No write permission!');
end
```

### Issue: Simulations very slow

**Normal timing:**
- Single test (passive): 1-2 minutes
- Single test (fuzzy): 2-4 minutes
- Full suite (6 tests, both): 45-60 minutes

**If significantly slower:**
- [ ] Check CPU usage (MATLAB might be throttled)
- [ ] Check network latency (API on localhost should be <5ms)
- [ ] Restart MATLAB and Python API
- [ ] Close other applications

---

## 📊 Output File Verification

After a test run, verify these files exist:

### Passive TMD Output
```
✅ tmd_v8_passive_simulation_YYYYMMDD_HHMMSS.json
   ├─ version: "v8_passive"
   ├─ controller_type: "passive"
   ├─ baseline: {...}
   ├─ tmd_results: {...}
   └─ improvements: {...}
```

### Fuzzy TMD Output
```
✅ tmd_v8_fuzzy_simulation_YYYYMMDD_HHMMSS.json
   ├─ version: "v8_fuzzy"
   ├─ controller_type: "fuzzy_logic"
   ├─ baseline: {...}
   ├─ fuzzy_results: {...}
   ├─ improvements: {...}
   └─ time_series: {...}

✅ data/fuzzy_outputs/fuzzy_output_latest.json
✅ data/fuzzy_outputs/fuzzy_batch_YYYYMMDD_HHMMSS.json
```

**Verify JSON structure:**
```matlab
data = jsondecode(fileread('tmd_v8_fuzzy_simulation_20241121_143022.json'));

% Should have these fields:
assert(isfield(data, 'version'));
assert(isfield(data, 'controller_type'));
assert(isfield(data, 'baseline'));
assert(isfield(data, 'improvements'));
fprintf('✅ JSON structure valid\n');
```

---

## 🎯 Ready to Run!

Once all items are checked:

```matlab
% Start your full test suite
RUN_ALL_6_TESTS_V8_WITH_FUZZY()
```

**Recommended first run:** Option 3 (Quick demo - Both controllers)
- Time: ~16 minutes
- Tests: 6 scenarios
- Output: 12 JSON files (6 passive + 6 fuzzy)

---

## 📞 Final Check

Before starting your comprehensive test suite:

### System Status
- [ ] Python API running and responding
- [ ] MATLAB has all required files
- [ ] All datasets present
- [ ] Write permissions verified
- [ ] Quick test passed

### Time Planning
- [ ] Quick demo (option 3): ~16 minutes
- [ ] Comprehensive all (option 6): ~45-60 minutes
- [ ] Have sufficient time allocated

### Backup
- [ ] Previous results backed up (if any)
- [ ] Disk space available (need ~100 MB for all results)

### Environment
- [ ] Computer won't sleep/hibernate during run
- [ ] No planned system updates
- [ ] Stable internet connection (for API calls)

---

## ✅ All Set!

Everything checked? You're ready to run your V8 simulations with fuzzy logic control!

```matlab
% Let's go! 🚀
RUN_ALL_6_TESTS_V8_WITH_FUZZY()
```

---

## 📝 Notes Section

Use this space to track your test runs:

**Test Run 1:**
- Date: ___________
- Configuration: ___________
- Results: ___________
- Issues: ___________

**Test Run 2:**
- Date: ___________
- Configuration: ___________
- Results: ___________
- Issues: ___________
