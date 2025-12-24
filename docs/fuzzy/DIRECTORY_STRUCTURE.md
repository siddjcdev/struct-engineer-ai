# 📁 V8 Project Directory Structure

This is the complete file structure for your V8 TMD simulation system.

```
your_project/
│
├── 📄 PYTHON FILES (REST API)
│   ├── main.py                                    ⭐ Primary API server
│   ├── models.py                                   (Your existing file)
│   ├── test_fuzzy_controller.py                    Quick test script
│   └── requirements.txt                            (Create this - see below)
│
├── 📄 MATLAB FILES (Simulation)
│   ├── RUN_ALL_6_TESTS_V8_WITH_FUZZY.m           ⭐ Main test runner
│   ├── thefunc_dcr_floor_tuner_v8_passive.m        V8 passive TMD
│   ├── thefunc_dcr_floor_tuner_v8_fuzzy.m        ⭐ V8 fuzzy TMD
│   ├── matlab_fuzzy_integration.m                ⭐ API wrapper class
│   ├── thefunc_dcr_floor_tuner_v7.m                (Your existing v7)
│   └── create_all_6_test_datasets.m                (Your existing script)
│
├── 📖 DOCUMENTATION
│   ├── V8_COMPLETE_SUMMARY.md                    ⭐ START HERE
│   ├── V8_FUZZY_LOGIC_GUIDE.md                     Complete usage guide
│   ├── V8_PREFLIGHT_CHECKLIST.md                   Pre-run checklist
│   ├── QUICK_START.md                              3-step quick start
│   ├── README_FUZZY_CONTROLLER.md                  Fuzzy controller docs
│   └── ARCHITECTURE.md                             System architecture
│
├── 📂 datasets/                                     Test input data
│   ├── TEST1_stationary_wind_12ms.csv
│   ├── TEST2_turbulent_wind_25ms.csv
│   ├── TEST3_small_earthquake_M4.5.csv
│   ├── TEST4_large_earthquake_M6.9.csv
│   ├── TEST5_earthquake_M6.7.csv
│   ├── TEST5_hurricane_wind_50ms.csv
│   ├── TEST6a_baseline_clean.csv
│   ├── TEST6b_with_10pct_noise.csv
│   ├── TEST6c_with_50ms_latency.csv
│   ├── TEST6d_with_5pct_dropout.csv
│   └── TEST6e_combined_stress.csv
│
├── 📂 data/                                         Runtime data
│   ├── simulation.json                              Latest MATLAB data
│   └── fuzzy_outputs/                               Fuzzy controller outputs
│       ├── fuzzy_output_latest.json                 Always latest result
│       ├── fuzzy_output_000001.json                 Individual outputs
│       ├── fuzzy_output_000002.json
│       └── fuzzy_batch_20241121_143045.json        Batch results
│
└── 📊 RESULTS (Generated after tests)
    ├── tmd_v8_passive_simulation_20241121_143022.json
    ├── tmd_v8_passive_simulation_20241121_143115.json
    ├── tmd_v8_fuzzy_simulation_20241121_143045.json
    ├── tmd_v8_fuzzy_simulation_20241121_143138.json
    └── ... (more result files)
```

---

## 🚀 Setup Instructions

### 1. Create requirements.txt

Create this file in your project root:

```txt
fastapi==0.104.1
uvicorn==0.24.0
scikit-fuzzy==0.4.2
numpy==1.24.3
pydantic==2.5.0
```

Then install:
```bash
pip install -r requirements.txt
```

### 2. Verify Directory Structure

Run this in MATLAB to create missing directories:

```matlab
% Create data directories
if ~exist('data', 'dir'), mkdir('data'); end
if ~exist('data/fuzzy_outputs', 'dir'), mkdir('data/fuzzy_outputs'); end

% Verify datasets exist
if ~exist('datasets', 'dir')
    fprintf('⚠️  datasets/ folder missing!\n');
    fprintf('   Run: create_all_6_test_datasets()\n');
end

fprintf('✅ Directory structure ready\n');
```

### 3. Add Files to MATLAB Path

```matlab
% Add current directory and subdirectories to path
addpath(genpath(pwd));
savepath;  % Save for future sessions

fprintf('✅ MATLAB path configured\n');
```

---

## 📝 File Descriptions

### ⭐ Critical Files (Must Have)

| File | Purpose | Type |
|------|---------|------|
| `main.py` | Python REST API with fuzzy controller | Python |
| `RUN_ALL_6_TESTS_V8_WITH_FUZZY.m` | Main test interface | MATLAB |
| `thefunc_dcr_floor_tuner_v8_fuzzy.m` | Fuzzy TMD simulation | MATLAB |
| `matlab_fuzzy_integration.m` | API wrapper class | MATLAB |
| `V8_COMPLETE_SUMMARY.md` | Getting started guide | Docs |

### 📄 Python Files

**main.py** (460 lines)
- FastAPI REST server
- Fuzzy logic controller (comprehensive)
- Simulation data API endpoints
- JSON output management

**test_fuzzy_controller.py** (130 lines)
- Quick validation script
- Tests fuzzy logic without MATLAB
- Verifies API functionality

### 📄 MATLAB Files

**RUN_ALL_6_TESTS_V8_WITH_FUZZY.m** (330 lines)
- Menu-driven test interface
- Options for passive/fuzzy/both
- Batch test execution
- Results comparison

**thefunc_dcr_floor_tuner_v8_passive.m** (75 lines)
- Wrapper for v7 passive TMD
- V8 naming conventions
- JSON metadata updates

**thefunc_dcr_floor_tuner_v8_fuzzy.m** (550 lines)
- Complete fuzzy TMD simulation
- API integration
- Newmark time integration
- Performance analysis
- Result visualization

**matlab_fuzzy_integration.m** (200 lines)
- FuzzyTMDController class
- API connection handling
- Single & batch computation
- Error handling
- Three complete examples

### 📖 Documentation Files

**V8_COMPLETE_SUMMARY.md**
- Project overview
- File descriptions
- Usage workflows
- Quick command reference

**V8_FUZZY_LOGIC_GUIDE.md**
- Complete usage guide
- Step-by-step tutorials
- Result analysis
- Troubleshooting

**V8_PREFLIGHT_CHECKLIST.md**
- Pre-run verification
- System requirements
- Common issues
- Quick tests

**QUICK_START.md**
- 3-step quickstart
- Minimal configuration
- Fast path to results

**README_FUZZY_CONTROLLER.md**
- Fuzzy controller details
- API documentation
- MATLAB integration
- Examples

**ARCHITECTURE.md**
- System architecture
- Data flow diagrams
- Component details
- Technical specs

---

## 🔄 Data Flow

```
1. START
   └─> python main.py                    (Start API)
   └─> RUN_ALL_6_TESTS_V8_WITH_FUZZY()  (Start tests)

2. FOR EACH TEST
   └─> Load earthquake data (datasets/)
   └─> Run baseline simulation
   └─> Extract displacement/velocity
   
3. IF FUZZY CONTROLLER
   └─> POST /fuzzylogic (batch)
   └─> Receive control forces
   └─> Apply forces to building
   └─> data/fuzzy_outputs/*.json  (Save API response)

4. ANALYZE RESULTS
   └─> Calculate improvements
   └─> Generate time series
   
5. SAVE RESULTS
   └─> tmd_v8_passive_*.json  (Passive results)
   └─> tmd_v8_fuzzy_*.json    (Fuzzy results)

6. END
```

---

## 💾 Disk Space Requirements

| Component | Size | Notes |
|-----------|------|-------|
| Python files | ~100 KB | Source code |
| MATLAB files | ~300 KB | Source code |
| Documentation | ~150 KB | Markdown files |
| Datasets | ~50 MB | CSV test data |
| Results (per test) | ~200 KB | JSON output |
| Fuzzy outputs | ~500 KB | API responses |
| **Total (with results)** | **~100 MB** | After all tests |

---

## 🔧 Configuration Files

### Optional: Create .gitignore

```
# Results
*.json
!models.json

# Python
__pycache__/
*.pyc
.venv/
venv/

# MATLAB
*.asv
*.mat

# Data
data/fuzzy_outputs/*
!data/fuzzy_outputs/.gitkeep

# OS
.DS_Store
Thumbs.db
```

### Optional: Create README.md (for Git)

```markdown
# V8 TMD Simulation System

Fuzzy Logic vs Passive TMD comparison across 6 test scenarios.

## Quick Start
1. `pip install -r requirements.txt`
2. `python main.py`
3. In MATLAB: `RUN_ALL_6_TESTS_V8_WITH_FUZZY()`

## Documentation
- [Complete Guide](V8_COMPLETE_SUMMARY.md)
- [Quick Start](QUICK_START.md)
- [Checklist](V8_PREFLIGHT_CHECKLIST.md)
```

---

## ✅ Verification

Run this to verify your setup:

```matlab
% Check all critical files exist
files_to_check = {
    'main.py'
    'RUN_ALL_6_TESTS_V8_WITH_FUZZY.m'
    'thefunc_dcr_floor_tuner_v8_fuzzy.m'
    'matlab_fuzzy_integration.m'
    'V8_COMPLETE_SUMMARY.md'
};

all_present = true;
for i = 1:length(files_to_check)
    if ~isfile(files_to_check{i})
        fprintf('❌ Missing: %s\n', files_to_check{i});
        all_present = false;
    else
        fprintf('✅ Found: %s\n', files_to_check{i});
    end
end

if all_present
    fprintf('\n✅ All critical files present!\n');
    fprintf('Ready to run: RUN_ALL_6_TESTS_V8_WITH_FUZZY()\n');
else
    fprintf('\n❌ Some files missing. Check list above.\n');
end
```

---

## 🎯 Ready!

Your V8 system is organized and ready to use.

**Next step:** Open `V8_COMPLETE_SUMMARY.md` and start testing!

```matlab
RUN_ALL_6_TESTS_V8_WITH_FUZZY()
```
