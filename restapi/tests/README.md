# API Test Suite

This directory contains test scripts for the TMD API endpoints.

## Available Tests

### 1. `test_rl_cl_simulate.py` - **NEW: Comprehensive Metrics Test**

Tests the `/rl-cl/simulate` endpoint that returns full simulation metrics.

**What it tests:**
- ✅ RMS roof displacement
- ✅ Peak roof displacement
- ✅ Maximum interstory drift
- ✅ DCR (Drift Concentration Ratio)
- ✅ Peak and mean control forces
- ✅ Force time series

**Usage:**
```bash
# Start the API server first (in another terminal)
cd restapi
python main.py

# Then run the test
cd restapi/tests
python test_rl_cl_simulate.py
```

**Test cases:**
1. Synthetic earthquake (quick validation)
2. TEST3 - Small Earthquake (M4.5)
3. TEST4 - Large Earthquake (M6.9)

**Expected output:**
```
📊 PERFORMANCE METRICS
─────────────────────────────────────────────────────────────────────────────
🏢 DISPLACEMENT METRICS:
   Peak roof displacement:     24.67 cm
   RMS roof displacement:       8.45 cm
   Maximum interstory drift:    3.21 cm

📐 DRIFT CONCENTRATION:
   DCR (Drift Concentration Ratio):   1.85
   Status: ✅ GOOD (acceptable drift distribution)

⚡ FORCE METRICS:
   Peak control force:        143.2 kN
   Mean control force:         85.3 kN
   Force efficiency:            1.17 %improvement/kN
```

### 2. `test_perfect_rl_api.py` - Basic RL-CL Tests

Tests basic prediction endpoints:
- `/health` - Health check
- `/info` - Model information
- `/predict` - Single prediction
- `/predict-batch` - Batch predictions

**Usage:**
```bash
cd restapi/tests
python test_perfect_rl_api.py
```

### 3. `test_fuzzy.py` - Fuzzy Controller Tests

Tests the fuzzy logic controller endpoints.

**Usage:**
```bash
cd restapi/tests
python test_fuzzy.py
```

## API Configuration

By default, tests connect to `http://localhost:8080`.

To test the deployed API, edit the test file:
```python
API_URL = "https://perfect-rl-api-887344515766.us-east4.run.app"
```

## Requirements

```bash
pip install requests numpy
```

## Troubleshooting

**Connection Error:**
- Make sure the API server is running: `python main.py` in the `restapi/` directory
- Check the API URL matches your server's address

**File Not Found:**
- `test_rl_cl_simulate.py` needs earthquake data from `matlab/datasets/`
- Make sure you're running from the `restapi/tests/` directory

**Timeout:**
- Simulations can take 30-60 seconds for large earthquakes
- The test has a 5-minute timeout by default

## Adding New Tests

To add a new test:

1. Create `test_<feature>.py` in this directory
2. Follow the pattern in existing tests
3. Import `requests` for API calls
4. Add clear print statements showing progress
5. Return `True` for success, `False` for failure
6. Update this README

## Test Results Documentation

For official test results and performance benchmarks, see:
- `/restapi/readme.md` - API documentation
- `/matlab/src/v1/` - MATLAB comparison scripts
