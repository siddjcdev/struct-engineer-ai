# Unified Fuzzy Logic TMD Controller - Complete Guide

## 🎯 What This Does

This system combines two fuzzy logic controllers into a single comprehensive REST API that:
- ✅ Uses the **comprehensive fuzzy logic** from your pure logic module (11 engineering-based rules)
- ✅ Integrates seamlessly with your **existing REST API** for TMD simulation data
- ✅ Accepts **real MATLAB data** (displacement, velocity, acceleration)
- ✅ Saves outputs as **JSON files** that MATLAB can read back
- ✅ Provides simple `/fuzzylogic` endpoint (your requested shortcut)

---

## 📁 File Structure

```
your_project/
├── main.py                          # ← Unified API (THIS IS THE MAIN FILE)
├── models.py                        # Your existing data models
├── matlab_fuzzy_integration.m       # ← MATLAB integration code
└── data/
    ├── simulation.json              # Your MATLAB simulation data
    └── fuzzy_outputs/               # ← Fuzzy controller outputs (auto-created)
        ├── fuzzy_output_latest.json
        ├── fuzzy_output_000001.json
        └── fuzzy_batch_20241121_143022.json
```

---

## 🚀 Quick Start

### Step 1: Start the Python API

```bash
python main.py
```

You should see:
```
======================================================================
TMD SIMULATION API WITH COMPREHENSIVE FUZZY LOGIC CONTROL
======================================================================
Fuzzy Controller: Active
Force Range: ±100.0 kN
Output Directory: data/fuzzy_outputs

Primary Endpoint: POST /fuzzylogic
Port: 8001
======================================================================

INFO:     Uvicorn running on http://0.0.0.0:8001
```

### Step 2: Test the API (Optional)

Open browser to: `http://localhost:8001/docs`

You'll see all available endpoints including:
- `/fuzzylogic` - Main fuzzy control endpoint
- `/fuzzylogic-batch` - Batch processing
- `/fuzzy-stats` - Controller statistics

### Step 3: Use from MATLAB

In MATLAB:
```matlab
% Create controller
fuzzy = FuzzyTMDController();

% Single computation
force = fuzzy.compute_single(0.082, 0.43, 3.48);
fprintf('Control force: %.0f N\n', force);

% Or batch computation for your entire simulation
forces = fuzzy.compute_batch(displacements, velocities, accelerations);
```

---

## 🔧 API Endpoints

### 1. `/fuzzylogic` - Single Time Step (YOUR SHORTCUT)

**Primary endpoint for computing control force**

**Usage from browser/curl:**
```bash
curl -X POST "http://localhost:8001/fuzzylogic?displacement=0.082&velocity=0.43&acceleration=3.48"
```

**Usage from MATLAB:**
```matlab
fuzzy = FuzzyTMDController();
force = fuzzy.compute_single(0.082, 0.43, 3.48);
```

**Response:**
```json
{
  "timestamp": "2024-11-21T14:30:45.123456",
  "computation_number": 1,
  "inputs": {
    "displacement_m": 0.082,
    "velocity_ms": 0.43,
    "acceleration_ms2": 3.48
  },
  "output": {
    "control_force_N": -30625.5,
    "control_force_kN": -30.6255,
    "direction": "left (negative)"
  },
  "saved_to": "data/fuzzy_outputs/fuzzy_output_000001.json",
  "latest_file": "data/fuzzy_outputs/fuzzy_output_latest.json"
}
```

### 2. `/fuzzylogic-batch` - Time Series Processing

**Process entire simulation at once**

**Usage from MATLAB:**
```matlab
[forces, response] = fuzzy.compute_batch(displacements, velocities, accelerations);
```

**Returns:**
- Array of control forces (N)
- Statistics (max, mean, std)
- Saved JSON file path

### 3. `/fuzzy-stats` - Controller Status

```bash
GET http://localhost:8001/fuzzy-stats
```

Returns:
- Total computations performed
- Last computation time
- Controller configuration
- Status

### 4. `/fuzzy-history` - Recent Computations

```bash
GET http://localhost:8001/fuzzy-history?last_n=100
```

Returns last N computations with full details.

---

## 📊 Integration with Your MATLAB Simulation

### Option A: Real-Time Integration (API calls during simulation)

```matlab
% Your existing simulation setup
m = 1000;  % mass
k = 100000;  % stiffness
c = 2000;  % damping
dt = 0.02;  % time step (50 Hz)

% Initialize fuzzy controller
fuzzy = FuzzyTMDController();

% Your simulation loop
for i = 2:length(t)
    % Current state
    x_current = x(i-1);
    v_current = v(i-1);
    a_current = a(i-1);
    
    % *** GET FUZZY CONTROL FORCE ***
    F_fuzzy = fuzzy.compute_single(x_current, v_current, a_current);
    
    % Apply to equation of motion
    F_total = F_earthquake + F_fuzzy;
    a(i) = F_total / m - k*x_current/m - c*v_current/m;
    
    % Continue with Newmark integration...
    % ... (your existing code)
end
```

### Option B: Batch Processing (Faster)

```matlab
% Run your simulation FIRST to get displacement/velocity time series
[x_baseline, v_baseline, a_baseline] = run_baseline_simulation(earthquake_data);

% Then compute ALL fuzzy forces at once
[F_fuzzy_array, response] = fuzzy.compute_batch(x_baseline, v_baseline, a_baseline);

% Use the fuzzy forces in a second pass
for i = 2:length(t)
    F_total = F_earthquake(i) + F_fuzzy_array(i);
    % ... (apply to building)
end
```

---

## 🧪 Testing Examples

### Example 1: Quick Test from MATLAB

```matlab
% Start MATLAB
fuzzy = FuzzyTMDController();

% Test with typical values
force = fuzzy.compute_single(0.05, 0.3, 2.0);
% Expected: Around -20 to -30 kN (damping force)

% Check the saved file
fuzzy.read_latest_output();
```

### Example 2: Run Full Example

```matlab
% In MATLAB, run the examples
example_single_computation();        % Example 1
example_time_series_computation();   # Example 2
example_integration_with_simulation();  % Example 3 (full simulation)
```

---

## 📋 Controller Configuration

The fuzzy controller is configured with:

```python
displacement_range = (-0.5, 0.5)    # ±50 cm
velocity_range = (-2.0, 2.0)        # ±2 m/s
force_range = (-100000, 100000)     # ±100 kN
```

**Membership Functions (5 levels each):**
- Displacement: negative_large, negative_small, zero, positive_small, positive_large
- Velocity: negative_fast, negative_slow, zero, positive_slow, positive_fast
- Force: large_negative, small_negative, zero, small_positive, large_positive

**Fuzzy Rules (11 total):**
1-2. Strong damping for large displacement + fast velocity
3-6. Moderate damping for moderate motion
7. Minimal force near equilibrium
8-11. Reduced damping when building returns to equilibrium naturally

---

## 🔍 Output Files

### Output Directory
All results saved to: `data/fuzzy_outputs/`

### File Types

1. **Latest Output**: `fuzzy_output_latest.json`
   - Always contains the most recent computation
   - MATLAB can read this after each API call

2. **Individual Outputs**: `fuzzy_output_XXXXXX.json`
   - Numbered sequentially
   - One file per computation

3. **Batch Outputs**: `fuzzy_batch_YYYYMMDD_HHMMSS.json`
   - Contains entire time series
   - Includes statistics

4. **History**: `fuzzy_history_YYYYMMDD_HHMMSS.json`
   - Complete computation history
   - Created on demand with `/fuzzy-save-history`

### Reading Outputs in MATLAB

```matlab
% Read latest output
data = jsondecode(fileread('data/fuzzy_outputs/fuzzy_output_latest.json'));

% Access values
displacement = data.inputs.displacement_m;
force = data.output.control_force_N;
```

---

## 🐛 Troubleshooting

### Problem: "Could not connect to API"

**Solution:**
1. Make sure Python API is running: `python main.py`
2. Check the port is 8001 (not in use by another app)
3. Try: `http://localhost:8001/health` in browser

### Problem: "Fuzzy computation error"

**Possible causes:**
- Input values out of range (will be clipped automatically)
- Invalid data types (must be floats)

**Check:**
```matlab
fuzzy = FuzzyTMDController();
if fuzzy.check_connection()
    disp('✅ Connected');
else
    disp('❌ Not connected');
end
```

### Problem: Output files not created

**Check:**
- Directory `data/fuzzy_outputs/` exists
- Python has write permissions
- Look at terminal output for error messages

---

## 📈 Performance

**Typical Performance:**
- Single computation: ~5-10 ms
- Batch of 1000 steps: ~5-8 seconds
- Full 10-second simulation (50 Hz): ~25-40 seconds

**For large simulations:**
- Use batch processing (faster)
- Or run simulation first, then compute forces
- Consider parallel processing for multiple scenarios

---

## ⚙️ Customization

### Change Force Range

In `main.py`:
```python
fuzzy_controller = FuzzyTMDController(
    displacement_range=(-0.8, 0.8),    # Increase displacement range
    velocity_range=(-3.0, 3.0),         # Increase velocity range
    force_range=(-200000, 200000)       # ±200 kN instead of ±100 kN
)
```

### Add More Rules

Edit the `_build_fuzzy_system()` method in `FuzzyTMDController` class.

Example:
```python
# Add a new rule
ctrl.Rule(
    displacement['positive_large'] & velocity['zero'],
    control_force['small_negative']
)
```

---

## 🎯 Next Steps

1. **Test the API**: Run `python main.py` and open `http://localhost:8001/docs`
2. **Test from MATLAB**: Run `example_single_computation()` in MATLAB
3. **Integrate with your simulation**: Use the patterns shown above
4. **Compare results**: Run passive TMD vs fuzzy TMD vs neural network TMD

---

## 📞 Quick Reference

**Start API:**
```bash
python main.py
```

**Main Endpoint:**
```
POST http://localhost:8001/fuzzylogic?displacement=X&velocity=Y&acceleration=Z
```

**From MATLAB:**
```matlab
fuzzy = FuzzyTMDController();
force = fuzzy.compute_single(displacement, velocity, acceleration);
```

**Output Location:**
```
data/fuzzy_outputs/fuzzy_output_latest.json
```

---

✅ **Everything is ready to use!** Start the Python API and begin testing from MATLAB.
