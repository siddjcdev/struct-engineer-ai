# 🚀 QUICK START - 3 Steps to Use Fuzzy Logic Controller

## Step 1: Start Python API (30 seconds)

Open terminal/command prompt:

```bash
cd your_project_folder
python main.py
```

You should see:
```
======================================================================
TMD SIMULATION API WITH COMPREHENSIVE FUZZY LOGIC CONTROL
======================================================================
Fuzzy Controller: Active
...
INFO:     Uvicorn running on http://0.0.0.0:8001
```

✅ **Leave this terminal open!** The API must stay running.

---

## Step 2: Test from Browser (Optional - 1 minute)

Open browser: `http://localhost:8001/docs`

Click on:
1. `POST /fuzzylogic`
2. Click "Try it out"
3. Enter values:
   - displacement: `0.082`
   - velocity: `0.43`
   - acceleration: `3.48`
4. Click "Execute"

You should see control force output (~30 kN).

---

## Step 3: Use from MATLAB (2 minutes)

### Option A: Quick Test

```matlab
% Copy matlab_fuzzy_integration.m to your MATLAB folder
% Then in MATLAB:

fuzzy = FuzzyTMDController();
force = fuzzy.compute_single(0.082, 0.43, 3.48);
fprintf('Control force: %.0f N\n', force);
```

Expected output:
```
✅ Connected to Fuzzy Logic API at http://localhost:8001
Control force: -30625 N
```

### Option B: Full Integration

Add to your simulation:

```matlab
% Initialize once before simulation loop
fuzzy = FuzzyTMDController();

% In your simulation loop:
for i = 2:length(t)
    % Your existing code...
    x_current = x(i-1);
    v_current = v(i-1);
    
    % *** ADD THIS LINE ***
    F_fuzzy = fuzzy.compute_single(x_current, v_current);
    
    % Apply force to your building
    F_total = F_earthquake(i) + F_fuzzy;
    
    % Continue with your dynamics...
end
```

---

## 🎯 That's It!

You now have:
- ✅ Fuzzy logic controller running
- ✅ MATLAB integration working
- ✅ Automatic JSON output saving

---

## 📊 Example Output

Every computation creates a JSON file in `data/fuzzy_outputs/`:

```json
{
  "timestamp": "2024-11-21T14:30:45.123456",
  "computation_number": 1,
  "inputs": {
    "displacement_m": 0.082,
    "velocity_ms": 0.43
  },
  "output": {
    "control_force_N": -30625.5,
    "control_force_kN": -30.6255,
    "direction": "left (negative)"
  }
}
```

Latest result always in: `data/fuzzy_outputs/fuzzy_output_latest.json`

---

## 🔧 Common Issues

### "Connection refused"
→ Start Python API: `python main.py`

### "Module not found"
→ Install: `pip install fastapi uvicorn scikit-fuzzy numpy`

### "MATLAB can't find FuzzyTMDController"
→ Make sure `matlab_fuzzy_integration.m` is in MATLAB's path
→ Or cd to the folder containing it

---

## 📖 For More Details

- Complete guide: `README_FUZZY_CONTROLLER.md`
- Architecture: `ARCHITECTURE.md`
- Test script: `python test_fuzzy_controller.py`

---

## 🎉 Success Checklist

- [ ] Python API running on port 8001
- [ ] Browser test works at localhost:8001/docs
- [ ] MATLAB can connect: `fuzzy.check_connection()` returns true
- [ ] Single computation works
- [ ] Output files created in data/fuzzy_outputs/

**All checked?** You're ready to integrate with your TMD simulation! 🚀
