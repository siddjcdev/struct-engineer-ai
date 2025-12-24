# PERFECT RL API - QUICK START

**Get your champion model running in 3 steps!**

---

## 🎯 **WHAT YOU'RE DEPLOYING**

```
🏆 Champion Perfect RL Model

Performance:
✅ TEST3 (M4.5): 24.67 cm (21.8% vs passive)
✅ TEST4 (M6.9): 20.80 cm (32% vs passive)
✅ Average: ~21.5 cm across all scenarios
✅ Beats fuzzy logic by 14%
✅ Exceptional robustness

Status: Production Ready!
```

---

## 🚀 **3-STEP QUICK START**

### **Step 1: Start API (1 command)**

```powershell
python perfect_rl_api.py
```

**Expected output:**
```
======================================================================
  PERFECT RL API SERVER
======================================================================

🏆 Champion model: Beats fuzzy logic by 14%
   Performance: 24.67 cm (TEST3), 20.80 cm (TEST4)

📊 Endpoints:
   POST /predict        - Single prediction
   POST /predict-batch  - Batch prediction
   GET  /health         - Health check
   GET  /info           - Model info
   GET  /docs           - Interactive docs

🚀 Starting server...
   URL: http://localhost:8000
   Docs: http://localhost:8000/docs
   Health: http://localhost:8000/health
```

---

### **Step 2: Test API (1 command)**

Open a new terminal:

```powershell
python test_perfect_rl_api.py
```

**Expected output:**
```
======================================================================
  PERFECT RL API TEST SUITE
======================================================================

1. TESTING HEALTH ENDPOINT
✅ Status: healthy
   Model loaded: True

2. TESTING INFO ENDPOINT
✅ Model: Perfect RL (Champion)

3. TESTING SINGLE PREDICTION
✅ Single prediction successful
   Force: -52340.50 N (-52.34 kN)

4. TESTING BATCH PREDICTION
✅ Batch prediction successful

5. TESTING PERFORMANCE
✅ Performance test complete
   Mean time: 8.5 ms

======================================================================
  ✅ ALL TESTS PASSED!
======================================================================
```

---

### **Step 3: Use in MATLAB (copy-paste)**

```matlab
% Add functions to path
addpath('path/to/perfect_rl_matlab.m');

% API URL
API_URL = 'http://localhost:8000';

% Health check
status = perfect_rl_health(API_URL);

% Single prediction
force = perfect_rl_predict(API_URL, 0.15, 0.8, 0.16, 0.9);
fprintf('Force: %.2f kN\n', force/1000);

% Batch prediction
n = 100;
roof_disp = 0.2 * randn(n, 1);
roof_vel = 1.0 * randn(n, 1);
tmd_disp = roof_disp + 0.05 * randn(n, 1);
tmd_vel = roof_vel + 0.1 * randn(n, 1);

forces = perfect_rl_predict_batch(API_URL, roof_disp, roof_vel, tmd_disp, tmd_vel);
fprintf('Mean force: %.2f kN\n', mean(forces)/1000);
```

---

## 📊 **API ENDPOINTS**

### **1. Health Check**

```powershell
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model": "Perfect RL (Champion)",
  "performance": "24.67 cm (TEST3), 20.80 cm (TEST4)"
}
```

---

### **2. Model Info**

```powershell
curl http://localhost:8000/info
```

**Response:**
```json
{
  "name": "Perfect RL (Champion)",
  "performance": {
    "TEST3_M4.5": "24.67 cm (21.8% vs passive)",
    "TEST4_M6.9": "20.80 cm (32% vs passive)",
    "average": "~21.5 cm"
  },
  "comparison": {
    "vs_fuzzy": "+14% better (average)",
    "rank": "🥇 1st place out of 5 methods"
  }
}
```

---

### **3. Single Prediction**

```powershell
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"roof_displacement":0.15,"roof_velocity":0.8,"tmd_displacement":0.16,"tmd_velocity":0.9}'
```

**Response:**
```json
{
  "force_N": -52340.5,
  "force_kN": -52.34,
  "inference_time_ms": 8.2,
  "model": "Perfect RL (Champion)"
}
```

---

### **4. Batch Prediction**

```powershell
curl -X POST http://localhost:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d '{"roof_displacements":[0.15,-0.10],"roof_velocities":[0.8,-0.5],"tmd_displacements":[0.16,-0.09],"tmd_velocities":[0.85,-0.48]}'
```

**Response:**
```json
{
  "forces_N": [-52340.5, 38920.3],
  "forces_kN": [-52.34, 38.92],
  "count": 2,
  "total_time_ms": 15.6,
  "avg_time_ms": 7.8
}
```

---

## 📁 **FILES YOU HAVE**

```
your-project/
├── perfect_rl_api.py              ← Main API server (RUN THIS!)
├── test_perfect_rl_api.py         ← Test client
├── perfect_rl_matlab.m            ← MATLAB functions
├── simple_rl_models/
│   └── perfect_rl_final.zip       ← Your champion model
└── requirements.txt               ← Dependencies
```

---

## 📋 **REQUIREMENTS**

Create `requirements.txt`:

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
numpy==1.24.3
torch==2.1.0
stable-baselines3==2.1.0
requests==2.31.0
```

**Install:**
```powershell
pip install -r requirements.txt
```

---

## 🌐 **INTERACTIVE DOCS**

**Visit:** http://localhost:8000/docs

You'll see:
- 🟢 Interactive API playground
- 📝 Try endpoints directly in browser
- 📊 See request/response examples
- 🔍 Test different inputs

**This is the EASIEST way to test!**

---

## 💻 **PYTHON CLIENT EXAMPLE**

```python
import requests

API_URL = "http://localhost:8000"

# Single prediction
data = {
    "roof_displacement": 0.15,
    "roof_velocity": 0.8,
    "tmd_displacement": 0.16,
    "tmd_velocity": 0.9
}

response = requests.post(f"{API_URL}/predict", json=data)
result = response.json()

print(f"Force: {result['force_kN']:.2f} kN")
print(f"Time: {result['inference_time_ms']:.2f} ms")
```

---

## 🔧 **TROUBLESHOOTING**

### **"Model not found"**
```
Make sure simple_rl_models/perfect_rl_final.zip exists!

Check path:
dir simple_rl_models\perfect_rl_final.zip
```

### **"Module not found"**
```
Install requirements:
pip install -r requirements.txt
```

### **"Port 8000 already in use"**
```
Change port in perfect_rl_api.py (last line):
uvicorn.run(app, host="0.0.0.0", port=8001)  # Change to 8001
```

### **MATLAB can't connect**
```
1. Check API is running: curl http://localhost:8000/health
2. Check firewall isn't blocking
3. Try 127.0.0.1 instead of localhost
```

---

## ⚡ **PERFORMANCE**

**Typical inference times:**
- Single prediction: 8-12 ms
- Batch (100): 15-20 ms total (~0.15-0.20 ms per prediction)

**Throughput:**
- Single: ~100-120 predictions/second
- Batch: ~500-600 predictions/second

**Memory:**
- Model size: ~8 MB
- RAM usage: ~200 MB

---

## 🎯 **QUICK VALIDATION**

**Test everything works:**

```powershell
# Terminal 1: Start API
python perfect_rl_api.py

# Terminal 2: Run tests
python test_perfect_rl_api.py

# Terminal 3: Check browser
# Visit http://localhost:8000/docs
```

**If all green checkmarks → You're ready!** ✅

---

## 🚀 **NEXT STEPS**

### **Local Testing:**
- ✅ API running on localhost:8000
- ✅ All tests passing
- ✅ MATLAB integration working

### **Production Deployment:**
1. Docker containerization (see PERFECT_RL_API_DEPLOYMENT.md)
2. Cloud deployment (Google Cloud Run)
3. Load balancing (if needed)
4. Monitoring setup

---

## 🏆 **YOU'RE READY!**

Your champion model is now deployed as an API:

```
✅ Single prediction endpoint
✅ Batch prediction endpoint  
✅ Health checks
✅ Model information
✅ Python client
✅ MATLAB integration
✅ Interactive docs
✅ Production ready

Performance: 🥇 1st place, beats fuzzy by 14%
```

---

**Start your API now:**

```powershell
python perfect_rl_api.py
```

**Then test it:**

```powershell
python test_perfect_rl_api.py
```

**🎉 Your champion model is LIVE!** 🚀
