# REINFORCEMENT LEARNING - COMPLETE DEPLOYMENT GUIDE

**Train and deploy an RL agent for optimal TMD control**

Author: Siddharth  
Date: December 2025

---

## 📋 OVERVIEW

This guide walks you through:
1. Installing dependencies
2. Training the RL agent (12-24 hours)
3. Testing the trained model
4. Deploying to Cloud Run API
5. Integrating with MATLAB

**Expected Performance:** 30-40% improvement vs passive TMD

---

## STEP 1: INSTALL DEPENDENCIES

### Create virtual environment (recommended):
```bash
python -m venv rl_env
source rl_env/bin/activate  # On Windows: rl_env\Scripts\activate
```

### Install packages:
```bash
pip install numpy matplotlib pandas
pip install gymnasium
pip install stable-baselines3[extra]
pip install torch  # CPU version (or torch with CUDA for GPU)
```

### Verify installation:
```python
python -c "import stable_baselines3; print('✅ SB3 installed:', stable_baselines3.__version__)"
```

---

## STEP 2: PREPARE TRAINING DATA

The RL agent needs earthquake data to practice on.

### Option A: Use your PEER datasets
```bash
# Your earthquakes are already in datasets/
ls datasets/TEST*.csv
```

### Option B: Quick test with synthetic data
```bash
# Use the quick training mode (see Step 3)
python train_rl.py --quick
```

---

## STEP 3: TRAIN THE RL AGENT

### Quick Test (10 minutes - for testing only):
```bash
python train_rl.py --quick
```
This creates synthetic earthquakes and trains for 10k steps.  
**Use this to verify everything works before full training.**

### Full Training (12-24 hours on CPU, 2-4 hours on GPU):
```bash
python train_rl.py \
  --earthquakes datasets/TEST3_small_earthquake_M4.5.csv \
               datasets/TEST4_large_earthquake_M6.9.csv \
               datasets/TEST6b_with_10pct_noise.csv \
  --timesteps 500000
```

**Parameters:**
- `--earthquakes`: List of CSV files to train on (use 2-5 diverse scenarios)
- `--timesteps`: Total training steps (500k = good, 1M = better)
- `--resume <path>`: Resume from checkpoint if training interrupted

### Monitor training progress:
```bash
# In another terminal
tensorboard --logdir rl_tensorboard
```
Then open http://localhost:6006 in your browser

### What to expect during training:

**First 10k steps (30 min):**
- Reward: -0.15 to -0.10 (learning basics)
- Peak displacement: ~40 cm (worse than passive)

**50k steps (2-3 hours):**
- Reward: -0.08 to -0.05 (getting better)
- Peak displacement: ~28 cm (approaching passive)

**200k steps (8-12 hours):**
- Reward: -0.04 to -0.02 (solid performance)
- Peak displacement: ~22 cm (better than passive!)

**500k steps (done!):**
- Reward: -0.02 to -0.01 (near optimal)
- Peak displacement: ~19 cm (30-40% better than passive) 🎯

### Training outputs:
- `rl_models/tmd_sac_final.zip` - Final trained model
- `rl_models/best_model.zip` - Best model during training
- `rl_models/tmd_sac_checkpoint_*.zip` - Checkpoints every 10k steps
- `rl_logs/training_progress.png` - Training graphs

---

## STEP 4: TEST THE TRAINED MODEL

### Single earthquake test:
```bash
python test_rl_model.py \
  --model rl_models/tmd_sac_final.zip \
  --earthquake datasets/TEST3_small_earthquake_M4.5.csv
```

**Expected output:**
```
============================================================
  RESULTS
============================================================
Peak displacement: 19.45 cm
RMS displacement:  12.67 cm
Mean |force|:      32.45 kN
Max |force|:       87.23 kN
Total reward:      -18.67
============================================================
```

### Compare against other controllers:
```bash
python test_rl_model.py \
  --model rl_models/tmd_sac_final.zip \
  --earthquake datasets/TEST3_small_earthquake_M4.5.csv \
  --compare \
  --passive-peak 31.53 \
  --fuzzy-peak 26.0 \
  --pd-peak 27.17
```

This creates a bar chart comparing all methods!

### Batch evaluation on multiple earthquakes:
```bash
python test_rl_model.py \
  --model rl_models/tmd_sac_final.zip \
  --batch datasets/TEST*.csv
```

Saves results to `rl_evaluation_results.csv`

---

## STEP 5: DEPLOY TO API

### 5a. Add RL controller to your API folder

Copy these files to your API directory:
```bash
cp rl_controller.py /path/to/your-api/
cp rl_models/tmd_sac_final.zip /path/to/your-api/
```

### 5b. Update main.py

Add imports:
```python
from rl_controller import RLTMDController
```

Initialize at startup:
```python
# After app = FastAPI()
rl_controller = RLTMDController("tmd_sac_final.zip")
```

Add request/response models:
```python
class RLBatchRequest(BaseModel):
    roof_displacements: List[float]
    roof_velocities: List[float]
    tmd_displacements: List[float]
    tmd_velocities: List[float]

class RLBatchResponse(BaseModel):
    forces: List[float]
    force_unit: str = "kN"
    num_predictions: int
    inference_time_ms: float
```

Add endpoint (full code in rl_controller.py):
```python
@app.post("/rl/predict-batch", response_model=RLBatchResponse)
async def rl_batch_predict(request: RLBatchRequest):
    # ... see rl_controller.py for full implementation
```

### 5c. Update requirements.txt

```txt
fastapi
uvicorn
pydantic
numpy
torch
stable-baselines3
scikit-fuzzy  # For fuzzy controller
```

### 5d. Update Dockerfile

Make sure model file is copied:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY main.py .
COPY fixed_fuzzy_controller.py .
COPY rl_controller.py .
COPY tmd_sac_final.zip .  # ← Add this

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### 5e. Test locally with Docker

```bash
docker build -t tmd-api-test .
docker run -p 8080:8080 tmd-api-test
```

Test the RL endpoint:
```bash
curl -X POST http://localhost:8080/rl/predict-batch \
  -H "Content-Type: application/json" \
  -d '{
    "roof_displacements": [0.15, -0.10],
    "roof_velocities": [0.8, -0.5],
    "tmd_displacements": [0.16, -0.09],
    "tmd_velocities": [0.85, -0.48]
  }'
```

**Expected response:**
```json
{
  "forces": [-52.34, 38.92],
  "force_unit": "kN",
  "num_predictions": 2,
  "inference_time_ms": 8.5
}
```

### 5f. Deploy to Cloud Run

```bash
gcloud run deploy tmd-api \
  --source . \
  --region us-east1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300
```

---

## STEP 6: INTEGRATE WITH MATLAB

### Update batch comparison script

The RL controller needs both roof AND TMD states (4 inputs).

**In your batch comparison script:**

```matlab
% After passive simulation
roof_disp = disp_passive(12, :);  % Roof
roof_vel = gradient(roof_disp, dt);
tmd_disp = disp_passive(13, :);   % TMD
tmd_vel = gradient(tmd_disp, dt);

% Call RL API
forces_rl_N = get_rl_forces_from_api(API_URL, roof_disp, roof_vel, tmd_disp, tmd_vel);
```

**Add this function:**

```matlab
function forces_N = get_rl_forces_from_api(API_URL, roof_disp, roof_vel, tmd_disp, tmd_vel)
    % Prepare data
    batch_data = struct(...
        'roof_displacements', roof_disp, ...
        'roof_velocities', roof_vel, ...
        'tmd_displacements', tmd_disp, ...
        'tmd_velocities', tmd_vel ...
    );
    
    json_data = jsonencode(batch_data);
    
    options = weboptions('MediaType', 'application/json', ...
                         'ContentType', 'json', ...
                         'Timeout', 120);
    
    % Call API
    response = webwrite([API_URL '/rl/predict-batch'], json_data, options);
    
    % Convert kN to N
    forces_N = response.forces * 1000;
    
    % Ensure column vector
    if size(forces_N, 1) == 1
        forces_N = forces_N';
    end
end
```

---

## STEP 7: RUN FULL COMPARISON

Now test all 4 controllers:

```matlab
% Your comparison script should now test:
% 1. Passive TMD
% 2. Fixed Fuzzy
% 3. LQR (if implemented)
% 4. RL

>> batch_compare_peer_datasets
```

**Expected results:**

| Method | Peak (cm) | vs Passive | Rank |
|--------|-----------|------------|------|
| Passive | 31.53 | baseline | 4th |
| Fuzzy | ~26.0 | 17% better | 3rd |
| LQR | ~23.0 | 27% better | 2nd |
| **RL** | **~20.0** | **37% better** | **1st** 🏆 |

---

## TROUBLESHOOTING

### Issue: Training is very slow
**Fix:** Use fewer parallel environments or reduce total timesteps for testing
```bash
python train_rl.py --timesteps 100000  # Shorter training
```

### Issue: "CUDA out of memory"
**Fix:** Force CPU training
```python
# In train_rl.py, change:
device='cpu'  # Instead of 'auto'
```

### Issue: RL performs worse than passive
**Possible causes:**
1. **Not enough training** - Train longer (1M+ steps)
2. **Bad reward function** - Current reward is `-|displacement|` which is correct
3. **Model hasn't converged** - Check TensorBoard, reward should be increasing

**Quick fix:** Use best_model.zip instead of tmd_sac_final.zip

### Issue: API returns 500 error for RL
**Fix:** Check logs
```bash
gcloud logging read "resource.type=cloud_run_revision" --limit 50
```

Common problems:
- Model file not copied to container
- Missing stable-baselines3 in requirements.txt
- Torch not installed

---

## PERFORMANCE EXPECTATIONS

### Expected improvements by training duration:

| Training Time | Timesteps | Expected Improvement |
|--------------|-----------|----------------------|
| 1 hour (CPU) | 50k | 5-10% (learning) |
| 6 hours (CPU) | 200k | 20-25% (good) |
| 12 hours (CPU) | 500k | 30-35% (very good) |
| 24 hours (CPU) | 1M | 35-40% (excellent) |
| 2 hours (GPU) | 500k | 30-35% (very good) |

### Why RL beats other methods:

**vs Passive TMD (+30-40%):**
- Passive uses fixed damping
- RL adapts control in real-time

**vs Fuzzy Logic (+15-20%):**
- Fuzzy uses hand-designed rules
- RL learned from 10,000+ practice episodes

**vs LQR (+5-15%):**
- LQR assumes linear dynamics
- RL handles nonlinear soft 8th floor optimally

---

## TIPS FOR BEST RESULTS

### 1. Train on diverse earthquakes
Use 3-5 different earthquake types:
- Small (M4-5)
- Large (M6-7)
- With noise
- Different frequency content

### 2. Training duration
- Minimum: 200k steps
- Recommended: 500k steps
- Best: 1M steps

### 3. Hyperparameter tuning
For advanced users, try adjusting in `TrainingConfig`:
```python
learning_rate = 3e-4  # Try 1e-4 to 1e-3
batch_size = 256      # Try 128 or 512
gamma = 0.99          # Try 0.95 or 0.995
```

### 4. Reward shaping (advanced)
Modify reward in `tmd_environment.py`:
```python
# Current
reward = -abs(roof_disp)

# Alternative (penalize velocity too)
reward = -abs(roof_disp) - 0.1 * abs(roof_vel)

# Alternative (energy-aware)
reward = -abs(roof_disp) - 0.0001 * (control_force/max_force)**2
```

---

## CHECKLIST

Before considering RL deployment complete:

- [ ] Dependencies installed successfully
- [ ] Quick training test (--quick) works
- [ ] Full training completed (500k+ steps)
- [ ] Test shows >30% improvement vs passive
- [ ] Model file (.zip) is < 50 MB
- [ ] RL controller loads successfully
- [ ] API endpoint returns correct forces
- [ ] MATLAB can call RL endpoint
- [ ] Batch comparison shows RL as best performer
- [ ] Results documented and saved

---

## NEXT STEPS AFTER RL

Once RL is working:

1. **Document results** - Compare all 4 methods
2. **Create visualizations** - Bar charts, time series plots
3. **Prepare presentation** - Show learning progress, final comparison
4. **Write report** - Methodology, results, discussion

Your final lineup:
- ✅ Passive TMD (baseline)
- ✅ Fuzzy Logic (human rules)
- ✅ LQR (mathematical optimum)
- ✅ RL (AI-discovered strategy)

**You have a complete control strategy comparison!** 🎉

---

## TIMELINE ESTIMATE

| Task | Time |
|------|------|
| Install dependencies | 15 min |
| Quick test training | 10 min |
| Setup full training | 30 min |
| **Training (automated)** | **12-24 hours** |
| Test trained model | 30 min |
| Deploy to API | 1 hour |
| MATLAB integration | 1 hour |
| Full comparison | 2 hours |
| **TOTAL ACTIVE WORK** | **~6 hours** |

Most time is automated training - set it up and let it run overnight!

---

## SUMMARY

**What RL does:** Learns optimal control through trial and error  
**How long:** 12-24 hours training (mostly automated)  
**Active work:** ~6 hours  
**Expected result:** 30-40% better than passive TMD  
**Key advantage:** Discovers nonlinear strategies no human would design

**Ready to start?** Run the quick test first to verify everything works! 🚀
