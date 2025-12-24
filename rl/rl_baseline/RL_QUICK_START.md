# RL QUICK START - GET RUNNING IN 30 MINUTES

**The absolute fastest way to start training your RL agent**

---

## STEP 1: INSTALL (5 minutes)

```bash
# Create environment
python -m venv rl_env
source rl_env/bin/activate  # Windows: rl_env\Scripts\activate

# Install packages
pip install numpy matplotlib gymnasium stable-baselines3[extra] torch
```

---

## STEP 2: QUICK TEST (10 minutes)

Test that everything works with synthetic earthquakes:

```bash
python train_rl.py --quick
```

This will:
- Create 3 synthetic earthquakes
- Train for 10,000 steps (~10 minutes)
- Save model to `rl_models/tmd_sac_final.zip`

**Expected output:**
```
Episode 10: Avg reward: -0.12, Avg peak: 35.2 cm
Episode 20: Avg reward: -0.09, Avg peak: 31.8 cm
...
✅ Training complete!
```

---

## STEP 3: TEST THE MODEL (5 minutes)

```bash
python test_rl_model.py \
  --model rl_models/tmd_sac_final.zip \
  --earthquake datasets/TEST3_small_earthquake_M4.5.csv
```

Even with just 10k training steps, you should see it working!

---

## STEP 4: FULL TRAINING (Setup: 5 min, Run: Overnight)

Now train on real earthquakes:

```bash
python train_rl.py \
  --earthquakes datasets/TEST3*.csv datasets/TEST4*.csv datasets/TEST6*.csv \
  --timesteps 500000
```

**Go to bed. Wake up to a trained model!** 🛌

---

## STEP 5: DEPLOY (See full guide)

Once training is done, follow the RL_DEPLOYMENT_GUIDE.md for deployment.

---

## MONITORING TRAINING

Open another terminal:
```bash
tensorboard --logdir rl_tensorboard
```

Visit http://localhost:6006 to watch training progress in real-time!

---

## TROUBLESHOOTING

**"ModuleNotFoundError"** → Install missing package with pip  
**Training very slow** → Use `--timesteps 100000` for faster testing  
**Out of memory** → Close other programs or use CPU with smaller batch size

---

## FILES YOU NEED

All in `/mnt/user-data/outputs/`:
- `tmd_environment.py` - Gymnasium environment
- `train_rl.py` - Training script ⭐
- `test_rl_model.py` - Testing script
- `rl_controller.py` - For API deployment
- `RL_DEPLOYMENT_GUIDE.md` - Full guide

---

## EXPECTED TIMELINE

- **Quick test:** 15 minutes (verify it works)
- **Full training:** 12-24 hours (mostly automated)
- **Testing:** 30 minutes
- **Deployment:** 1-2 hours

**Total active work:** ~3 hours (plus overnight training)

---

**Start with the quick test to make sure everything works, then launch the full training before bed!** 🚀
