"""
Analyze why TMD displacement runs away with RL control
"""
import sys
sys.path.insert(0, '/Users/Shared/dev/git/struct-engineer-ai')
import numpy as np
from rl.rl_cl.tmd_environment import make_improved_tmd_env
from stable_baselines3 import SAC
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

test_file = "../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv"
model = SAC.load("rl_cl_robust_models/stage1_50kN_final_robust.zip")

env = make_improved_tmd_env(test_file, max_force=50000)
obs, _ = env.reset()

# Track time histories
time = []
roof_disp_hist = []
tmd_disp_hist = []
tmd_rel_hist = []
control_force_hist = []
step_count = 0

done = False
while not done and step_count < 300:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    
    roof_disp = env.displacement[11]
    tmd_disp = env.displacement[12]
    tmd_rel = tmd_disp - roof_disp
    control_force = info['control_force']
    
    time.append(step_count * env.dt)
    roof_disp_hist.append(roof_disp * 100)  # cm
    tmd_disp_hist.append(tmd_disp * 100)  # cm
    tmd_rel_hist.append(tmd_rel * 100)  # cm
    control_force_hist.append(control_force / 1000)  # kN
    
    done = done or truncated
    step_count += 1

# Plot
fig, axes = plt.subplots(4, 1, figsize=(12, 10))

axes[0].plot(time, roof_disp_hist)
axes[0].set_ylabel('Roof Disp (cm)')
axes[0].grid(True)
axes[0].axhline(y=120, color='r', linestyle='--', label='Obs bound')
axes[0].axhline(y=-120, color='r', linestyle='--')
axes[0].legend()

axes[1].plot(time, tmd_disp_hist, label='Absolute')
axes[1].set_ylabel('TMD Abs Disp (cm)')
axes[1].grid(True)
axes[1].axhline(y=150, color='r', linestyle='--', label='Obs bound')
axes[1].axhline(y=-150, color='r', linestyle='--')
axes[1].legend()

axes[2].plot(time, tmd_rel_hist, label='Relative (TMD - roof)')
axes[2].set_ylabel('TMD Rel Disp (cm)')
axes[2].grid(True)

axes[3].plot(time, control_force_hist)
axes[3].set_ylabel('Control Force (kN)')
axes[3].set_xlabel('Time (s)')
axes[3].grid(True)
axes[3].axhline(y=50, color='r', linestyle='--', label='Max force')
axes[3].axhline(y=-50, color='r', linestyle='--')
axes[3].legend()

plt.tight_layout()
plt.savefig('tmd_runaway_analysis.png', dpi=150)
print("✅ Saved: tmd_runaway_analysis.png")

# Print statistics
print("\n" + "="*70)
print("TMD RUNAWAY ANALYSIS")
print("="*70)
print(f"Max roof displacement: {max(np.abs(roof_disp_hist)):.2f} cm")
print(f"Max TMD absolute displacement: {max(np.abs(tmd_disp_hist)):.2f} cm")
print(f"Max TMD relative displacement: {max(np.abs(tmd_rel_hist)):.2f} cm")
print(f"Max control force: {max(np.abs(control_force_hist)):.2f} kN")

# Check when TMD exceeds bounds
tmd_exceed_idx = next((i for i, val in enumerate(tmd_disp_hist) if abs(val) > 150), None)
if tmd_exceed_idx:
    print(f"\nTMD exceeded bounds at step {tmd_exceed_idx} (t={time[tmd_exceed_idx]:.2f}s)")
    print(f"  Roof disp: {roof_disp_hist[tmd_exceed_idx]:.2f} cm")
    print(f"  TMD abs disp: {tmd_disp_hist[tmd_exceed_idx]:.2f} cm")
    print(f"  Control force: {control_force_hist[tmd_exceed_idx]:.2f} kN")
