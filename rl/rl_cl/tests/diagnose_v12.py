"""
Diagnostic script to understand V12 performance issues
"""
import sys
import os
sys.path.insert(0, '../../restapi/rl_cl')

import numpy as np
from stable_baselines3 import PPO
from tmd_environment_v12_soft_story import make_soft_story_tmd_env

# Test file
test_file = '../../matlab/datasets/test/PEER_small_M4.5_PGA0.25g.csv'
model_path = 'models/v12_soft_story_breakthrough/final_model.zip'

print("="*80)
print("V12 DIAGNOSTIC - Understanding Poor Performance")
print("="*80)

# Load model
print("\nLoading model...")
if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
    sys.exit(1)

model = PPO.load(model_path)
print(f"[OK] Model loaded from {model_path}")

# Create environment
print(f"\nCreating environment with test file...")
env = make_soft_story_tmd_env(test_file, max_force=300000, reward_scale=1.0)

# Run controlled episode
print("\nRunning CONTROLLED episode with trained model...")
obs, _ = env.reset()
done = False
truncated = False

controlled_displacements = []
controlled_forces = []
controlled_floor8_drifts = []
controlled_all_drifts = []  # Track all floors

while not (done or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

    controlled_displacements.append(env.d[:12].copy())  # All building floors
    controlled_forces.append(abs(info['control_force']))
    controlled_floor8_drifts.append(info['floor8_drift'])

controlled_displacements = np.array(controlled_displacements)
controlled_metrics = env.get_episode_metrics()

# Run uncontrolled episode
print("Running UNCONTROLLED episode (no TMD)...")
env2 = make_soft_story_tmd_env(test_file, max_force=300000, reward_scale=1.0)
obs, _ = env2.reset()
done = False
truncated = False

uncontrolled_displacements = []

while not (done or truncated):
    obs, reward, done, truncated, info = env2.step(np.array([0.0]))  # No control
    uncontrolled_displacements.append(env2.d[:12].copy())

uncontrolled_displacements = np.array(uncontrolled_displacements)
uncontrolled_metrics = env2.get_episode_metrics()

# Analysis
print("\n" + "="*80)
print("DISPLACEMENT ANALYSIS")
print("="*80)

print("\nPeak displacement by floor:")
print(f"{'Floor':<8} {'Uncontrolled (cm)':<20} {'Controlled (cm)':<20} {'Reduction (%)':<15}")
print("-"*80)

story_height = 3.0  # meters

for floor_idx in range(12):
    uncont_peak = np.max(np.abs(uncontrolled_displacements[:, floor_idx])) * 100
    cont_peak = np.max(np.abs(controlled_displacements[:, floor_idx])) * 100
    reduction = ((uncont_peak - cont_peak) / uncont_peak) * 100 if uncont_peak > 0 else 0

    marker = " <-- SOFT STORY" if floor_idx == 7 else ""
    print(f"{floor_idx+1:<8} {uncont_peak:<20.2f} {cont_peak:<20.2f} {reduction:<15.1f}{marker}")

print("\n" + "="*80)
print("DRIFT ANALYSIS (ISDR)")
print("="*80)

print("\nInterstory drift ratio by floor:")
print(f"{'Floor':<8} {'Uncontrolled (%)':<20} {'Controlled (%)':<20} {'Reduction (%)':<15}")
print("-"*80)

uncont_max_isdrs = []
cont_max_isdrs = []

for floor_idx in range(12):
    if floor_idx == 0:
        uncont_drifts = np.abs(uncontrolled_displacements[:, 0])
        cont_drifts = np.abs(controlled_displacements[:, 0])
    else:
        uncont_drifts = np.abs(uncontrolled_displacements[:, floor_idx] - uncontrolled_displacements[:, floor_idx-1])
        cont_drifts = np.abs(controlled_displacements[:, floor_idx] - controlled_displacements[:, floor_idx-1])

    uncont_isdr = (np.max(uncont_drifts) / story_height) * 100
    cont_isdr = (np.max(cont_drifts) / story_height) * 100

    uncont_max_isdrs.append(uncont_isdr)
    cont_max_isdrs.append(cont_isdr)

    reduction = ((uncont_isdr - cont_isdr) / uncont_isdr) * 100 if uncont_isdr > 0 else 0

    marker = " <-- SOFT STORY" if floor_idx == 7 else ""
    print(f"{floor_idx+1:<8} {uncont_isdr:<20.3f} {cont_isdr:<20.3f} {reduction:<15.1f}{marker}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

uncont_max_isdr_overall = max(uncont_max_isdrs)
cont_max_isdr_overall = max(cont_max_isdrs)
isdr_floor_uncont = uncont_max_isdrs.index(uncont_max_isdr_overall) + 1
isdr_floor_cont = cont_max_isdrs.index(cont_max_isdr_overall) + 1

print(f"\nUncontrolled:")
print(f"  Peak roof displacement: {np.max(np.abs(uncontrolled_displacements[:, 11])) * 100:.2f} cm")
print(f"  Max ISDR: {uncont_max_isdr_overall:.3f}% (at floor {isdr_floor_uncont})")
print(f"  Floor 8 ISDR: {uncont_max_isdrs[7]:.3f}%")

print(f"\nControlled (V12 Soft-Story TMD):")
print(f"  Peak roof displacement: {np.max(np.abs(controlled_displacements[:, 11])) * 100:.2f} cm")
print(f"  Max ISDR: {cont_max_isdr_overall:.3f}% (at floor {isdr_floor_cont})")
print(f"  Floor 8 ISDR: {cont_max_isdrs[7]:.3f}%")
print(f"  Mean control force: {np.mean(controlled_forces)/1000:.1f} kN")
print(f"  Peak control force: {np.max(controlled_forces)/1000:.1f} kN")

print(f"\nImprovement:")
disp_improvement = ((np.max(np.abs(uncontrolled_displacements[:, 11])) - np.max(np.abs(controlled_displacements[:, 11]))) /
                    np.max(np.abs(uncontrolled_displacements[:, 11]))) * 100
isdr_improvement = ((uncont_max_isdr_overall - cont_max_isdr_overall) / uncont_max_isdr_overall) * 100

print(f"  Displacement reduction: {disp_improvement:.1f}%")
print(f"  ISDR reduction: {isdr_improvement:.1f}%")

print(f"\nTarget Achievement:")
disp_status = "[OK]" if np.max(np.abs(controlled_displacements[:, 11])) * 100 <= 14.0 else "[FAIL]"
isdr_status = "[OK]" if cont_max_isdr_overall <= 0.4 else "[FAIL]"
print(f"  Displacement: {np.max(np.abs(controlled_displacements[:, 11])) * 100:.2f} cm (target: 14.0 cm) {disp_status}")
print(f"  ISDR: {cont_max_isdr_overall:.3f}% (target: 0.4%) {isdr_status}")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

print(f"\nThe V12 environment is only tracking floor 8 drift in get_episode_metrics()!")
print(f"Reported ISDR from metrics: {controlled_metrics['max_isdr_percent']:.3f}%")
print(f"Actual max ISDR across all floors: {cont_max_isdr_overall:.3f}%")
print(f"\nThis means:")
print(f"  1. The reward function only penalizes floor 8 drift, not overall max ISDR")
print(f"  2. The agent may be reducing floor 8 drift while allowing other floors to drift more")
print(f"  3. The DCR calculation is also flawed (based only on floor 8 drift history)")

print("\n" + "="*80)
