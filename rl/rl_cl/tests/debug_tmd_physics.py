"""
DEBUG TMD PHYSICS - Test if passive TMD works correctly
"""
import numpy as np
from tmd_environment import make_improved_tmd_env

print("\n" + "="*70)
print("DEBUGGING TMD PHYSICS")
print("="*70)

# Test 1: NO TMD, NO CONTROL (baseline worst case)
print("\n[TEST 1] NO TMD, NO CONTROL")
print("-" * 70)
test_file = "../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv"
env = make_improved_tmd_env(test_file, max_force=1.0)
obs, _ = env.reset()

# Disable TMD by setting its properties to zero
env.tmd_k = 0.0  # No TMD stiffness
env.tmd_c = 0.0  # No TMD damping

done = False
peak = 0
while not done:
    action = np.array([0.0])  # No active control
    obs, reward, done, truncated, info = env.step(action)
    peak = max(peak, abs(info['roof_displacement']))
    done = done or truncated

print(f"Peak displacement (no TMD): {peak*100:.2f} cm")

# Test 2: PASSIVE TMD (mechanical only, no active control)
print("\n[TEST 2] PASSIVE TMD (mechanical tuning only)")
print("-" * 70)
env = make_improved_tmd_env(test_file, max_force=150000.0)
obs, _ = env.reset()

# TMD is active mechanically but no control force
done = False
peak = 0
while not done:
    action = np.array([0.0])  # No active control
    obs, reward, done, truncated, info = env.step(action)
    peak = max(peak, abs(info['roof_displacement']))
    done = done or truncated

print(f"Peak displacement (passive TMD): {peak*100:.2f} cm")

# Test 3: CONSTANT CONTROL FORCE (test if control helps or hurts)
print("\n[TEST 3] CONSTANT CONTROL FORCE (sanity check)")
print("-" * 70)
for force_level in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    env = make_improved_tmd_env(test_file, max_force=150000.0)
    obs, _ = env.reset()

    done = False
    peak = 0
    while not done:
        action = np.array([force_level])  # Constant force
        obs, reward, done, truncated, info = env.step(action)
        peak = max(peak, abs(info['roof_displacement']))
        done = done or truncated

    print(f"Force={force_level:+.1f} → Peak: {peak*100:.2f} cm")

# Test 4: OBSERVATION VALUES (check if they're reasonable)
print("\n[TEST 4] OBSERVATION VALUE RANGES")
print("-" * 70)
env = make_improved_tmd_env(test_file, max_force=150000.0)
obs, _ = env.reset()

obs_min = obs.copy()
obs_max = obs.copy()

done = False
step = 0
while not done and step < 100:  # First 100 steps
    action = np.array([0.0])
    obs, reward, done, truncated, info = env.step(action)
    obs_min = np.minimum(obs_min, obs)
    obs_max = np.maximum(obs_max, obs)
    done = done or truncated
    step += 1

obs_labels = ['roof_disp', 'roof_vel', 'floor8_disp', 'floor8_vel',
              'floor6_disp', 'floor6_vel', 'tmd_disp', 'tmd_vel']

print("Observation ranges (first 100 steps, no control):")
for i, label in enumerate(obs_labels):
    print(f"  {label:12s}: [{obs_min[i]:+.4f}, {obs_max[i]:+.4f}]")

# Test 5: REWARD SIGNAL (check if it makes sense)
print("\n[TEST 5] REWARD SIGNAL ANALYSIS")
print("-" * 70)
env = make_improved_tmd_env(test_file, max_force=150000.0)
obs, _ = env.reset()

rewards_no_control = []
rewards_with_control = []

# No control
done = False
step = 0
while not done and step < 50:
    action = np.array([0.0])
    obs, reward, done, truncated, info = env.step(action)
    rewards_no_control.append(reward)
    done = done or truncated
    step += 1

# With random control
env = make_improved_tmd_env(test_file, max_force=150000.0)
obs, _ = env.reset()
done = False
step = 0
while not done and step < 50:
    action = np.random.uniform(-1.0, 1.0, size=1)
    obs, reward, done, truncated, info = env.step(action)
    rewards_with_control.append(reward)
    done = done or truncated
    step += 1

print(f"Average reward (no control):   {np.mean(rewards_no_control):.4f}")
print(f"Average reward (random control): {np.mean(rewards_with_control):.4f}")

# Test 6: CHECK TMD MASS RATIO
print("\n[TEST 6] TMD CONFIGURATION")
print("-" * 70)
env = make_improved_tmd_env(test_file, max_force=150000.0)
print(f"Floor mass: {env.floor_mass} kg")
print(f"TMD mass: {env.tmd_mass} kg")
print(f"TMD mass ratio: {env.tmd_mass/env.floor_mass*100:.2f}%")
print(f"TMD stiffness: {env.tmd_k} N/m")
print(f"TMD damping: {env.tmd_c} N·s/m")

# Calculate natural frequency
import math
tmd_freq = math.sqrt(env.tmd_k / env.tmd_mass) / (2 * math.pi)
print(f"TMD natural frequency: {tmd_freq:.3f} Hz")

# Building natural frequency (approximate)
M_total = env.floor_mass * env.n_floors
K_total = np.mean(env.story_stiffness) * env.n_floors
building_freq = math.sqrt(K_total / M_total) / (2 * math.pi)
print(f"Building natural frequency (approx): {building_freq:.3f} Hz")
print(f"TMD tuning ratio: {tmd_freq/building_freq:.3f}")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
