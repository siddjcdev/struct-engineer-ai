"""
Test TMD physics after fixing tuning parameters
"""
import numpy as np
from tmd_environment import make_improved_tmd_env

print("\n" + "="*70)
print("TESTING FIXED TMD (Den Hartog Optimal Tuning)")
print("="*70)

test_file = "../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv"

# Test 1: NO TMD (baseline)
print("\n[TEST 1] NO TMD, NO CONTROL")
print("-" * 70)
env = make_improved_tmd_env(test_file, max_force=1.0)
obs, _ = env.reset()
env.tmd_k = 0.0
env.tmd_c = 0.0
env.K = env._build_stiffness_matrix()
env.C = env._build_damping_matrix()

done = False
peak = 0
while not done:
    action = np.array([0.0])
    obs, reward, done, truncated, info = env.step(action)
    peak = max(peak, abs(info['roof_displacement']))
    done = done or truncated

print(f"Peak displacement (no TMD): {peak*100:.2f} cm")

# Test 2: PASSIVE TMD with OPTIMAL tuning
print("\n[TEST 2] PASSIVE TMD (Den Hartog optimal k=5840, c=241)")
print("-" * 70)
env = make_improved_tmd_env(test_file, max_force=150000.0)
obs, _ = env.reset()

print(f"TMD stiffness: {env.tmd_k} N/m")
print(f"TMD damping: {env.tmd_c} N·s/m")

import math
tmd_freq = math.sqrt(env.tmd_k / env.tmd_mass) / (2 * math.pi)
print(f"TMD frequency: {tmd_freq:.3f} Hz")

done = False
peak = 0
while not done:
    action = np.array([0.0])
    obs, reward, done, truncated, info = env.step(action)
    peak = max(peak, abs(info['roof_displacement']))
    done = done or truncated

print(f"Peak displacement (passive TMD): {peak*100:.2f} cm")

# Test 3: OLD (WRONG) TMD tuning
print("\n[TEST 3] OLD (WRONG) TMD (k=50000, c=2000)")
print("-" * 70)
env = make_improved_tmd_env(test_file, max_force=150000.0)
obs, _ = env.reset()

# Override with old wrong values
env.tmd_k = 50000
env.tmd_c = 2000
env.K = env._build_stiffness_matrix()
env.C = env._build_damping_matrix()

tmd_freq_old = math.sqrt(env.tmd_k / env.tmd_mass) / (2 * math.pi)
print(f"TMD frequency: {tmd_freq_old:.3f} Hz")

done = False
peak = 0
while not done:
    action = np.array([0.0])
    obs, reward, done, truncated, info = env.step(action)
    peak = max(peak, abs(info['roof_displacement']))
    done = done or truncated

print(f"Peak displacement (old wrong TMD): {peak*100:.2f} cm")

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
