"""
Sweep TMD parameters to find what actually works best
"""
import numpy as np
import math
from tmd_environment import make_improved_tmd_env

test_file = "../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv"

print("="*70)
print("TMD PARAMETER SWEEP")
print("="*70)

# Building frequency is 0.193 Hz
building_freq = 0.193
tmd_mass = 4000.0

# Test different stiffness values (different frequency ratios)
frequency_ratios = [0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0]

print("\nTesting different TMD frequency ratios:")
print("-" * 70)

best_peak = float('inf')
best_ratio = 0

for ratio in frequency_ratios:
    f_tmd = ratio * building_freq
    omega_tmd = f_tmd * 2 * np.pi
    k_tmd = tmd_mass * (omega_tmd ** 2)
    
    # Use Den Hartog damping for this frequency
    mu = tmd_mass / (12 * 200000)
    zeta = math.sqrt(3 * mu / (8 * (1 + mu)))
    c_tmd = 2 * zeta * math.sqrt(k_tmd * tmd_mass)
    
    env = make_improved_tmd_env(test_file, max_force=1.0)
    obs, _ = env.reset()
    
    env.tmd_k = k_tmd
    env.tmd_c = c_tmd
    env.K = env._build_stiffness_matrix()
    env.C = env._build_damping_matrix()
    
    done = False
    peak = 0
    while not done:
        action = np.array([0.0])
        obs, reward, done, truncated, info = env.step(action)
        peak = max(peak, abs(info['roof_displacement']))
        done = done or truncated
    
    if peak < best_peak:
        best_peak = peak
        best_ratio = ratio
    
    print(f"Ratio={ratio:.2f} (f_tmd={f_tmd:.3f}Hz, k={k_tmd:.0f}N/m) → {peak*100:.2f} cm")

print("\n" + "="*70)
print(f"Best frequency ratio: {best_ratio:.2f} → {best_peak*100:.2f} cm")
print("="*70)
