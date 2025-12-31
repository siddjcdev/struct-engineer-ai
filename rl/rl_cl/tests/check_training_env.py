"""
Check what training environment was actually used
"""
import sys
sys.path.insert(0, '/Users/Shared/dev/git/struct-engineer-ai')

from rl.rl_cl.tmd_environment import make_improved_tmd_env

# Create environment exactly as training does
env = make_improved_tmd_env(
    "../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv",
    max_force=50000
)

print("="*70)
print("TRAINING ENVIRONMENT CONFIGURATION")
print("="*70)
print(f"TMD stiffness: {env.tmd_k} N/m")
print(f"TMD damping: {env.tmd_c} N·s/m")
print(f"Floor mass: {env.floor_mass} kg")
print(f"Story stiffness (typical): {env.story_stiffness[0]} N/m")
print(f"Soft story stiffness: {env.story_stiffness[7]} N/m")
print(f"Max force: {env.max_force} N")
print(f"Max steps: {env.max_steps}")

import math
tmd_freq = math.sqrt(env.tmd_k / env.tmd_mass) / (2 * math.pi)
print(f"TMD frequency: {tmd_freq:.3f} Hz")

# Calculate building frequency
import numpy as np
M_floors = env.floor_mass * np.ones(env.n_floors)
M_mat = np.diag(M_floors)
K = env.K[:env.n_floors, :env.n_floors]

eigenvalues = np.linalg.eigvals(np.linalg.inv(M_mat) @ K)
natural_frequencies = np.sqrt(eigenvalues.real) / (2 * np.pi)
building_freq = np.min(natural_frequencies)
print(f"Building fundamental frequency: {building_freq:.3f} Hz")
print(f"TMD tuning ratio: {tmd_freq/building_freq:.3f}")
