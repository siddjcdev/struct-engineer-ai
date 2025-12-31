"""
Check what TMD displacement actually means
"""
import sys
sys.path.insert(0, '/Users/Shared/dev/git/struct-engineer-ai')
import numpy as np
from rl.rl_cl.tmd_environment import make_improved_tmd_env

print("="*70)
print("TMD DISPLACEMENT INTERPRETATION")
print("="*70)

test_file = "../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv"
env = make_improved_tmd_env(test_file, max_force=50000)
obs, _ = env.reset()

print("\nRunning passive TMD (no control)...")
done = False
max_tmd_abs = 0
max_tmd_relative = 0
max_roof = 0
step = 0

while not done and step < 200:
    action = np.array([0.0])
    obs, reward, done, truncated, info = env.step(action)
    
    # TMD displacement from state vector
    tmd_disp_abs = env.displacement[12]  # Absolute displacement
    roof_disp = env.displacement[11]      # Roof displacement
    tmd_disp_relative = tmd_disp_abs - roof_disp  # Relative displacement
    
    max_tmd_abs = max(max_tmd_abs, abs(tmd_disp_abs))
    max_tmd_relative = max(max_tmd_relative, abs(tmd_disp_relative))
    max_roof = max(max_roof, abs(roof_disp))
    
    done = done or truncated
    step += 1

print(f"\nResults (first 200 steps):")
print(f"  Max roof displacement:             {max_roof:.4f} m")
print(f"  Max TMD absolute displacement:     {max_tmd_abs:.4f} m")
print(f"  Max TMD relative displacement:     {max_tmd_relative:.4f} m")

print(f"\n🔍 Observation in state vector:")
print(f"  obs[6] (tmd_disp) = {obs[6]:.4f}")
print(f"  This is: {'ABSOLUTE' if abs(obs[6] - tmd_disp_abs) < 0.001 else 'RELATIVE'} displacement")

# Check what the observation actually contains
print(f"\n🔍 Verifying observation content...")
print(f"  env.displacement[12] (TMD abs): {env.displacement[12]:.4f}")
print(f"  env.displacement[11] (roof):    {env.displacement[11]:.4f}")
print(f"  Relative (TMD - roof):          {env.displacement[12] - env.displacement[11]:.4f}")
print(f"  obs[6] from environment:        {obs[6]:.4f}")
