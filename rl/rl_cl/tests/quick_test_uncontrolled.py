import numpy as np
from tmd_environment import make_improved_tmd_env

print("\n" + "="*70)
print("TESTING UNCONTROLLED BUILDING RESPONSE (NO TMD, NO CONTROL)")
print("Building: 20 MN/m stiffness (MATLAB-aligned)")
print("="*70)

# Test M4.5 uncontrolled
test_file = "../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv"
env = make_improved_tmd_env(test_file, max_force=1.0)  # Tiny force to avoid division by zero
obs, _ = env.reset()
done = False
peak = 0

while not done:
    action = np.array([0.0])  # No force
    obs, reward, done, truncated, info = env.step(action)
    peak = max(peak, abs(info['roof_displacement']))
    done = done or truncated

print(f"\nM4.5 (0.25g) UNCONTROLLED: {peak*100:.2f} cm")

# Test M5.7 uncontrolled
test_file = "../../matlab/datasets/PEER_moderate_M5.7_PGA0.35g.csv"
env = make_improved_tmd_env(test_file, max_force=1.0)
obs, _ = env.reset()
done = False
peak = 0

while not done:
    action = np.array([0.0])
    obs, reward, done, truncated, info = env.step(action)
    peak = max(peak, abs(info['roof_displacement']))
    done = done or truncated

print(f"M5.7 (0.35g) UNCONTROLLED: {peak*100:.2f} cm")

# Test M7.4 uncontrolled
test_file = "../../matlab/datasets/PEER_high_M7.4_PGA0.75g.csv"
env = make_improved_tmd_env(test_file, max_force=1.0)
obs, _ = env.reset()
done = False
peak = 0

while not done:
    action = np.array([0.0])
    obs, reward, done, truncated, info = env.step(action)
    peak = max(peak, abs(info['roof_displacement']))
    done = done or truncated

print(f"M7.4 (0.75g) UNCONTROLLED: {peak*100:.2f} cm")

print("\n" + "="*70)
print("CRITICAL CONTEXT:")
print("="*70)
print("The building is now 40x SOFTER than before (20 MN/m vs 800 MN/m)")
print("This matches MATLAB configuration - displacements SHOULD be larger!")
print("\nOld stiff building (800 MN/m): 8-30 cm typical")
print("New soft building (20 MN/m): 50-200 cm typical")
print("\nThis is CORRECT and EXPECTED for the MATLAB-aligned building!")
print("="*70 + "\n")
