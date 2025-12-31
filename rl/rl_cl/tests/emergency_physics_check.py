"""
EMERGENCY: Verify basic physics and reward are working correctly
"""
import numpy as np
from tmd_environment import make_improved_tmd_env

print("="*70)
print("EMERGENCY PHYSICS & REWARD CHECK")
print("="*70)

test_file = "../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv"

# Test 1: Verify reward function gives negative penalty for displacement
print("\n[TEST 1] Reward function sanity check")
print("-" * 70)
env = make_improved_tmd_env(test_file, max_force=50000.0)
obs, _ = env.reset()

# Apply zero action for a few steps, check if reward is negative when displacement occurs
for i in range(100):
    obs, reward, done, truncated, info = env.step(np.array([0.0]))
    if abs(info['roof_displacement']) > 0.01:  # 1cm displacement
        print(f"Step {i}: displacement={info['roof_displacement']*100:.2f}cm, reward={reward:.2f}")
        if reward > 0:
            print(f"   ❌ BUG: Reward is POSITIVE when displacement is {info['roof_displacement']*100:.2f}cm!")
        else:
            print(f"   ✓ Reward is negative (correct)")
        break

# Test 2: Verify control force actually affects displacement
print("\n[TEST 2] Control force effectiveness check")
print("-" * 70)

# Run with NO control
env1 = make_improved_tmd_env(test_file, max_force=50000.0)
obs, _ = env1.reset()
done = False
peak_no_control = 0
while not done and env1.current_step < 1000:
    obs, reward, done, truncated, info = env1.step(np.array([0.0]))
    peak_no_control = max(peak_no_control, abs(info['roof_displacement']))
    done = done or truncated

# Run with MAXIMUM positive control
env2 = make_improved_tmd_env(test_file, max_force=50000.0)
obs, _ = env2.reset()
done = False
peak_max_positive = 0
while not done and env2.current_step < 1000:
    obs, reward, done, truncated, info = env2.step(np.array([1.0]))  # Max positive
    peak_max_positive = max(peak_max_positive, abs(info['roof_displacement']))
    done = done or truncated

# Run with MAXIMUM negative control
env3 = make_improved_tmd_env(test_file, max_force=50000.0)
obs, _ = env3.reset()
done = False
peak_max_negative = 0
while not done and env3.current_step < 1000:
    obs, reward, done, truncated, info = env3.step(np.array([-1.0]))  # Max negative
    peak_max_negative = max(peak_max_negative, abs(info['roof_displacement']))
    done = done or truncated

print(f"No control:       {peak_no_control*100:.2f} cm")
print(f"Max +1.0 force:   {peak_max_positive*100:.2f} cm")
print(f"Max -1.0 force:   {peak_max_negative*100:.2f} cm")

if abs(peak_max_positive - peak_no_control) < 0.5 and abs(peak_max_negative - peak_no_control) < 0.5:
    print("   ❌ BUG: Control forces have NO EFFECT on displacement!")
elif peak_max_positive < peak_no_control * 0.9 or peak_max_negative < peak_no_control * 0.9:
    print("   ✓ Control forces CAN reduce displacement")
else:
    print("   ⚠️  Control forces make things WORSE")

# Test 3: Check if observations contain useful information
print("\n[TEST 3] Observation information check")
print("-" * 70)
env = make_improved_tmd_env(test_file, max_force=50000.0)
obs, _ = env.reset()
obs_history = []
for i in range(200):
    obs, reward, done, truncated, info = env.step(np.array([0.0]))
    obs_history.append(obs.copy())
    done = done or truncated
    if done:
        break

obs_history = np.array(obs_history)
print(f"Observation ranges over first 200 steps:")
print(f"   Roof displacement: [{obs_history[:,0].min():.4f}, {obs_history[:,0].max():.4f}]")
print(f"   Roof velocity:     [{obs_history[:,1].min():.4f}, {obs_history[:,1].max():.4f}]")
print(f"   TMD displacement:  [{obs_history[:,6].min():.4f}, {obs_history[:,6].max():.4f}]")
print(f"   TMD velocity:      [{obs_history[:,7].min():.4f}, {obs_history[:,7].max():.4f}]")

if obs_history[:,0].max() - obs_history[:,0].min() < 0.001:
    print("   ❌ BUG: Observations are nearly constant (no useful signal)!")
else:
    print("   ✓ Observations vary (contain information)")

# Test 4: Verify reward improves with better control
print("\n[TEST 4] Reward gradient check")
print("-" * 70)
print("Testing if applying optimal control at peak gives better reward...")

env = make_improved_tmd_env(test_file, max_force=50000.0)
obs, _ = env.reset()

# Find peak displacement moment
for i in range(500):
    obs, reward, done, truncated, info = env.step(np.array([0.0]))
    if abs(info['roof_displacement']) > 0.15:  # Near peak
        roof_disp = info['roof_displacement']
        roof_vel = obs[1]

        # Try different actions
        print(f"\n   At peak moment (disp={roof_disp*100:.2f}cm):")

        # Simulate what reward would be for different actions
        # (This is approximate - we can't actually test without re-running)
        if roof_vel > 0 and roof_disp > 0:
            print(f"      Roof moving right (+disp, +vel)")
            print(f"      Best action: Apply LEFT force (negative) to oppose motion")
        elif roof_vel < 0 and roof_disp < 0:
            print(f"      Roof moving left (-disp, -vel)")
            print(f"      Best action: Apply RIGHT force (positive) to oppose motion")

        break

print("\n" + "="*70)
print("DIAGNOSIS:")
print("If ALL tests pass ✓, the environment is correct.")
print("If ANY test fails ❌, there's a physics/reward bug.")
print("="*70 + "\n")
