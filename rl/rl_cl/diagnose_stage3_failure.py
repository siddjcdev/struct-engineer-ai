"""
Diagnose why Stage 3 (M7.4) is failing catastrophically
"""
import numpy as np
from stable_baselines3 import SAC
from tmd_environment import make_improved_tmd_env

print("\n" + "="*70)
print("DIAGNOSING STAGE 3 FAILURE")
print("="*70)

# Load Stage 3 model
model_path = "rl_cl_robust_models_5_datafix/stage3_150kN_final_robust.zip"
model = SAC.load(model_path)

# Test on M7.4 earthquake
test_file = "../../matlab/datasets/PEER_high_M7.4_PGA0.75g.csv"
env = make_improved_tmd_env(test_file, max_force=150000.0)

print("\n[TEST] Stage 3 Model on M7.4 Earthquake")
print("-" * 70)

obs, _ = env.reset()
done = False
step = 0

# Track metrics
peak_roof = 0
peak_tmd_abs = 0
peak_tmd_rel = 0
max_force = 0
clipped_steps = 0
total_steps = 0

obs_min = obs.copy()
obs_max = obs.copy()

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

    # Track peaks
    peak_roof = max(peak_roof, abs(info['roof_displacement']))
    peak_tmd_abs = max(peak_tmd_abs, abs(env.displacement[12]))  # TMD absolute
    peak_tmd_rel = max(peak_tmd_rel, abs(env.displacement[12] - env.displacement[11]))  # TMD relative to roof
    max_force = max(max_force, abs(action[0] * 150000))

    # Check for clipping
    if np.any(obs <= env.observation_space.low) or np.any(obs >= env.observation_space.high):
        clipped_steps += 1

    # Track observation ranges
    obs_min = np.minimum(obs_min, obs)
    obs_max = np.maximum(obs_max, obs)

    total_steps += 1
    done = done or truncated
    step += 1

print(f"\n📊 Performance Metrics:")
print(f"   Peak roof displacement: {peak_roof*100:.2f} cm")
print(f"   Peak TMD absolute displacement: {peak_tmd_abs*100:.2f} cm")
print(f"   Peak TMD relative displacement: {peak_tmd_rel*100:.2f} cm")
print(f"   Max control force: {max_force/1000:.2f} kN")
print(f"   Clipped steps: {clipped_steps}/{total_steps} ({clipped_steps/total_steps*100:.1f}%)")

print(f"\n📊 Observation Ranges:")
obs_labels = ['roof_disp', 'roof_vel', 'floor8_disp', 'floor8_vel',
              'floor6_disp', 'floor6_vel', 'tmd_disp', 'tmd_vel']
bounds_low = env.observation_space.low
bounds_high = env.observation_space.high

for i, label in enumerate(obs_labels):
    exceeded_low = "❌ EXCEEDED" if obs_min[i] < bounds_low[i] else "✅"
    exceeded_high = "❌ EXCEEDED" if obs_max[i] > bounds_high[i] else "✅"
    print(f"   {label:12s}: [{obs_min[i]:+8.3f}, {obs_max[i]:+8.3f}]  "
          f"(bounds: [{bounds_low[i]:+.1f}, {bounds_high[i]:+.1f}])  "
          f"{exceeded_low if obs_min[i] < bounds_low[i] else exceeded_high}")

# Test uncontrolled for comparison
print(f"\n[BASELINE] Uncontrolled M7.4 (no TMD, no control)")
print("-" * 70)

env_uncontrolled = make_improved_tmd_env(test_file, max_force=1.0)
# Disable TMD
env_uncontrolled.tmd_k = 0.0
env_uncontrolled.tmd_c = 0.0

obs, _ = env_uncontrolled.reset()
done = False
peak_uncontrolled = 0

while not done:
    action = np.array([0.0])
    obs, reward, done, truncated, info = env_uncontrolled.step(action)
    peak_uncontrolled = max(peak_uncontrolled, abs(info['roof_displacement']))
    done = done or truncated

print(f"   Peak displacement (no TMD, no control): {peak_uncontrolled*100:.2f} cm")

# Test passive TMD
print(f"\n[BASELINE] Passive TMD (no active control)")
print("-" * 70)

env_passive = make_improved_tmd_env(test_file, max_force=150000.0)
obs, _ = env_passive.reset()
done = False
peak_passive = 0
peak_tmd_passive = 0

while not done:
    action = np.array([0.0])  # No control force
    obs, reward, done, truncated, info = env_passive.step(action)
    peak_passive = max(peak_passive, abs(info['roof_displacement']))
    peak_tmd_passive = max(peak_tmd_passive, abs(env_passive.displacement[12]))
    done = done or truncated

print(f"   Peak roof displacement (passive TMD): {peak_passive*100:.2f} cm")
print(f"   Peak TMD displacement (passive): {peak_tmd_passive*100:.2f} cm")

# Summary
print(f"\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)

print(f"\nPerformance Comparison (M7.4 earthquake):")
print(f"   Uncontrolled (no TMD):       {peak_uncontrolled*100:8.2f} cm")
print(f"   Passive TMD (no control):    {peak_passive*100:8.2f} cm")
print(f"   Active RL control (Stage 3): {peak_roof*100:8.2f} cm")

if peak_roof > peak_uncontrolled:
    print(f"\n❌ CRITICAL: RL control is WORSE than uncontrolled!")
    print(f"   Amplification: {peak_roof/peak_uncontrolled:.2f}x")
elif peak_roof > peak_passive:
    print(f"\n⚠️  WARNING: RL control is worse than passive TMD")
    print(f"   Degradation: {peak_roof/peak_passive:.2f}x")
else:
    print(f"\n✅ RL control is working")
    print(f"   Reduction: {(1 - peak_roof/peak_uncontrolled)*100:.1f}%")

if clipped_steps/total_steps > 0.05:
    print(f"\n❌ CRITICAL: {clipped_steps/total_steps*100:.1f}% observation clipping!")
    print(f"   This indicates TMD runaway or extreme building response")

if peak_tmd_abs > 3.0:  # 3 meters
    print(f"\n❌ CRITICAL: TMD displacement is {peak_tmd_abs*100:.0f} cm ({peak_tmd_abs:.1f} m)")
    print(f"   This is physically unrealistic - TMD is running away!")

print("\n" + "="*70)
