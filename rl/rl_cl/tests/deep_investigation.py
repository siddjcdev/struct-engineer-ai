"""
Deep investigation of training vs testing mismatch
"""
import sys
sys.path.insert(0, '/Users/Shared/dev/git/struct-engineer-ai')
import numpy as np
from rl.rl_cl.tmd_environment import make_improved_tmd_env
from stable_baselines3 import SAC

print("="*70)
print("DEEP INVESTIGATION: TRAINING VS TESTING MISMATCH")
print("="*70)

# Load the Stage 1 model
model_path = "rl_cl_robust_models/stage1_50kN_final_robust.zip"
model = SAC.load(model_path)

print(f"\n✅ Loaded model: {model_path}")

# Test on M4.5 with DIFFERENT environment configurations
test_file = "../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv"

configs = [
    {"name": "Clean (no augmentation)", "kwargs": {"max_force": 50000}},
    {"name": "With sensor noise 5%", "kwargs": {"max_force": 50000, "sensor_noise_std": 0.05}},
    {"name": "With actuator noise 2.5%", "kwargs": {"max_force": 50000, "actuator_noise_std": 0.025}},
    {"name": "With latency 20ms", "kwargs": {"max_force": 50000, "latency_steps": 1}},
    {"name": "With all augmentation", "kwargs": {"max_force": 50000, "sensor_noise_std": 0.05, "actuator_noise_std": 0.025, "latency_steps": 1}},
]

print("\n" + "-"*70)
print("TESTING MODEL UNDER DIFFERENT CONDITIONS")
print("-"*70)

for config in configs:
    env = make_improved_tmd_env(test_file, **config["kwargs"])
    obs, _ = env.reset()
    done = False
    peak = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        peak = max(peak, abs(info['roof_displacement']))
        done = done or truncated
    
    print(f"{config['name']:30s}: {peak*100:6.2f} cm")

# Now check the OBSERVATION SPACE bounds
print("\n" + "="*70)
print("OBSERVATION SPACE ANALYSIS")
print("="*70)

env = make_improved_tmd_env(test_file, max_force=50000)
obs, _ = env.reset()

print(f"\nObservation space bounds:")
print(f"  Low:  {env.observation_space.low}")
print(f"  High: {env.observation_space.high}")

# Run episode and track observations
obs_min = obs.copy()
obs_max = obs.copy()
done = False
step = 0

while not done and step < 200:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    obs_min = np.minimum(obs_min, obs)
    obs_max = np.maximum(obs_max, obs)
    done = done or truncated
    step += 1

print(f"\nActual observation ranges (first 200 steps with RL control):")
obs_labels = ['roof_disp', 'roof_vel', 'floor8_disp', 'floor8_vel',
              'floor6_disp', 'floor6_vel', 'tmd_disp', 'tmd_vel']

for i, label in enumerate(obs_labels):
    bounded = "✅" if obs_min[i] >= env.observation_space.low[i] and obs_max[i] <= env.observation_space.high[i] else "❌"
    print(f"  {label:12s}: [{obs_min[i]:+.4f}, {obs_max[i]:+.4f}]  {bounded}")

# Check model's policy network
print("\n" + "="*70)
print("MODEL ARCHITECTURE")
print("="*70)
print(f"Policy network: {model.policy}")
print(f"Observation normalization: {model.policy.normalize_images}")

# Check reward during evaluation
print("\n" + "="*70)
print("REWARD ANALYSIS")
print("="*70)

env = make_improved_tmd_env(test_file, max_force=50000)
obs, _ = env.reset()
done = False
total_reward = 0
step = 0

while not done and step < 50:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    done = done or truncated
    step += 1

print(f"Average reward (first 50 steps): {total_reward/step:.2f}")
print(f"Steps taken: {step}")
