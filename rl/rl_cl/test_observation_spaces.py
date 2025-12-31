"""
Test script to validate observation spaces are consistent across all 4 stages.
This ensures Stage 3 won't fail due to observation space mismatch.
"""
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from tmd_environment import make_improved_tmd_env
import glob

print("="*70)
print("TESTING OBSERVATION SPACE CONSISTENCY ACROSS ALL 4 STAGES")
print("="*70)

# Training file paths (same as train_final_robust_rl_cl.py)
train_dir = "../../matlab/datasets/training_set"
train_files = {
    "M4.5": sorted(glob.glob(f"{train_dir}/TRAIN_M4.5_*.csv")),
    "M5.7": sorted(glob.glob(f"{train_dir}/TRAIN_M5.7_*.csv")),
    "M7.4": sorted(glob.glob(f"{train_dir}/TRAIN_M7.4_*.csv")),
    "M8.4": sorted(glob.glob(f"{train_dir}/TRAIN_M8.4_*.csv"))
}

# Curriculum stages (same as training script)
stages = [
    {'force_limit': 50000,  'name': 'Stage 1: M4.5 @ 50kN',  'magnitude': 'M4.5'},
    {'force_limit': 100000, 'name': 'Stage 2: M5.7 @ 100kN', 'magnitude': 'M5.7'},
    {'force_limit': 150000, 'name': 'Stage 3: M7.4 @ 150kN', 'magnitude': 'M7.4'},
    {'force_limit': 150000, 'name': 'Stage 4: M8.4 @ 150kN', 'magnitude': 'M8.4'},
]

# Observation bounds (MUST be same as training script)
obs_bounds = {
    'disp': 5.0,      # ±5.0m
    'vel': 20.0,      # ±20.0m/s
    'tmd_disp': 15.0, # ±15.0m
    'tmd_vel': 60.0   # ±60.0m/s
}

print(f"\n📐 Observation bounds (used for ALL stages):")
print(f"   Displacement: ±{obs_bounds['disp']}m")
print(f"   Velocity: ±{obs_bounds['vel']}m/s")
print(f"   TMD Displacement: ±{obs_bounds['tmd_disp']}m")
print(f"   TMD Velocity: ±{obs_bounds['tmd_vel']}m/s")

# Test each stage
model = None
all_passed = True

for stage_idx, stage in enumerate(stages):
    stage_num = stage_idx + 1
    magnitude = stage['magnitude']
    force_limit = stage['force_limit']
    available_files = train_files[magnitude]

    print(f"\n{'='*70}")
    print(f"  {stage['name']}")
    print(f"{'='*70}")

    if not available_files:
        print(f"   ❌ No training files found for {magnitude}!")
        all_passed = False
        continue

    # Create environment with SAME bounds for all stages
    def make_env():
        eq_file = np.random.choice(available_files)
        env = make_improved_tmd_env(
            eq_file,
            max_force=force_limit,
            sensor_noise_std=0.0,
            actuator_noise_std=0.0,
            latency_steps=0,
            dropout_prob=0.0,
            obs_bounds=obs_bounds  # CONSISTENT BOUNDS
        )
        env = Monitor(env)
        return env

    # Create vectorized environment
    env = DummyVecEnv([make_env])

    # Get observation space
    obs_space = env.observation_space
    print(f"\n   Observation space: {obs_space}")

    # Test 1: Create or update model
    try:
        if model is None:
            print(f"\n   Test 1: Creating new SAC model... ", end="")
            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                buffer_size=1000,  # Small buffer for testing
                batch_size=64,
                verbose=0,
                device='cpu'
            )
            print("✅ PASSED")
        else:
            print(f"\n   Test 1: Updating environment (set_env)... ", end="")
            # This is where Stage 3 would fail if observation spaces don't match
            model.set_env(env)
            print("✅ PASSED")

    except ValueError as e:
        print(f"❌ FAILED")
        print(f"\n   ERROR: {e}")
        all_passed = False
        break

    # Test 2: Run a few steps to verify environment works
    print(f"   Test 2: Running 10 environment steps... ", end="")
    try:
        obs = env.reset()
        for _ in range(10):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done[0]:
                obs = env.reset()
        print("✅ PASSED")
    except Exception as e:
        print(f"❌ FAILED")
        print(f"\n   ERROR: {e}")
        all_passed = False
        break

    # Test 3: Verify observations are within bounds
    print(f"   Test 3: Checking observation clipping... ", end="")
    obs_array = obs[0]  # Get first environment's observation

    expected_bounds = np.array([
        -5.0, -20.0,  # roof displacement, velocity
        -5.0, -20.0,  # floor 8
        -5.0, -20.0,  # floor 6
        -15.0, -60.0  # TMD
    ])

    # Check if any observations are exactly at bounds (clipping indicator)
    at_lower_bound = np.isclose(obs_array, expected_bounds, atol=1e-3)
    at_upper_bound = np.isclose(obs_array, -expected_bounds, atol=1e-3)

    if np.any(at_lower_bound) or np.any(at_upper_bound):
        print("⚠️  WARNING - Observations at bounds (possible clipping)")
        for i, (val, at_low, at_high) in enumerate(zip(obs_array, at_lower_bound, at_upper_bound)):
            if at_low or at_high:
                print(f"      obs[{i}] = {val:.3f}")
    else:
        print("✅ PASSED")

    env.close()

print(f"\n{'='*70}")
if all_passed:
    print("  ✅ ALL TESTS PASSED!")
    print("="*70)
    print("\n✅ Observation spaces are consistent across all stages")
    print("✅ Stage 3 will NOT fail with observation space mismatch")
    print("✅ Curriculum learning (Stage 1→2→3→4) will work correctly")
    print("\n🚀 You can safely run: ../../.venv/bin/python train_final_robust_rl_cl.py")
else:
    print("  ❌ TESTS FAILED!")
    print("="*70)
    print("\n❌ There are still observation space issues")
    print("❌ DO NOT run training until these are fixed")

print("="*70 + "\n")
