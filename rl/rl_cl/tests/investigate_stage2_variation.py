"""
Investigate why Stage 2 performance varies between tests
"""
import numpy as np
from stable_baselines3 import SAC
from tmd_environment import make_improved_tmd_env
import os

print("\n" + "="*70)
print("INVESTIGATING STAGE 2 PERFORMANCE VARIATION")
print("="*70)
print("\nObserved results:")
print("  - Earlier test: 7.34 cm")
print("  - Current test: 48.82 cm")
print("\nPossible causes:")
print("  1. Different model versions (stage2 vs final)")
print("  2. Domain randomization enabled vs disabled")
print("  3. Different test files")
print("  4. Model loading issues")
print("="*70)

# Check what models are available
print("\n[CHECKING] Available model files...")
model_dir = "rl_cl_robust_models_5_datafix"
if os.path.exists(model_dir):
    models = [f for f in os.listdir(model_dir) if f.endswith('.zip')]
    print(f"Found {len(models)} models in {model_dir}:")
    for m in sorted(models):
        size = os.path.getsize(os.path.join(model_dir, m)) / 1024 / 1024
        print(f"  - {m} ({size:.1f} MB)")
else:
    print(f"❌ Directory not found: {model_dir}")

# Test file
test_file = "../../matlab/datasets/PEER_moderate_M5.7_PGA0.35g.csv"
if not os.path.exists(test_file):
    print(f"\n❌ Test file not found: {test_file}")
    exit(1)

print(f"\n[TEST 1] Stage 2 Model (stage2_100kN_final_robust.zip)")
print("-" * 70)

model_path = f"{model_dir}/stage2_100kN_final_robust.zip"
if os.path.exists(model_path):
    model = SAC.load(model_path)

    # Test WITHOUT domain randomization (clean)
    env = make_improved_tmd_env(test_file, max_force=100000.0)
    obs, _ = env.reset()
    done = False
    peak = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        peak = max(peak, abs(info['roof_displacement']))
        done = done or truncated

    print(f"Peak displacement (NO augmentation): {peak*100:.2f} cm")

    # Test WITH domain randomization
    env_aug = make_improved_tmd_env(
        test_file,
        max_force=100000.0,
        sensor_noise_std=0.05,
        actuator_noise_std=0.025,
        latency_steps=1,
        dropout_prob=0.04
    )
    obs, _ = env_aug.reset()
    done = False
    peak_aug = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env_aug.step(action)
        peak_aug = max(peak_aug, abs(info['roof_displacement']))
        done = done or truncated

    print(f"Peak displacement (WITH augmentation): {peak_aug*100:.2f} cm")
else:
    print(f"❌ Model not found: {model_path}")

print(f"\n[TEST 2] Final Model (rl_cl_models_alpha_1.zip)")
print("-" * 70)

final_model_path = f"{model_dir}/rl_cl_models_alpha_1.zip"
if os.path.exists(final_model_path):
    model_final = SAC.load(final_model_path)

    env = make_improved_tmd_env(test_file, max_force=100000.0)
    obs, _ = env.reset()
    done = False
    peak_final = 0

    while not done:
        action, _ = model_final.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        peak_final = max(peak_final, abs(info['roof_displacement']))
        done = done or truncated

    print(f"Peak displacement (final model): {peak_final*100:.2f} cm")
else:
    print(f"❌ Model not found: {final_model_path}")

# Check uncontrolled baseline
print(f"\n[BASELINE] Uncontrolled (no TMD, no control)")
print("-" * 70)

env_uncontrolled = make_improved_tmd_env(test_file, max_force=1.0)
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

print(f"Peak displacement (uncontrolled): {peak_uncontrolled*100:.2f} cm")

# Check passive TMD
print(f"\n[BASELINE] Passive TMD (no control)")
print("-" * 70)

env_passive = make_improved_tmd_env(test_file, max_force=100000.0)
obs, _ = env_passive.reset()
done = False
peak_passive = 0

while not done:
    action = np.array([0.0])
    obs, reward, done, truncated, info = env_passive.step(action)
    peak_passive = max(peak_passive, abs(info['roof_displacement']))
    done = done or truncated

print(f"Peak displacement (passive TMD): {peak_passive*100:.2f} cm")

# Summary
print(f"\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nPerformance on M5.7 earthquake:")
print(f"  Uncontrolled:        {peak_uncontrolled*100:6.2f} cm")
print(f"  Passive TMD:         {peak_passive*100:6.2f} cm")
if os.path.exists(model_path):
    print(f"  Stage 2 model:       {peak*100:6.2f} cm (clean)")
    print(f"  Stage 2 model:       {peak_aug*100:6.2f} cm (augmented)")
if os.path.exists(final_model_path):
    print(f"  Final model:         {peak_final*100:6.2f} cm")

print(f"\n💡 Key insights:")
print(f"  - Stage 2 model (after Stage 2 training): optimized for M5.7")
print(f"  - Final model (after Stage 3-4 training): may have forgotten Stage 2 performance")
print(f"  - This is called 'catastrophic forgetting' in continual learning")

print("\n" + "="*70)
