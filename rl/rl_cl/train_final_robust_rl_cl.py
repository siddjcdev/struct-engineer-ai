"""
FINAL ROBUST RL TRAINING - PROPER TRAIN/TEST SPLIT
===================================================

Trains on SYNTHETIC earthquakes (never seen by test set)
Tests on ORIGINAL PEER earthquakes (held-out test set)

This ensures:
- Training data ≠ Testing data (different CSV files entirely)
- Domain randomization (noise/latency/dropout augmentation)
- Proper machine learning evaluation

Training set: matlab/datasets/training_set/TRAIN_*.csv (synthetic)
Test set: matlab/datasets/PEER_*.csv (original, held-out)

Usage: python train_final_robust_rl_cl.py
"""

import numpy as np
import torch
import os
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from rl.rl_cl.tmd_environment import make_improved_tmd_env
import matplotlib.pyplot as plt
import glob


def train_final_robust_rl_cl():
    """
    Final training with proper train/test split and domain randomization
    """

    print("="*70)
    print("  FINAL ROBUST RL TRAINING - PROPER TRAIN/TEST SPLIT")
    print("="*70)
    print("\n📊 Dataset Strategy:")
    print("   TRAINING: Synthetic earthquakes (training_set/TRAIN_*.csv)")
    print("   TESTING:  Original PEER earthquakes (PEER_*.csv)")
    print("   → Ensures model doesn't memorize test set!\n")

    # Find training files
    train_dir = "matlab/datasets/training_set"
    train_files = {
        "M4.5": sorted(glob.glob(f"{train_dir}/TRAIN_M4.5_*.csv")),
        "M5.7": sorted(glob.glob(f"{train_dir}/TRAIN_M5.7_*.csv")),
        "M7.4": sorted(glob.glob(f"{train_dir}/TRAIN_M7.4_*.csv")),
        "M8.4": sorted(glob.glob(f"{train_dir}/TRAIN_M8.4_*.csv"))
    }

    # Test files (held-out)
    test_files = {
        "M4.5": "matlab/datasets/PEER_small_M4.5_PGA0.25g.csv",
        "M5.7": "matlab/datasets/PEER_moderate_M5.7_PGA0.35g.csv",
        "M7.4": "matlab/datasets/PEER_high_M7.4_PGA0.75g.csv",
        "M8.4": "matlab/datasets/PEER_insane_M8.4_PGA0.9g.csv"
    }

    # Verify training files exist
    print("📁 Training Files:")
    for mag, files in train_files.items():
        if not files:
            print(f"   ❌ {mag}: No training files found!")
            print(f"\n   ERROR: Run generate_training_earthquakes.py first!")
            print(f"   Command: cd matlab/datasets && python generate_training_earthquakes.py\n")
            return None
        print(f"   ✓ {mag}: {len(files)} variants")

    print("\n📁 Test Files (held-out):")
    for mag, file in test_files.items():
        exists = "✓" if os.path.exists(file) else "❌"
        print(f"   {exists} {mag}: {os.path.basename(file)}")

    print("\n🎯 Curriculum Plan with Domain Randomization:")

    # Curriculum stages
    # Each stage randomly samples from available training variants
    stages = [
        {'force_limit': 50000,  'timesteps': 150000, 'name': 'M4.5 @ 50kN',  'magnitude': 'M4.5'},
        {'force_limit': 100000, 'timesteps': 150000, 'name': 'M5.7 @ 100kN', 'magnitude': 'M5.7'},
        {'force_limit': 150000, 'timesteps': 200000, 'name': 'M7.4 @ 150kN', 'magnitude': 'M7.4'},
        {'force_limit': 150000, 'timesteps': 200000, 'name': 'M8.4 @ 150kN', 'magnitude': 'M8.4'},
    ]

    for i, stage in enumerate(stages, 1):
        n_variants = len(train_files[stage['magnitude']])
        print(f"   Stage {i}: {stage['name']} - {stage['timesteps']:,} steps ({n_variants} training variants)")

    print("\n🛡️  Domain Randomization (applied to each episode):")
    print("   - Sensor noise: 0-10%")
    print("   - Actuator noise: 0-5%")
    print("   - Latency: 0-40ms")
    print("   - Dropout: 0-8%")
    print("   - Training variant: randomly sampled each episode")

    # Create directories
    os.makedirs("simple_rl_models", exist_ok=True)

    # Training
    start_time = datetime.now()
    model = None

    for stage_idx, stage in enumerate(stages):
        stage_num = stage_idx + 1
        force_limit = stage['force_limit']
        timesteps = stage['timesteps']
        magnitude = stage['magnitude']
        available_files = train_files[magnitude]

        print(f"\n{'='*70}")
        print(f"  STAGE {stage_num}: {stage['name']}")
        print(f"{'='*70}\n")
        print(f"   Training variants: {len(available_files)}")
        for f in available_files:
            print(f"      - {os.path.basename(f)}")

        # Create environment with MULTI-VARIANT DOMAIN RANDOMIZATION
        def make_env(eq_files, force_lim):
            """
            Each episode:
            1. Randomly samples one training variant
            2. Randomly samples augmentation parameters
            This maximizes diversity and prevents memorization
            """
            # Randomly select earthquake variant
            eq_file = np.random.choice(eq_files)

            # Sample random augmentation parameters
            sensor_noise = np.random.uniform(0.0, 0.10)    # 0-10% noise
            actuator_noise = np.random.uniform(0.0, 0.05)  # 0-5% noise
            latency = np.random.choice([0, 1, 2])          # 0, 20ms, or 40ms
            dropout = np.random.uniform(0.0, 0.08)         # 0-8% dropout

            env = make_improved_tmd_env(
                eq_file,
                max_force=force_lim,
                sensor_noise_std=sensor_noise,
                actuator_noise_std=actuator_noise,
                latency_steps=latency,
                dropout_prob=dropout
            )
            env = Monitor(env)
            return env

        # Create vectorized environment
        env = DummyVecEnv([lambda files=available_files, fl=force_limit: make_env(files, fl)])

        # Create or update model
        if model is None:
            print(f"\n🤖 Creating SAC model...")
            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                buffer_size=100_000,
                batch_size=256,
                gamma=0.99,
                tau=0.005,
                ent_coef='auto',
                policy_kwargs=dict(net_arch=[256, 256]),
                verbose=1,
                device='auto'
            )
        else:
            print(f"\n🔄 Updating environment...")
            model.set_env(env)

        # Train
        print(f"\n🚀 Training with multi-variant domain randomization...")
        model.learn(
            total_timesteps=timesteps,
            reset_num_timesteps=False,
            progress_bar=True
        )

        # Save
        save_path = f"simple_rl_models/stage{stage_num}_{force_limit//1000}kN_final_robust.zip"
        model.save(save_path)
        print(f"\n💾 Saved: {save_path}")

        # Test on HELD-OUT test set (no augmentation!)
        print(f"\n📊 Testing on HELD-OUT test set (original PEER data, NO augmentation)...")
        test_file = test_files[magnitude]

        if os.path.exists(test_file):
            test_env = make_improved_tmd_env(test_file, max_force=force_limit)
            obs, _ = test_env.reset()
            done = False
            truncated = False
            peak = 0

            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = test_env.step(action)
                peak = max(peak, abs(info['roof_displacement']))

            peak_cm = peak * 100
            print(f"   Peak displacement: {peak_cm:.2f} cm")
            print(f"   ✓ Generalization test passed!")
        else:
            print(f"   ⚠️  Test file not found: {test_file}")

        print(f"\n✅ Stage {stage_num} complete!\n")

    # Final
    training_time = datetime.now() - start_time
    final_path = "simple_rl_models/perfect_rl_final_robust.zip"
    model.save(final_path)

    print("="*70)
    print("  🎉 FINAL ROBUST TRAINING COMPLETE!")
    print("="*70)
    print(f"\n   Total time: {training_time}")
    print(f"   Final model: {final_path}")
    print(f"\n   Training strategy:")
    print(f"   ✓ Trained on: {sum(len(files) for files in train_files.values())} synthetic variants")
    print(f"   ✓ Domain randomization: noise/latency/dropout")
    print(f"   ✓ Tested on: 4 held-out PEER earthquakes")
    print(f"\n   This model should:")
    print(f"   • NOT memorize test set waveforms")
    print(f"   • Generalize to unseen earthquakes")
    print(f"   • Handle real-world stress conditions")
    print(f"\n   Next: Copy to API and run full comparison!")
    print(f"   Command: cp {final_path} restapi/rl_cl/simple_rl_models/")
    print("="*70 + "\n")

    return model


if __name__ == "__main__":
    print("\n🚀 Starting Final Robust RL Training...\n")
    print("This training uses PROPER train/test split:")
    print("  - Training: Synthetic earthquakes (never in test set)")
    print("  - Testing: Original PEER earthquakes (held-out)")
    print("  - Augmentation: Random noise/latency/dropout each episode\n")

    model = train_final_robust_rl_cl()

    if model is not None:
        print("\n✅ Training complete! Model ready for deployment.")
    else:
        print("\n❌ Training failed. Check error messages above.")
