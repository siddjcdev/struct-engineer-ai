"""
v4 Training with PROPER Train/Test Split
=========================================

CRITICAL FIX:
- Train on: training_set/TRAIN_*.csv (synthetic variants)
- Test on: PEER_*.csv (held-out original earthquakes)
- Simple reward: -1.0 * disp - 0.3 * vel
- NO DCR penalty (conflicts with displacement minimization)
- NO force direction shaping (was teaching wrong behavior)
- Curriculum learning with domain randomization

Usage: python train_v4_proper_split.py
"""

import numpy as np
import os
import glob
import random
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from tmd_environment_shaped_reward import make_improved_tmd_env


def train_v4_proper_split():
    """Train with proper train/test split"""

    print("="*70)
    print("  v4 TRAINING - PROPER TRAIN/TEST SPLIT")
    print("="*70)
    print("\n📊 Dataset Strategy:")
    print("   TRAINING: Synthetic earthquakes (training_set/TRAIN_*.csv)")
    print("   TESTING:  Original PEER earthquakes (PEER_*.csv)")
    print("   → Ensures model doesn't memorize test set!\n")

    # Find training files
    train_dir = "../../matlab/datasets/training_set"
    train_files = {
        "M4.5": sorted(glob.glob(f"{train_dir}/TRAIN_M4.5_PGA0.25g_variant*.csv")),
        "M5.7": sorted(glob.glob(f"{train_dir}/TRAIN_M5.7_PGA0.35g_variant*.csv")),
        "M7.4": sorted(glob.glob(f"{train_dir}/TRAIN_M7.4_PGA0.75g_variant*.csv")),
        "M8.4": sorted(glob.glob(f"{train_dir}/TRAIN_M8.4_PGA0.9g_variant*.csv"))
    }

    # Test files (held-out)
    test_files = {
        "M4.5": "../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv",
        "M5.7": "../../matlab/datasets/PEER_moderate_M5.7_PGA0.35g.csv",
        "M7.4": "../../matlab/datasets/PEER_high_M7.4_PGA0.75g.csv",
        "M8.4": "../../matlab/datasets/PEER_insane_M8.4_PGA0.9g.csv"
    }

    # Verify training files exist
    print("📁 Training Files:")
    for mag, files in train_files.items():
        if not files:
            print(f"   ❌ {mag}: No training files found!")
            print(f"\n   ERROR: Training files missing!")
            print(f"   Expected location: {train_dir}/TRAIN_{mag}_*.csv\n")
            return None
        print(f"   ✓ {mag}: {len(files)} variants")

    print("\n📁 Test Files (held-out):")
    for mag, file in test_files.items():
        exists = "✓" if os.path.exists(file) else "❌"
        print(f"   {exists} {mag}: {os.path.basename(file)}")

    print("\n🎯 Reward Configuration (v4):")
    print("   • Displacement penalty: -1.0 (gentle, original)")
    print("   • Velocity penalty: -0.3 (gentle, original)")
    print("   • DCR penalty: DISABLED (conflicts with displacement)")
    print("   • Force direction: DISABLED (was teaching wrong behavior)")
    print("   • Let agent discover optimal control\n")

    # Curriculum stages
    print("🎯 Curriculum Plan:")
    stages = [
        {'force_limit': 50000,  'timesteps': 150000, 'name': 'M4.5 @ 50kN',  'magnitude': 'M4.5'},
        {'force_limit': 100000, 'timesteps': 150000, 'name': 'M5.7 @ 100kN', 'magnitude': 'M5.7'},
        {'force_limit': 150000, 'timesteps': 200000, 'name': 'M7.4 @ 150kN', 'magnitude': 'M7.4'},
        {'force_limit': 150000, 'timesteps': 200000, 'name': 'M8.4 @ 150kN', 'magnitude': 'M8.4'},
    ]

    for i, stage in enumerate(stages, 1):
        n_variants = len(train_files[stage['magnitude']])
        print(f"   Stage {i}: {stage['name']} - {stage['timesteps']:,} steps ({n_variants} training variants)")

    print(f"\n   Total timesteps: {sum(s['timesteps'] for s in stages):,}")

    # Create directory
    os.makedirs("models/rl_v4_proper_split", exist_ok=True)

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
        print(f"  STAGE {stage_num}/{len(stages)}: {stage['name']}")
        print(f"{'='*70}\n")
        print(f"   Training variants: {len(available_files)}")
        for f in available_files:
            print(f"      - {os.path.basename(f)}")

        # Create environment that randomly samples from training variants
        def make_env():
            # Randomly pick a training variant each episode
            eq_file = random.choice(available_files)
            env = make_improved_tmd_env(eq_file, max_force=force_limit)
            env = Monitor(env)
            return env

        env = DummyVecEnv([make_env])

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
                device='cpu'
            )
        else:
            print(f"\n🔄 Continuing from Stage {stage_num-1}...")
            model.set_env(env)

        # Train
        print(f"\n🚀 Training...")
        model.learn(
            total_timesteps=timesteps,
            reset_num_timesteps=False,
            progress_bar=True
        )

        # Save
        save_path = f"models/rl_v4_proper_split/stage{stage_num}_{force_limit//1000}kN.zip"
        model.save(save_path)
        print(f"\n💾 Saved: {save_path}")

        # Test on HELD-OUT test set
        test_file = test_files[magnitude]
        print(f"\n📊 Testing on HELD-OUT test earthquake...")
        print(f"   Test file: {os.path.basename(test_file)}")

        test_env = make_improved_tmd_env(test_file, max_force=force_limit)
        obs, _ = test_env.reset()
        done = False
        peak = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            peak = max(peak, abs(info['roof_displacement']))
            done = done or truncated

        peak_cm = peak * 100
        print(f"   Peak displacement: {peak_cm:.2f} cm")
        print(f"\n✅ Stage {stage_num} complete!\n")

    # Final model save
    training_time = datetime.now() - start_time
    final_path = "models/rl_v4_proper_split/final_v4_proper.zip"
    model.save(final_path)

    print("="*70)
    print("  🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"\n   Total training time: {training_time}")
    print(f"   Final model: {final_path}")
    print(f"\n   This model:")
    print(f"   • Trained on SYNTHETIC variants (training_set/)")
    print(f"   • Tested on HELD-OUT PEER earthquakes")
    print(f"   • Used simple reward (-1.0 * disp, -0.3 * vel)")
    print(f"   • No DCR penalty (let it emerge naturally)")
    print(f"   • No force direction shaping")
    print(f"   • {sum(s['timesteps'] for s in stages):,} total timesteps")

    # Final comprehensive test on ALL held-out earthquakes
    print(f"\n{'='*70}")
    print(f"  FINAL TEST ON ALL HELD-OUT EARTHQUAKES")
    print(f"{'='*70}\n")

    results = []
    for magnitude, test_file in test_files.items():
        test_env = make_improved_tmd_env(test_file, max_force=150000)
        obs, _ = test_env.reset()
        done = False
        peak = 0
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            peak = max(peak, abs(info['roof_displacement']))
            total_reward += reward
            done = done or truncated

        peak_cm = peak * 100
        episode_metrics = test_env.get_episode_metrics()
        dcr = episode_metrics.get('dcr', 0.0)

        results.append({
            'magnitude': magnitude,
            'peak_cm': peak_cm,
            'dcr': dcr,
            'total_reward': total_reward
        })

        print(f"   {magnitude} ({os.path.basename(test_file)}):")
        print(f"      Peak displacement: {peak_cm:.2f} cm")
        print(f"      DCR: {dcr:.2f}")
        print(f"      Total reward: {total_reward:.2f}\n")

    print("="*70 + "\n")

    return model


if __name__ == "__main__":
    print("\n🚀 Starting v4 Training with PROPER Train/Test Split...\n")
    model = train_v4_proper_split()
    if model is not None:
        print("\n✅ Training complete! Model ready for evaluation.")
    else:
        print("\n❌ Training failed - check training files.")
