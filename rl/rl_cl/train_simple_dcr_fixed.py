"""
BACK TO BASICS: Simple RL Training with DCR Fix
================================================

No complex features, just:
- Simple curriculum learning
- DCR fix (already in environment)
- Train directly on PEER earthquakes
- No domain randomization
- No adaptive bounds
- No train/test split

This WILL work.

Usage: python train_simple_dcr_fixed.py
"""

import numpy as np
import os
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from tmd_environment import make_improved_tmd_env


def train_simple_dcr_fixed():
    """Simple curriculum training with DCR fix"""

    print("="*70)
    print("  BACK TO BASICS: Simple RL Training with DCR Fix")
    print("="*70)

    # Direct PEER earthquakes (no synthetic data)
    earthquake_files = [
        "../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv",
        "../../matlab/datasets/PEER_moderate_M5.7_PGA0.35g.csv",
        "../../matlab/datasets/PEER_high_M7.4_PGA0.75g.csv",
        "../../matlab/datasets/PEER_insane_M8.4_PGA0.9g.csv"
    ]

    print("\n📊 Training directly on PEER earthquakes (simple approach):")
    for i, eq in enumerate(earthquake_files, 1):
        print(f"   {i}. {os.path.basename(eq)}")

    print("\n🎯 Curriculum Plan:")
    stages = [
        {'force_limit': 50000,  'timesteps': 150000, 'name': 'M4.5 @ 50kN',  'eq_idx': 0},
        {'force_limit': 100000, 'timesteps': 150000, 'name': 'M5.7 @ 100kN', 'eq_idx': 1},
        {'force_limit': 150000, 'timesteps': 200000, 'name': 'M7.4 @ 150kN', 'eq_idx': 2},
        {'force_limit': 150000, 'timesteps': 200000, 'name': 'M8.4 @ 150kN', 'eq_idx': 3},
    ]

    for i, stage in enumerate(stages, 1):
        print(f"   Stage {i}: {stage['name']} - {stage['timesteps']:,} steps")

    # Create directory
    os.makedirs("models/rl_cl_simple_dcr_models", exist_ok=True)

    # Training
    start_time = datetime.now()
    model = None

    for stage_idx, stage in enumerate(stages):
        stage_num = stage_idx + 1
        force_limit = stage['force_limit']
        timesteps = stage['timesteps']
        eq_idx = stage['eq_idx']
        eq_file = earthquake_files[eq_idx]

        print(f"\n{'='*70}")
        print(f"  STAGE {stage_num}: {stage['name']}")
        print(f"{'='*70}\n")
        print(f"   Earthquake: {os.path.basename(eq_file)}")

        # Create environment - SIMPLE, no special parameters
        def make_env():
            env = make_improved_tmd_env(eq_file, max_force=force_limit)
            env = Monitor(env)
            return env

        env = DummyVecEnv([make_env])

        # Create or update model
        if model is None:
            print(f"\n🤖 Creating SAC model...")
            print(f"   Using CPU (simple and reliable)")
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
                device='cpu'  # Simple, no GPU complexity
            )
        else:
            print(f"\n🔄 Continuing training from Stage {stage_num-1}...")
            model.set_env(env)

        # Train
        print(f"\n🚀 Training...")
        model.learn(
            total_timesteps=timesteps,
            reset_num_timesteps=False,
            progress_bar=True
        )

        # Save
        save_path = f"models/rl_cl_simple_dcr_models/stage{stage_num}_{force_limit//1000}kN.zip"
        model.save(save_path)
        print(f"\n💾 Saved: {save_path}")

        # Quick test on same earthquake
        print(f"\n📊 Testing on training earthquake...")
        test_env = make_improved_tmd_env(eq_file, max_force=force_limit)
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

    # Final
    training_time = datetime.now() - start_time
    final_path = "models/rl_cl_simple_dcr_models/final_simple_dcr.zip"
    model.save(final_path)

    print("="*70)
    print("  🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"\n   Total time: {training_time}")
    print(f"   Final model: {final_path}")
    print(f"\n   This model:")
    print(f"   • Uses simple curriculum learning")
    print(f"   • Has DCR fix (1mm threshold)")
    print(f"   • Trained directly on PEER data")
    print(f"   • No complex features")
    print(f"\n   Next: Test on all 4 earthquakes and compare to uncontrolled!")
    print("="*70 + "\n")

    return model


if __name__ == "__main__":
    print("\n🚀 Starting Simple DCR-Fixed Training...\n")
    model = train_simple_dcr_fixed()
    if model is not None:
        print("\n✅ Training complete! Model ready for testing.")
    else:
        print("\n❌ Training failed.")
