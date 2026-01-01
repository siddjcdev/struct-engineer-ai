"""
Train with Shaped Rewards + Curriculum Learning
================================================

v4: Back to basics - simple reward with curriculum learning
- Displacement penalty: -1.0 (original)
- Velocity penalty: -0.3 (original)
- NO force direction bonus (was teaching wrong behavior)
- NO DCR penalty (conflicts with displacement minimization)
- 4-stage curriculum learning
- 700K total timesteps

Usage: python train_shaped_rewards.py
"""

import numpy as np
import os
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from tmd_environment_shaped_reward import make_improved_tmd_env


def train_shaped_rewards_curriculum():
    """Train with simple reward + curriculum learning"""

    print("="*70)
    print("  Training with Simple Reward + Curriculum Learning (v4)")
    print("="*70)

    # PEER earthquakes for curriculum
    earthquake_files = [
        "../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv",
        "../../matlab/datasets/PEER_moderate_M5.7_PGA0.35g.csv",
        "../../matlab/datasets/PEER_high_M7.4_PGA0.75g.csv",
        "../../matlab/datasets/PEER_insane_M8.4_PGA0.9g.csv"
    ]

    print("\n📊 Training Configuration:")
    print("   Using PEER earthquakes with curriculum learning")
    for i, eq in enumerate(earthquake_files, 1):
        print(f"   {i}. {os.path.basename(eq)}")

    print("\n🎯 Reward Features (v4 - Back to Basics):")
    print("   • Displacement penalty: -1.0 (original, gentle)")
    print("   • Velocity penalty: -0.3 (original, gentle)")
    print("   • Force direction bonus: DISABLED (was teaching wrong behavior)")
    print("   • DCR penalty: DISABLED (conflicts with displacement)")
    print("   • Let agent discover optimal control through exploration")

    # Curriculum stages
    print("\n🎯 Curriculum Plan:")
    stages = [
        {'force_limit': 50000,  'timesteps': 150000, 'name': 'M4.5 @ 50kN',  'eq_idx': 0},
        {'force_limit': 100000, 'timesteps': 150000, 'name': 'M5.7 @ 100kN', 'eq_idx': 1},
        {'force_limit': 150000, 'timesteps': 200000, 'name': 'M7.4 @ 150kN', 'eq_idx': 2},
        {'force_limit': 150000, 'timesteps': 200000, 'name': 'M8.4 @ 150kN', 'eq_idx': 3},
    ]

    for i, stage in enumerate(stages, 1):
        print(f"   Stage {i}: {stage['name']} - {stage['timesteps']:,} steps")

    print(f"\n   Total timesteps: {sum(s['timesteps'] for s in stages):,}")

    # Create directory
    os.makedirs("models/rl_shaped_rewards_curriculum", exist_ok=True)

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
        print(f"  STAGE {stage_num}/{len(stages)}: {stage['name']}")
        print(f"{'='*70}\n")
        print(f"   Earthquake: {os.path.basename(eq_file)}")
        print(f"   Force limit: {force_limit/1000:.0f} kN")
        print(f"   Timesteps: {timesteps:,}")

        # Create environment
        def make_env():
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
        save_path = f"models/rl_shaped_rewards_curriculum/stage{stage_num}_{force_limit//1000}kN.zip"
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

    # Final model save
    training_time = datetime.now() - start_time
    final_path = "models/rl_shaped_rewards_curriculum/final_v4_curriculum.zip"
    model.save(final_path)

    print("="*70)
    print("  🎉 CURRICULUM TRAINING COMPLETE!")
    print("="*70)
    print(f"\n   Total training time: {training_time}")
    print(f"   Final model: {final_path}")
    print(f"\n   This model:")
    print(f"   • Used simple reward (-1.0 * disp, -0.3 * vel)")
    print(f"   • No force direction shaping (let agent discover)")
    print(f"   • No DCR penalty (let it emerge naturally)")
    print(f"   • 4-stage curriculum learning")
    print(f"   • {sum(s['timesteps'] for s in stages):,} total timesteps")

    # Final comprehensive test
    print(f"\n{'='*70}")
    print(f"  FINAL TESTING ON ALL EARTHQUAKES")
    print(f"{'='*70}\n")

    results = []
    for eq_file in earthquake_files:
        eq_name = os.path.basename(eq_file)
        test_env = make_improved_tmd_env(eq_file, max_force=150000)
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
            'earthquake': eq_name,
            'peak_cm': peak_cm,
            'dcr': dcr,
            'total_reward': total_reward
        })

        print(f"   {eq_name}:")
        print(f"      Peak displacement: {peak_cm:.2f} cm")
        print(f"      DCR: {dcr:.2f}")
        print(f"      Total reward: {total_reward:.2f}\n")

    print("="*70 + "\n")

    return model


if __name__ == "__main__":
    print("\n🚀 Starting Curriculum Training (v4 - Simple Reward)...\n")
    model = train_shaped_rewards_curriculum()
    if model is not None:
        print("\n✅ Curriculum training complete!")
    else:
        print("\n❌ Training failed.")
