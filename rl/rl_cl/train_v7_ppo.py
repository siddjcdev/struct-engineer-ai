"""
v7 Training - ADAPTIVE REWARD SCALING with PPO
===============================================

PPO VERSION of v7 adaptive training!

INTELLIGENT FIX: Magnitude-adaptive reward scaling!

Problem with v6:
- 10× reward helped M5.7 (+10.2% improvement)
- 10× reward hurt M7.4 (-10.6% worse than v5)
- One-size-fits-all approach doesn't work!

v7 Solution - Adaptive Scaling:
- M4.5: 3× multiplier (gentle - avoid over-penalizing)
- M5.7: 7× multiplier (strong - showed best results)
- M7.4: 4× multiplier (balanced - between v5's 5× and v6's 10×)
- M8.4: 3× multiplier (conservative - extreme earthquakes)

PPO vs SAC:
- PPO: On-policy, more stable, better for continuous control
- May need slightly more timesteps than SAC
- No replay buffer, uses GAE and clipped updates

Expected Results:
- M4.5: ~19 cm (match v4's best)
- M5.7: ~40 cm (maintain v6's improvement)
- M7.4: ~210 cm (beat v5's 229cm and v6's 260cm)
- M8.4: ~320 cm (show consistent learning)

Usage: python train_v7_ppo.py
"""

import numpy as np
import os
import glob
import random
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from tmd_environment_adaptive_reward import make_improved_tmd_env


def train_v7_ppo():
    """Train with magnitude-adaptive reward scaling using PPO"""

    print("="*70)
    print("  v7 TRAINING - ADAPTIVE REWARD SCALING (PPO)")
    print("="*70)
    print("\n📊 Dataset Strategy:")
    print("   TRAINING: Improved synthetic earthquakes (training_set_v2/)")
    print("   → 10 variants per magnitude")
    print("   TESTING:  Original PEER earthquakes (PEER_*.csv)")
    print("   → Ensures model doesn't memorize test set!\n")

    print("🎯 Adaptive Reward Scaling Strategy:")
    print("   M4.5 (PGA 0.25g): 3× multiplier (gentle)")
    print("   M5.7 (PGA 0.35g): 7× multiplier (strong - best in v6!)")
    print("   M7.4 (PGA 0.75g): 4× multiplier (balanced)")
    print("   M8.4 (PGA 0.90g): 3× multiplier (conservative)")
    print("   → Auto-detected based on earthquake PGA\n")

    print("🤖 Algorithm: PPO (Proximal Policy Optimization)")
    print("   • On-policy learning (no replay buffer)")
    print("   • Clipped objective for stable updates")
    print("   • GAE for advantage estimation\n")

    # Find training files (v2 dataset with 10 variants)
    train_dir = "../../matlab/datasets/training_set_v2"
    train_files = {
        "M4.5": sorted(glob.glob(f"{train_dir}/TRAIN_M4.5_*.csv")),
        "M5.7": sorted(glob.glob(f"{train_dir}/TRAIN_M5.7_*.csv")),
        "M7.4": sorted(glob.glob(f"{train_dir}/TRAIN_M7.4_*.csv")),
        "M8.4": sorted(glob.glob(f"{train_dir}/TRAIN_M8.4_*.csv"))
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
            print(f"\n   ERROR: Run generate_training_earthquakes_v2.py first!")
            return None
        print(f"   ✓ {mag}: {len(files)} variants")

    print("\n📁 Test Files (held-out):")
    for mag, file in test_files.items():
        exists = "✓" if os.path.exists(file) else "❌"
        print(f"   {exists} {mag}: {os.path.basename(file)}")

    # Curriculum stages
    print("\n🎯 Curriculum Plan:")
    stages = [
        {'force_limit': 50000,  'timesteps': 150000, 'name': 'M4.5 @ 50kN',  'magnitude': 'M4.5'},
        {'force_limit': 100000, 'timesteps': 150000, 'name': 'M5.7 @ 100kN', 'magnitude': 'M5.7'},
        {'force_limit': 150000, 'timesteps': 200000, 'name': 'M7.4 @ 150kN', 'magnitude': 'M7.4'},
        {'force_limit': 150000, 'timesteps': 200000, 'name': 'M8.4 @ 150kN', 'magnitude': 'M8.4'},
    ]

    for i, stage in enumerate(stages, 1):
        n_variants = len(train_files[stage['magnitude']])
        print(f"   Stage {i}: {stage['name']} - {stage['timesteps']:,} steps ({n_variants} variants)")

    print(f"\n   Total timesteps: {sum(s['timesteps'] for s in stages):,}")

    # Create directory
    os.makedirs("models/rl_v7_ppo", exist_ok=True)

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

        # Create environment with adaptive reward (will auto-detect scaling)
        def make_env():
            eq_file = random.choice(available_files)
            env = make_improved_tmd_env(eq_file, max_force=force_limit)
            env = Monitor(env)
            return env

        env = DummyVecEnv([make_env])

        # Create or update model
        if model is None:
            print(f"\n🤖 Creating PPO model...")
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=2048,        # PPO collects this many steps before update
                batch_size=64,       # Smaller batches for PPO
                n_epochs=10,         # PPO-specific: epochs per update
                gamma=0.99,
                gae_lambda=0.95,     # PPO-specific: GAE parameter
                clip_range=0.2,      # PPO-specific: clipping parameter
                ent_coef=0.01,       # Fixed entropy coefficient for PPO
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
        save_path = f"models/rl_v7_ppo/stage{stage_num}_{force_limit//1000}kN.zip"
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
    final_path = "models/rl_v7_ppo/final_v7_ppo.zip"
    model.save(final_path)

    print("="*70)
    print("  🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"\n   Total training time: {training_time}")
    print(f"   Final model: {final_path}")
    print(f"\n   This model uses:")
    print(f"   • PPO algorithm (on-policy)")
    print(f"   • ADAPTIVE reward scaling:")
    print(f"     - M4.5: 3× multiplier")
    print(f"     - M5.7: 7× multiplier")
    print(f"     - M7.4: 4× multiplier")
    print(f"     - M8.4: 3× multiplier")

    # Final comprehensive test
    print(f"\n{'='*70}")
    print(f"  FINAL TEST ON ALL HELD-OUT EARTHQUAKES")
    print(f"{'='*70}\n")

    results = []
    uncontrolled_baselines = {
        'M4.5': 21.02,
        'M5.7': 46.02,
        'M7.4': 235.55,
        'M8.4': 357.06
    }

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

        uncont = uncontrolled_baselines[magnitude]
        improvement = 100 * (uncont - peak_cm) / uncont

        results.append({
            'magnitude': magnitude,
            'peak_cm': peak_cm,
            'dcr': dcr,
            'improvement': improvement
        })

        status = "✓" if improvement > 0 else "✗"
        print(f"   {magnitude}:")
        print(f"      Peak disp: {peak_cm:.2f} cm")
        print(f"      Uncontrolled: {uncont:.2f} cm")
        print(f"      Improvement: {improvement:+.1f}% {status}")
        print(f"      DCR: {dcr:.2f}\n")

    print("="*70)
    print("  COMPARISON: v7-PPO vs v7-SAC vs v6 vs v5")
    print("="*70)
    v7_sac_results = {'M4.5': 20.72, 'M5.7': 46.45, 'M7.4': 219.30, 'M8.4': 363.36}
    v6_results = {'M4.5': 20.97, 'M5.7': 41.31, 'M7.4': 260.37, 'M8.4': None}
    v5_results = {'M4.5': 20.73, 'M5.7': 44.84, 'M7.4': 229.20, 'M8.4': 366.04}

    print()
    for r in results:
        mag = r['magnitude']
        v7_ppo_peak = r['peak_cm']
        v7_sac_peak = v7_sac_results[mag]
        v6_peak = v6_results[mag]
        v5_peak = v5_results[mag]

        print(f"   {mag}:")
        print(f"      v5 (5× uniform):     {v5_peak:.2f} cm")
        if v6_peak:
            print(f"      v6 (10× uniform):    {v6_peak:.2f} cm")
        print(f"      v7-SAC (adaptive):   {v7_sac_peak:.2f} cm")
        print(f"      v7-PPO (adaptive):   {v7_ppo_peak:.2f} cm  ← {r['improvement']:+.1f}%\n")

    print("="*70 + "\n")

    return model


if __name__ == "__main__":
    print("\n🚀 Starting v7 PPO Training with ADAPTIVE Rewards...\n")
    model = train_v7_ppo()
    if model is not None:
        print("\n✅ Training complete! PPO adaptive reward scaling model ready.")
    else:
        print("\n❌ Training failed - check training files.")
