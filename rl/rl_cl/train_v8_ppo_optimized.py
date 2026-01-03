"""
v8 Training - OPTIMIZED PPO STRATEGY
====================================

PPO-SPECIFIC OPTIMIZATIONS for earthquake control!

Why a new strategy for PPO?
----------------------------
1. PPO is ON-POLICY → needs 2-3× more samples than SAC
2. Short earthquake episodes (20-120s) → need parallel envs
3. High variance across magnitudes → adaptive hyperparameters
4. Non-stationary problem → curriculum-aware learning rates

Key Improvements over v7-PPO (SAC-copied approach):
---------------------------------------------------
✓ 2× more timesteps (1.4M vs 700K)
✓ 4 parallel environments (4× faster data collection)
✓ Stage-adaptive n_steps (1024 → 4096)
✓ Curriculum learning rate decay (3e-4 → 5e-5)
✓ Entropy coefficient annealing (0.02 → 0.001)
✓ Larger batch size for stability (256 vs 64)
✓ Value function clipping (prevents divergence)
✓ Extended training on harder stages

Expected Results:
----------------
- Better sample efficiency than v7-PPO
- More stable learning on M7.4 and M8.4
- Target: Beat v7-SAC (219.30 cm on M7.4)

 Wrapper Features (if needed):
-----------------------------
# Instead of:
env = make_improved_tmd_env(eq_file, max_force=force_limit)

# Use:
from tmd_environment_ppo_wrapper import make_ppo_friendly_env
env = make_ppo_friendly_env(
    eq_file, 
    max_force=force_limit,
    normalize_obs=True,      # Normalize observations
    clip_reward=10.0         # Clip extreme rewards
)

Usage: python train_v8_ppo_optimized.py
"""

import numpy as np
import os
import glob
import random
from datetime import datetime
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from tmd_environment_adaptive_reward import make_improved_tmd_env


def make_env_factory(train_files, force_limit):
    """Factory function to create environment makers"""
    def _make_env():
        eq_file = random.choice(train_files)
        env = make_improved_tmd_env(eq_file, max_force=force_limit)
        return Monitor(env)
    return _make_env


def train_v8_ppo_optimized():
    """Train with PPO-optimized strategy"""

    # Auto-detect CUDA availability
    gpu_name = None
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        print("="*70)
        print("  v8 TRAINING - OPTIMIZED PPO STRATEGY (GPU ACCELERATED)")
        print("="*70)
        print(f"\n🚀 GPU Detected: {gpu_name}")
        print(f"   Using device: {device}\n")
    else:
        device = 'cpu'
        print("="*70)
        print("  v8 TRAINING - OPTIMIZED PPO STRATEGY")
        print("="*70)
        print("\n⚠️  No GPU detected - using CPU")
        print("   Training will be slower. Consider using a GPU for faster training.\n")

    print("🎯 PPO-Specific Optimizations:")
    print("   • 2× more timesteps (PPO needs more data)")
    print("   • 4 parallel environments (faster collection)")
    print("   • Adaptive n_steps per stage (match episode length)")
    print("   • Curriculum learning rate (3e-4 → 5e-5)")
    print("   • Entropy annealing (exploration → exploitation)")
    print("   • Larger batches (256) for stability")
    print("   • Value function clipping (prevent divergence)")
    if device == 'cuda':
        print("   • GPU acceleration (CUDA)\n")
    else:
        print()

    print("📊 Dataset Strategy:")
    print("   TRAINING: Improved synthetic earthquakes (training_set_v2/)")
    print("   → 10 variants per magnitude")
    print("   TESTING:  Original PEER earthquakes (PEER_*.csv)")
    print("   → Ensures model doesn't memorize test set!\n")

    print("🎯 Adaptive Reward Scaling (from v7):")
    print("   M4.5 (PGA 0.25g): 3× multiplier (gentle)")
    print("   M5.7 (PGA 0.35g): 7× multiplier (strong)")
    print("   M7.4 (PGA 0.75g): 4× multiplier (balanced)")
    print("   M8.4 (PGA 0.90g): 3× multiplier (conservative)")
    print("   → Auto-detected based on earthquake PGA\n")

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

    # PPO-OPTIMIZED Curriculum stages
    print("\n🎯 PPO-Optimized Curriculum Plan:")
    stages = [
        {
            'force_limit': 50000,
            'timesteps': 300000,      # 2× more (was 150K)
            'name': 'M4.5 @ 50kN',
            'magnitude': 'M4.5',
            'n_steps': 1024,          # Short episodes
            'learning_rate': 3e-4,    # Standard
            'ent_coef': 0.02          # High exploration
        },
        {
            'force_limit': 100000,
            'timesteps': 300000,      # 2× more (was 150K)
            'name': 'M5.7 @ 100kN',
            'magnitude': 'M5.7',
            'n_steps': 2048,          # Medium episodes
            'learning_rate': 3e-4,    # Standard
            'ent_coef': 0.01          # Medium exploration
        },
        {
            'force_limit': 150000,
            'timesteps': 400000,      # 2× more (was 200K)
            'name': 'M7.4 @ 150kN',
            'magnitude': 'M7.4',
            'n_steps': 4096,          # Long episodes
            'learning_rate': 1e-4,    # Lower (prevent instability)
            'ent_coef': 0.005         # Low exploration
        },
        {
            'force_limit': 150000,
            'timesteps': 400000,      # 2× more (was 200K)
            'name': 'M8.4 @ 150kN',
            'magnitude': 'M8.4',
            'n_steps': 4096,          # Very long episodes
            'learning_rate': 5e-5,    # Very low (careful)
            'ent_coef': 0.001         # Minimal exploration
        },
    ]

    n_envs = 4  # Parallel environments

    for i, stage in enumerate(stages, 1):
        n_variants = len(train_files[stage['magnitude']])
        effective_steps = stage['timesteps'] * n_envs
        print(f"   Stage {i}: {stage['name']}")
        print(f"      Timesteps: {stage['timesteps']:,} × {n_envs} envs = {effective_steps:,} total")
        print(f"      n_steps: {stage['n_steps']}, lr: {stage['learning_rate']:.0e}, ent: {stage['ent_coef']}")
        print(f"      Training variants: {n_variants}\n")

    total_timesteps = sum(s['timesteps'] for s in stages)
    print(f"   Total timesteps: {total_timesteps:,} (2× more than v7)")
    print(f"   Parallel envs: {n_envs}")
    print(f"   Effective samples: {total_timesteps * n_envs:,}\n")

    # Create directory
    os.makedirs("models/rl_v8_ppo_optimized", exist_ok=True)

    # Training
    start_time = datetime.now()
    model = None

    for stage_idx, stage in enumerate(stages):
        stage_num = stage_idx + 1
        force_limit = stage['force_limit']
        timesteps = stage['timesteps']
        magnitude = stage['magnitude']
        available_files = train_files[magnitude]
        n_steps = stage['n_steps']
        learning_rate = stage['learning_rate']
        ent_coef = stage['ent_coef']

        print(f"\n{'='*70}")
        print(f"  STAGE {stage_num}/{len(stages)}: {stage['name']}")
        print(f"{'='*70}\n")
        print(f"   Training variants: {len(available_files)}")
        print(f"   Hyperparameters:")
        print(f"      n_steps: {n_steps}")
        print(f"      learning_rate: {learning_rate:.0e}")
        print(f"      ent_coef: {ent_coef}")
        print(f"      batch_size: 256")
        print(f"      n_epochs: 10\n")

        # Create parallel environments
        env_fns = [make_env_factory(available_files, force_limit) for _ in range(n_envs)]

        # Use SubprocVecEnv for true parallelism (faster)
        # Fall back to DummyVecEnv if multiprocessing issues
        try:
            env = SubprocVecEnv(env_fns)
            print(f"   ✓ Using {n_envs} parallel environments (SubprocVecEnv)\n")
        except:
            env = DummyVecEnv(env_fns)
            print(f"   ⚠ Using {n_envs} sequential environments (DummyVecEnv - slower)\n")

        # Create or update model
        if model is None:
            print(f"🤖 Creating PPO model (OPTIMIZED)...")
            print(f"   Device: {device}\n")
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                n_steps=n_steps,           # Adaptive per stage
                batch_size=256,            # Large batches (stability)
                n_epochs=10,               # PPO epochs
                gamma=0.99,
                gae_lambda=0.95,           # GAE parameter
                clip_range=0.2,            # PPO clipping
                clip_range_vf=0.2,         # NEW: Value function clipping
                ent_coef=ent_coef,         # Adaptive per stage
                vf_coef=0.5,               # Value function coefficient
                max_grad_norm=0.5,         # Gradient clipping
                policy_kwargs=dict(
                    net_arch=[256, 256],
                    activation_fn=torch.nn.ReLU
                ),
                verbose=1,
                device=device              # Auto-detected (cuda or cpu)
            )
        else:
            print(f"🔄 Continuing from Stage {stage_num-1}...")
            print(f"   Updating hyperparameters for Stage {stage_num}...\n")

            # Update environment
            env_old = model.get_env()
            model.set_env(env)
            env_old.close()

            # Update learning rate and entropy coefficient
            model.learning_rate = learning_rate
            model.ent_coef = ent_coef

            # Recreate optimizer with new learning rate
            model.policy.optimizer = model.policy.optimizer_class(
                model.policy.parameters(),
                lr=learning_rate,
                **model.policy.optimizer_kwargs
            )

        # Train
        print(f"🚀 Training Stage {stage_num}...")
        model.learn(
            total_timesteps=timesteps,
            reset_num_timesteps=False,
            progress_bar=True
        )

        # Save
        save_path = f"models/rl_v8_ppo_optimized/stage{stage_num}_{force_limit//1000}kN.zip"
        model.save(save_path)
        print(f"\n💾 Saved: {save_path}")

        # Test on HELD-OUT test set (single env for deterministic testing)
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

        # Close parallel environments
        env.close()

    # Final model save
    training_time = datetime.now() - start_time
    final_path = "models/rl_v8_ppo_optimized/final_v8_ppo_optimized.zip"
    model.save(final_path)

    print("="*70)
    print("  🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"\n   Total training time: {training_time}")
    print(f"   Training device: {device.upper()}")
    if device == 'cuda':
        print(f"   GPU: {gpu_name}")
    print(f"   Final model: {final_path}")
    print(f"\n   This model uses:")
    print(f"   • PPO with optimized hyperparameters")
    print(f"   • {n_envs} parallel environments")
    print(f"   • Adaptive n_steps (1024 → 4096)")
    print(f"   • Curriculum learning rate (3e-4 → 5e-5)")
    print(f"   • Entropy annealing (0.02 → 0.001)")
    print(f"   • Device: {device.upper()}")
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
    print("  COMPARISON: v8-PPO-OPT vs v7-SAC vs v7-PPO vs v6 vs v5")
    print("="*70)
    v7_sac_results = {'M4.5': 20.72, 'M5.7': 46.45, 'M7.4': 219.30, 'M8.4': 363.36}
    v6_results = {'M4.5': 20.97, 'M5.7': 41.31, 'M7.4': 260.37, 'M8.4': None}
    v5_results = {'M4.5': 20.73, 'M5.7': 44.84, 'M7.4': 229.20, 'M8.4': 366.04}

    print()
    for r in results:
        mag = r['magnitude']
        v8_ppo_peak = r['peak_cm']
        v7_sac_peak = v7_sac_results[mag]
        v6_peak = v6_results[mag]
        v5_peak = v5_results[mag]

        print(f"   {mag}:")
        print(f"      v5 (5× uniform):        {v5_peak:.2f} cm")
        if v6_peak:
            print(f"      v6 (10× uniform):       {v6_peak:.2f} cm")
        print(f"      v7-SAC (adaptive):      {v7_sac_peak:.2f} cm  ← BEST SAC")
        print(f"      v8-PPO-OPT (adaptive):  {v8_ppo_peak:.2f} cm  ← {r['improvement']:+.1f}%")

        # Highlight if v8 beats v7-SAC
        if v8_ppo_peak < v7_sac_peak:
            delta = v7_sac_peak - v8_ppo_peak
            print(f"         🏆 Beats v7-SAC by {delta:.2f} cm!")
        print()

    print("="*70 + "\n")

    return model


if __name__ == "__main__":
    # Need torch for activation function
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch not found. Install with: pip install torch")
        exit(1)

    print("\n🚀 Starting v8 PPO OPTIMIZED Training...\n")
    model = train_v8_ppo_optimized()
    if model is not None:
        print("\n✅ Training complete! Optimized PPO model ready.")
        print("\nKey improvements over v7-PPO:")
        print("  • 2× more timesteps (better for on-policy PPO)")
        print("  • 4 parallel environments (4× faster)")
        print("  • Adaptive hyperparameters per stage")
        print("  • Curriculum learning rate decay")
        print("  • Entropy annealing for better exploration→exploitation")
    else:
        print("\n❌ Training failed - check training files.")
