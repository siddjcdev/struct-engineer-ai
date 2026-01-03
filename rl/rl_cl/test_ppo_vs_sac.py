"""
Quick Test: PPO vs SAC on M4.5
================================

Compare PPO and SAC on a single earthquake to see which works better.

Test configuration:
- Earthquake: M4.5 (quickest to train)
- Timesteps: 200K (enough for comparison)
- Same adaptive reward (3× for M4.5)
- Same training data

Expected runtime: ~15-20 minutes
"""

import numpy as np
from datetime import datetime
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from tmd_environment_adaptive_reward import make_improved_tmd_env
import glob
import random


def test_algorithm(algorithm_name, algorithm_class, timesteps=200000):
    """Test an algorithm on M4.5"""

    print(f"\n{'='*70}")
    print(f"  TESTING {algorithm_name}")
    print(f"{'='*70}\n")

    # Training files
    train_dir = "../../matlab/datasets/training_set_v2"
    train_files = sorted(glob.glob(f"{train_dir}/TRAIN_M4.5_*.csv"))

    # Test file
    test_file = "../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv"

    print(f"Training variants: {len(train_files)}")
    print(f"Test file: {test_file.split('/')[-1]}")
    print(f"Timesteps: {timesteps:,}\n")

    # Create environment
    def make_env():
        eq_file = random.choice(train_files)
        env = make_improved_tmd_env(eq_file, max_force=50000)
        return Monitor(env)

    env = DummyVecEnv([make_env])

    # Create model with algorithm-specific hyperparameters
    start_time = datetime.now()

    if algorithm_name == "SAC":
        model = algorithm_class(
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
    else:  # PPO
        model = algorithm_class(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,  # Larger for PPO
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1,
            device='cpu'
        )

    print(f"🚀 Training {algorithm_name}...")
    model.learn(total_timesteps=timesteps, progress_bar=True)

    training_time = datetime.now() - start_time

    # Test on held-out earthquake
    print(f"\n📊 Testing on held-out M4.5 earthquake...")
    test_env = make_improved_tmd_env(test_file, max_force=50000)
    obs, _ = test_env.reset()
    done = False
    peak = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        peak = max(peak, abs(info['roof_displacement']))
        done = done or truncated

    peak_cm = peak * 100
    uncont = 21.02  # M4.5 uncontrolled baseline
    improvement = 100 * (uncont - peak_cm) / uncont

    print(f"\n{'='*70}")
    print(f"  {algorithm_name} RESULTS")
    print(f"{'='*70}")
    print(f"  Training time: {training_time}")
    print(f"  Peak displacement: {peak_cm:.2f} cm")
    print(f"  Uncontrolled: {uncont:.2f} cm")
    print(f"  Improvement: {improvement:+.1f}%")
    print(f"{'='*70}\n")

    return {
        'algorithm': algorithm_name,
        'peak_cm': peak_cm,
        'improvement': improvement,
        'training_time': training_time
    }


def main():
    print("="*70)
    print("  PPO vs SAC COMPARISON TEST")
    print("="*70)
    print("\nQuick test on M4.5 to compare algorithms")
    print("This will take ~15-20 minutes total\n")

    # Test both algorithms
    results = []

    # Test SAC first (our current baseline)
    sac_result = test_algorithm("SAC", SAC, timesteps=200000)
    results.append(sac_result)

    # Test PPO
    ppo_result = test_algorithm("PPO", PPO, timesteps=200000)
    results.append(ppo_result)

    # Comparison
    print("="*70)
    print("  FINAL COMPARISON")
    print("="*70)
    print(f"\n{'Algorithm':<10} {'Peak Disp':<15} {'Improvement':<15} {'Time'}")
    print("-"*70)
    for r in results:
        print(f"{r['algorithm']:<10} {r['peak_cm']:>6.2f} cm      {r['improvement']:>+6.1f}%        {r['training_time']}")

    print("\n" + "="*70)

    # Winner
    winner = min(results, key=lambda x: x['peak_cm'])
    print(f"\n🏆 WINNER: {winner['algorithm']} with {winner['peak_cm']:.2f} cm")
    print(f"   ({winner['improvement']:+.1f}% vs uncontrolled)\n")

    # Recommendation
    if winner['algorithm'] == 'PPO':
        print("✅ RECOMMENDATION: Use PPO for full training")
        print("   - More stable")
        print("   - Better peak displacement on M4.5")
        print("   - May need 2× timesteps (1.4M total) for full curriculum")
    else:
        print("✅ RECOMMENDATION: Stick with SAC")
        print("   - Better sample efficiency")
        print("   - Already achieving good results with v7 adaptive")
        print("   - PPO didn't show significant improvement")

    print("="*70 + "\n")


if __name__ == "__main__":
    main()
