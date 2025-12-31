"""
Train with Shaped Rewards
=========================

Uses the shaped reward environment to provide stronger learning signals:
- 10x stronger displacement penalty
- 10x stronger velocity penalty
- Force direction bonus (+5.0 for correct, -2.0 for wrong)
- No smoothness/acceleration penalties

This should allow the agent to discover that opposing velocity reduces displacement.

Usage: python train_shaped_rewards.py
"""

import numpy as np
import os
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from tmd_environment_shaped_reward import make_improved_tmd_env


def train_shaped_rewards():
    """Train with shaped rewards on M4.5 earthquake"""

    print("="*70)
    print("  Training with Shaped Rewards")
    print("="*70)

    # Start with M4.5 only
    earthquake_file = "../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv"
    force_limit = 50000  # 50kN

    print(f"\n📊 Training Configuration:")
    print(f"   Earthquake: {os.path.basename(earthquake_file)}")
    print(f"   Force limit: {force_limit/1000:.0f} kN")
    print(f"   Timesteps: 200,000")
    print(f"\n🎯 Shaped Reward Features:")
    print(f"   • Displacement penalty: -10.0 (was -1.0)")
    print(f"   • Velocity penalty: -3.0 (was -0.3)")
    print(f"   • Force direction bonus: +5.0 for correct, -2.0 for wrong")
    print(f"   • No smoothness/acceleration penalties")

    # Create directory
    os.makedirs("models/rl_shaped_rewards", exist_ok=True)

    # Create environment
    def make_env():
        env = make_improved_tmd_env(earthquake_file, max_force=force_limit)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])

    # Create model
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

    # Train
    print(f"\n🚀 Training for 200,000 timesteps...")
    start_time = datetime.now()

    model.learn(
        total_timesteps=200_000,
        progress_bar=True
    )

    training_time = datetime.now() - start_time

    # Save
    save_path = "models/rl_shaped_rewards/m4.5_shaped.zip"
    model.save(save_path)
    print(f"\n💾 Saved: {save_path}")

    # Test
    print(f"\n📊 Testing on M4.5...")
    test_env = make_improved_tmd_env(earthquake_file, max_force=force_limit)
    obs, _ = test_env.reset()
    done = False
    peak = 0
    total_reward = 0
    force_history = []
    vel_history = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        peak = max(peak, abs(info['roof_displacement']))
        total_reward += reward
        force_history.append(info['control_force'])
        vel_history.append(obs[1])  # roof velocity
        done = done or truncated

    peak_cm = peak * 100

    # Get DCR from environment
    episode_metrics = test_env.get_episode_metrics()
    dcr = episode_metrics.get('dcr', 0.0)

    # Analyze force behavior
    force_history = np.array(force_history)
    vel_history = np.array(vel_history)

    # Check if agent learned to oppose velocity
    correct_direction = 0
    total_with_motion = 0
    for i in range(len(vel_history)):
        if abs(vel_history[i]) > 0.01:  # Significant motion
            total_with_motion += 1
            if (vel_history[i] > 0 and force_history[i] < 0) or \
               (vel_history[i] < 0 and force_history[i] > 0):
                correct_direction += 1

    if total_with_motion > 0:
        correct_pct = 100 * correct_direction / total_with_motion
    else:
        correct_pct = 0

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"\n   Training time: {training_time}")
    print(f"   Peak displacement: {peak_cm:.2f} cm")
    print(f"   DCR (Drift Concentration Ratio): {dcr:.2f}")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Force direction correctness: {correct_pct:.1f}%")
    print(f"   Mean force magnitude: {np.mean(np.abs(force_history)):.0f} N")

    # Compare to uncontrolled
    uncontrolled_peak = 21.02  # From emergency_physics_check.py
    improvement = 100 * (uncontrolled_peak - peak_cm) / uncontrolled_peak

    print(f"\n   Uncontrolled peak: {uncontrolled_peak:.2f} cm")
    print(f"   Improvement: {improvement:.1f}%")

    # Analyze results
    print(f"\n{'='*70}")
    print(f"  ANALYSIS")
    print(f"{'='*70}")

    if peak_cm < 19.0:
        print(f"\n   ✅ SUCCESS! Agent learned to reduce displacement!")
    elif improvement > 5:
        print(f"\n   ⚠️  Partial success - some improvement but not optimal")
    else:
        print(f"\n   ❌ Agent did not learn effectively")

    if dcr <= 1.5:
        print(f"   ✅ DCR is good ({dcr:.2f} ≤ 1.5) - uniform drift distribution")
        print(f"   → Hypothesis CONFIRMED: Good control naturally produces good DCR")
    elif dcr <= 2.0:
        print(f"   ⚠️  DCR is acceptable ({dcr:.2f} ≤ 2.0)")
    else:
        print(f"   ❌ DCR is high ({dcr:.2f} > 2.0) - drift concentration detected")
        print(f"   → May need to reconsider DCR penalty removal")

    print("="*70 + "\n")

    return model


if __name__ == "__main__":
    print("\n🚀 Starting Shaped Reward Training...\n")
    model = train_shaped_rewards()
    if model is not None:
        print("\n✅ Training complete!")
    else:
        print("\n❌ Training failed.")
