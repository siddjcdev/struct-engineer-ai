"""
Quick test of the latest shaped reward model with CORRECT direction check
"""
import numpy as np
from tmd_environment_shaped_reward import make_improved_tmd_env

print("="*70)
print("  Testing Latest Shaped Reward Model")
print("="*70)

# Load model
from stable_baselines3 import SAC
model_path = "models/rl_shaped_rewards_4/m4.5_shaped.zip"
print(f"\nLoading model: {model_path}")
model = SAC.load(model_path)

# Test
earthquake_file = "../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv"
force_limit = 50000

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

# Get DCR
episode_metrics = test_env.get_episode_metrics()
dcr = episode_metrics.get('dcr', 0.0)

# Analyze force behavior with CORRECT logic
force_history = np.array(force_history)
vel_history = np.array(vel_history)

correct_direction = 0
total_with_motion = 0
for i in range(len(vel_history)):
    if abs(vel_history[i]) > 0.01:
        total_with_motion += 1
        # CORRECT: Same signs = correct (due to F_eq[roof] -= control_force)
        if (vel_history[i] > 0 and force_history[i] > 0) or \
           (vel_history[i] < 0 and force_history[i] < 0):
            correct_direction += 1

if total_with_motion > 0:
    correct_pct = 100 * correct_direction / total_with_motion
else:
    correct_pct = 0

print(f"\n{'='*70}")
print(f"  RESULTS (with CORRECT direction check)")
print(f"{'='*70}")
print(f"\n   Peak displacement: {peak_cm:.2f} cm")
print(f"   DCR: {dcr:.2f}")
print(f"   Total reward: {total_reward:.2f}")
print(f"   Force direction correctness: {correct_pct:.1f}%")
print(f"   Mean force magnitude: {np.mean(np.abs(force_history)):.0f} N")

uncontrolled_peak = 21.02
improvement = 100 * (uncontrolled_peak - peak_cm) / uncontrolled_peak

print(f"\n   Uncontrolled peak: {uncontrolled_peak:.2f} cm")
print(f"   Improvement: {improvement:.1f}%")

print(f"\n{'='*70}")
print(f"  ANALYSIS")
print(f"{'='*70}")

if correct_pct > 90:
    print(f"\n   ✅ Agent learned correct direction! ({correct_pct:.1f}% correctness)")
else:
    print(f"\n   ❌ Agent did not learn direction ({correct_pct:.1f}% correctness)")

if peak_cm < 19.0:
    print(f"   ✅ SUCCESS! Displacement reduced!")
elif improvement > 5:
    print(f"   ⚠️  Partial improvement")
else:
    print(f"   ❌ No improvement in displacement")

print("="*70 + "\n")
