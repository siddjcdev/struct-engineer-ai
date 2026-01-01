"""
Diagnose why Stage 1 model is underperforming
"""
import numpy as np
from stable_baselines3 import SAC
from tmd_environment import make_improved_tmd_env

print("\n" + "="*70)
print("DIAGNOSING STAGE 1 MODEL PERFORMANCE")
print("="*70)

test_file = "../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv"

# Test 1: Uncontrolled (baseline)
print("\n[TEST 1] Uncontrolled (no TMD, no control)")
print("-" * 70)
env = make_improved_tmd_env(test_file, max_force=1.0)
env.tmd_k = 0.0
env.tmd_c = 0.0
obs, _ = env.reset()
done = False
peak_uncontrolled = 0
while not done:
    action = np.array([0.0])
    obs, reward, done, truncated, info = env.step(action)
    peak_uncontrolled = max(peak_uncontrolled, abs(info['roof_displacement']))
    done = done or truncated
print(f"Peak displacement: {peak_uncontrolled*100:.2f} cm")

# Test 2: Random actions
print("\n[TEST 2] Random control (baseline for RL)")
print("-" * 70)
env = make_improved_tmd_env(test_file, max_force=50000.0)
obs, _ = env.reset()
done = False
peak_random = 0
np.random.seed(42)
while not done:
    action = np.random.uniform(-1.0, 1.0, size=1)
    obs, reward, done, truncated, info = env.step(action)
    peak_random = max(peak_random, abs(info['roof_displacement']))
    done = done or truncated
print(f"Peak displacement: {peak_random*100:.2f} cm")

# Test 3: Zero action (passive TMD)
print("\n[TEST 3] Passive TMD (no active control)")
print("-" * 70)
env = make_improved_tmd_env(test_file, max_force=50000.0)
obs, _ = env.reset()
done = False
peak_passive = 0
while not done:
    action = np.array([0.0])
    obs, reward, done, truncated, info = env.step(action)
    peak_passive = max(peak_passive, abs(info['roof_displacement']))
    done = done or truncated
print(f"Peak displacement: {peak_passive*100:.2f} cm")

# Test 4: Stage 1 trained model
print("\n[TEST 4] Stage 1 Trained Model")
print("-" * 70)
model_path = "rl_cl_models_alpha_1/stage1_50kN_final_robust.zip"
try:
    model = SAC.load(model_path)
    env = make_improved_tmd_env(test_file, max_force=50000.0)
    obs, _ = env.reset()
    done = False
    peak_rl = 0

    # Track observations and actions
    obs_history = []
    action_history = []
    force_history = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs_history.append(obs.copy())
        action_history.append(action[0])

        obs, reward, done, truncated, info = env.step(action)
        force_history.append(action[0] * 50000)
        peak_rl = max(peak_rl, abs(info['roof_displacement']))
        done = done or truncated

    print(f"Peak displacement: {peak_rl*100:.2f} cm")

    # Analyze control behavior
    print(f"\n📊 Control Analysis:")
    print(f"   Action range: [{min(action_history):.3f}, {max(action_history):.3f}]")
    print(f"   Force range: [{min(force_history)/1000:.1f}, {max(force_history)/1000:.1f}] kN")
    print(f"   Mean action: {np.mean(action_history):.3f}")
    print(f"   Action std: {np.std(action_history):.3f}")

    # Check if model is doing anything
    if np.std(action_history) < 0.01:
        print("   ⚠️  WARNING: Actions are nearly constant (model may not be learning)")

    # Check observation ranges
    obs_history = np.array(obs_history)
    print(f"\n📊 Observation Ranges:")
    obs_labels = ['roof_disp', 'roof_vel', 'floor8_disp', 'floor8_vel',
                  'floor6_disp', 'floor6_vel', 'tmd_disp', 'tmd_vel']
    for i, label in enumerate(obs_labels):
        print(f"   {label:12s}: [{obs_history[:, i].min():+.4f}, {obs_history[:, i].max():+.4f}]")

except FileNotFoundError:
    print(f"❌ Model not found: {model_path}")
    peak_rl = None

# Summary
print(f"\n" + "="*70)
print("PERFORMANCE COMPARISON")
print("="*70)
print(f"Uncontrolled (no TMD):     {peak_uncontrolled*100:6.2f} cm")
print(f"Random control:            {peak_random*100:6.2f} cm")
print(f"Passive TMD:               {peak_passive*100:6.2f} cm")
if peak_rl is not None:
    print(f"RL Stage 1 model:          {peak_rl*100:6.2f} cm")

    # Diagnosis
    print(f"\n📊 Diagnosis:")
    if peak_rl > peak_uncontrolled:
        print(f"   ❌ RL is WORSE than uncontrolled by {(peak_rl/peak_uncontrolled - 1)*100:.1f}%")
        print(f"   → Model learned a destructive policy!")
    elif peak_rl > peak_passive:
        print(f"   ⚠️  RL is worse than passive TMD")
        print(f"   → Model hasn't learned to use control effectively")
    elif peak_rl > peak_random:
        print(f"   ⚠️  RL is worse than random control")
        print(f"   → Model may be stuck in local minimum")
    else:
        improvement = (1 - peak_rl/peak_uncontrolled) * 100
        print(f"   ✅ RL is working! {improvement:.1f}% reduction from uncontrolled")

print("\n" + "="*70)
