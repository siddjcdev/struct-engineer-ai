"""
COMPREHENSIVE INVESTIGATION - Check EVERYTHING
"""
import sys
sys.path.insert(0, '/Users/Shared/dev/git/struct-engineer-ai')
import numpy as np
from rl.rl_cl.tmd_environment import make_improved_tmd_env
from stable_baselines3 import SAC

print("="*70)
print("COMPREHENSIVE INVESTIGATION - LOOKING FOR ALL ISSUES")
print("="*70)

# ===========================================================================
# ISSUE CHECK 1: Verify earthquake data integrity
# ===========================================================================
print("\n" + "="*70)
print("CHECK 1: EARTHQUAKE DATA INTEGRITY")
print("="*70)

test_files = [
    "../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv",
    "../../matlab/datasets/PEER_moderate_M5.7_PGA0.35g.csv",
    "../../matlab/datasets/PEER_high_M7.4_PGA0.75g.csv",
]

for eq_file in test_files:
    data = np.loadtxt(eq_file, delimiter=',', skiprows=1)
    t = data[:, 0]
    accel = data[:, 1]
    
    dt = np.diff(t)
    dt_consistent = np.allclose(dt, dt[0], rtol=1e-6)
    final_accel = accel[-1]
    
    print(f"\n{eq_file.split('/')[-1]}:")
    print(f"  Length: {len(t)} samples")
    print(f"  Duration: {t[-1]:.2f} s")
    print(f"  Timestep: {dt[0]:.4f} s ({'✅ Consistent' if dt_consistent else '❌ Inconsistent'})")
    print(f"  Final accel: {final_accel:.10f} m/s² ({'✅ Zero' if abs(final_accel) < 1e-6 else '❌ NON-ZERO'})")
    print(f"  Peak accel: {np.max(np.abs(accel)):.3f} m/s²")

# ===========================================================================
# ISSUE CHECK 2: Mass matrix consistency
# ===========================================================================
print("\n" + "="*70)
print("CHECK 2: MASS MATRIX CONSISTENCY")
print("="*70)

env = make_improved_tmd_env(test_files[0], max_force=50000)

M = env.M
print(f"\nMass matrix shape: {M.shape}")
print(f"Mass matrix diagonal (first 5 and last 2):")
for i in [0, 1, 2, 3, 4, 11, 12]:
    print(f"  M[{i:2d},{i:2d}] = {M[i,i]:10.0f} kg")

# Check if mass matrix is diagonal
is_diagonal = np.allclose(M, np.diag(np.diagonal(M)))
print(f"\nMass matrix is diagonal: {'✅ YES' if is_diagonal else '❌ NO'}")

# ===========================================================================
# ISSUE CHECK 3: Stiffness matrix symmetry
# ===========================================================================
print("\n" + "="*70)
print("CHECK 3: STIFFNESS MATRIX PROPERTIES")
print("="*70)

K = env.K

is_symmetric = np.allclose(K, K.T)
print(f"Stiffness matrix is symmetric: {'✅ YES' if is_symmetric else '❌ NO'}")

# Check TMD coupling
print(f"\nTMD coupling in stiffness matrix:")
print(f"  K[11,11] (roof diagonal): {K[11,11]:.3e} N/m")
print(f"  K[11,12] (roof-TMD coupling): {K[11,12]:.3e} N/m")
print(f"  K[12,11] (TMD-roof coupling): {K[12,11]:.3e} N/m")
print(f"  K[12,12] (TMD diagonal): {K[12,12]:.3e} N/m")

expected_coupling = -env.tmd_k
actual_coupling = K[11,12]
print(f"\nExpected coupling: {expected_coupling:.0f} N/m")
print(f"Actual coupling: {actual_coupling:.0f} N/m")
print(f"Coupling correct: {'✅ YES' if abs(actual_coupling - expected_coupling) < 1 else '❌ NO'}")

# ===========================================================================
# ISSUE CHECK 4: Reward function components
# ===========================================================================
print("\n" + "="*70)
print("CHECK 4: REWARD FUNCTION ANALYSIS")
print("="*70)

env = make_improved_tmd_env(test_files[0], max_force=50000)
obs, _ = env.reset()

# Run a few steps and check reward components
rewards = []
done = False
step = 0

while not done and step < 20:
    action = np.array([0.0])  # No control
    obs, reward, done, truncated, info = env.step(action)
    rewards.append(info['reward_breakdown'])
    done = done or truncated
    step += 1

# Average reward components
avg_breakdown = {
    key: np.mean([r[key] for r in rewards])
    for key in rewards[0].keys()
}

print(f"\nAverage reward components (first 20 steps, no control):")
for key, value in sorted(avg_breakdown.items(), key=lambda x: abs(x[1]), reverse=True):
    pct = abs(value) / sum(abs(v) for v in avg_breakdown.values()) * 100
    print(f"  {key:15s}: {value:10.2f} ({pct:5.1f}%)")

# ===========================================================================
# ISSUE CHECK 5: Action to force mapping
# ===========================================================================
print("\n" + "="*70)
print("CHECK 5: ACTION SPACE AND FORCE MAPPING")
print("="*70)

print(f"\nAction space: {env.action_space}")
print(f"Max force: {env.max_force} N ({env.max_force/1000:.0f} kN)")

test_actions = [-1.0, -0.5, 0.0, 0.5, 1.0]
print(f"\nAction to force mapping:")
for action in test_actions:
    force = action * env.max_force
    print(f"  Action={action:+.1f} → Force={force/1000:+6.1f} kN")

# ===========================================================================
# ISSUE CHECK 6: Observation space bounds vs actual values
# ===========================================================================
print("\n" + "="*70)
print("CHECK 6: OBSERVATION BOUNDS (PASSIVE TMD)")
print("="*70)

env = make_improved_tmd_env(test_files[1], max_force=100000)  # M5.7
obs, _ = env.reset()

obs_min = obs.copy()
obs_max = obs.copy()
done = False

print("\nRunning passive TMD on M5.7 (40s)...")
while not done:
    action = np.array([0.0])
    obs, reward, done, truncated, info = env.step(action)
    obs_min = np.minimum(obs_min, obs)
    obs_max = np.maximum(obs_max, obs)
    done = done or truncated

obs_labels = ['roof_disp', 'roof_vel', 'floor8_disp', 'floor8_vel',
              'floor6_disp', 'floor6_vel', 'tmd_disp', 'tmd_vel']

print(f"\nPassive TMD observation ranges:")
for i, label in enumerate(obs_labels):
    low_bound = env.observation_space.low[i]
    high_bound = env.observation_space.high[i]
    exceeds = obs_min[i] < low_bound or obs_max[i] > high_bound
    
    if exceeds:
        exceed_factor = max(abs(obs_min[i]/low_bound), abs(obs_max[i]/high_bound))
        status = f"❌ EXCEEDS by {exceed_factor:.1f}x"
    else:
        status = "✅ OK"
    
    print(f"  {label:12s}: [{obs_min[i]:+7.3f}, {obs_max[i]:+7.3f}]  "
          f"(bounds: [{low_bound:+4.1f}, {high_bound:+4.1f}])  {status}")

# ===========================================================================
# ISSUE CHECK 7: Check for NaN or Inf values
# ===========================================================================
print("\n" + "="*70)
print("CHECK 7: NUMERICAL STABILITY")
print("="*70)

env = make_improved_tmd_env(test_files[2], max_force=150000)  # M7.4
obs, _ = env.reset()

nan_count = 0
inf_count = 0
done = False
step = 0

while not done and step < 100:
    action = np.array([0.5])  # Some control
    obs, reward, done, truncated, info = env.step(action)
    
    if np.any(np.isnan(obs)):
        nan_count += 1
    if np.any(np.isinf(obs)):
        inf_count += 1
    
    done = done or truncated
    step += 1

print(f"\nNumerical stability check (first 100 steps, M7.4):")
print(f"  NaN occurrences: {nan_count} ({'❌ PROBLEM' if nan_count > 0 else '✅ OK'})")
print(f"  Inf occurrences: {inf_count} ({'❌ PROBLEM' if inf_count > 0 else '✅ OK'})")

# ===========================================================================
# ISSUE CHECK 8: Training vs inference observation mismatch
# ===========================================================================
print("\n" + "="*70)
print("CHECK 8: TRAINING VS INFERENCE")
print("="*70)

# Load model
model = SAC.load("rl_cl_robust_models/stage1_50kN_final_robust.zip")

# Create environment
env = make_improved_tmd_env(test_files[0], max_force=50000)

print(f"\nModel observation space:")
print(f"  Shape: {model.observation_space.shape}")
print(f"  Low: {model.observation_space.low}")
print(f"  High: {model.observation_space.high}")

print(f"\nEnvironment observation space:")
print(f"  Shape: {env.observation_space.shape}")
print(f"  Low: {env.observation_space.low}")
print(f"  High: {env.observation_space.high}")

spaces_match = (model.observation_space.shape == env.observation_space.shape and
                np.allclose(model.observation_space.low, env.observation_space.low) and
                np.allclose(model.observation_space.high, env.observation_space.high))

print(f"\nObservation spaces match: {'✅ YES' if spaces_match else '❌ NO'}")

print("\n" + "="*70)
print("COMPREHENSIVE INVESTIGATION COMPLETE")
print("="*70)
