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
print(f"Mass matrix diagonal:")
for i in range(13):
    print(f"  M[{i},{i}] = {M[i,i]:.0f} kg")

# Check if mass matrix is diagonal
is_diagonal = np.allclose(M, np.diag(np.diagonal(M)))
print(f"\nMass matrix is diagonal: {'✅ YES' if is_diagonal else '❌ NO'}")

# ===========================================================================
# ISSUE CHECK 3: Stiffness matrix symmetry and positive definiteness
# ===========================================================================
print("\n" + "="*70)
print("CHECK 3: STIFFNESS MATRIX PROPERTIES")
print("="*70)

K = env.K
print(f"\nStiffness matrix shape: {K.shape}")

is_symmetric = np.allclose(K, K.T)
print(f"Stiffness matrix is symmetric: {'✅ YES' if is_symmetric else '❌ NO'}")

# Check eigenvalues
eigenvalues = np.linalg.eigvals(K)
min_eigenvalue = np.min(eigenvalues.real)
print(f"Minimum eigenvalue: {min_eigenvalue:.3e}")
print(f"Positive semi-definite: {'✅ YES' if min_eigenvalue >= -1e-6 else '❌ NO'}")

# Check TMD coupling
print(f"\nTMD coupling in stiffness matrix:")
print(f"  K[11,11] (roof): {K[11,11]:.3e} N/m")
print(f"  K[11,12] (roof-TMD): {K[11,12]:.3e} N/m")
print(f"  K[12,11] (TMD-roof): {K[12,11]:.3e} N/m")
print(f"  K[12,12] (TMD): {K[12,12]:.3e} N/m")

expected_coupling = -env.tmd_k
actual_coupling = K[11,12]
print(f"\nExpected coupling: {expected_coupling:.0f} N/m")
print(f"Actual coupling: {actual_coupling:.0f} N/m")
print(f"Coupling correct: {'✅ YES' if abs(actual_coupling - expected_coupling) < 1 else '❌ NO'}")

# ===========================================================================
# ISSUE CHECK 4: Damping matrix properties
# ===========================================================================
print("\n" + "="*70)
print("CHECK 4: DAMPING MATRIX PROPERTIES")
print("="*70)

C = env.C
print(f"\nDamping matrix shape: {C.shape}")

is_symmetric = np.allclose(C, C.T)
print(f"Damping matrix is symmetric: {'✅ YES' if is_symmetric else '❌ NO'}")

# Check Rayleigh damping coefficients
print(f"\nRayleigh damping coefficients:")
print(f"  Target damping ratio: {env.damping_ratio*100:.1f}%")

# ===========================================================================
# ISSUE CHECK 5: Newmark integration stability
# ===========================================================================
print("\n" + "="*70)
print("CHECK 5: NEWMARK INTEGRATION PARAMETERS")
print("="*70)

print(f"\nNewmark parameters:")
print(f"  Beta: {env.beta}")
print(f"  Gamma: {env.gamma}")

# Check stability (unconditionally stable if beta >= 0.25, gamma >= 0.5)
is_stable = env.beta >= 0.25 and env.gamma >= 0.5
print(f"\nUnconditionally stable: {'✅ YES' if is_stable else '❌ NO'}")

# ===========================================================================
# ISSUE CHECK 6: Reward function sanity
# ===========================================================================
print("\n" + "="*70)
print("CHECK 6: REWARD FUNCTION ANALYSIS")
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
for key, value in avg_breakdown.items():
    print(f"  {key:15s}: {value:10.2f}")

total = sum(avg_breakdown.values())
print(f"  {'TOTAL':15s}: {total:10.2f}")

# Check if any component dominates
max_component = max(avg_breakdown.items(), key=lambda x: abs(x[1]))
print(f"\nDominating component: {max_component[0]} ({abs(max_component[1])/abs(total)*100:.1f}% of total)")

# ===========================================================================
# ISSUE CHECK 7: Action space and clipping
# ===========================================================================
print("\n" + "="*70)
print("CHECK 7: ACTION SPACE")
print("="*70)

print(f"\nAction space: {env.action_space}")
print(f"  Low: {env.action_space.low[0]}")
print(f"  High: {env.action_space.high[0]}")
print(f"  Max force: {env.max_force} N")

# Test if action scaling is correct
test_actions = [-1.0, -0.5, 0.0, 0.5, 1.0]
print(f"\nAction to force mapping:")
for action in test_actions:
    force = action * env.max_force
    print(f"  Action={action:+.1f} → Force={force/1000:+.1f} kN")

# ===========================================================================
# ISSUE CHECK 8: Control force application (Newton's 3rd law)
# ===========================================================================
print("\n" + "="*70)
print("CHECK 8: CONTROL FORCE APPLICATION")
print("="*70)

# Check if force is applied correctly
env = make_improved_tmd_env(test_files[0], max_force=50000)
obs, _ = env.reset()

# Apply a force and check the effect
action = np.array([1.0])  # Max force
obs, reward, done, truncated, info = env.step(action)

print(f"\nApplied action: {action[0]}")
print(f"Control force: {info['control_force']} N")
print(f"Expected force: {env.max_force} N")
print(f"Force correct: {'✅ YES' if abs(info['control_force'] - env.max_force) < 1 else '❌ NO'}")

# ===========================================================================
# ISSUE CHECK 9: Domain randomization implementation
# ===========================================================================
print("\n" + "="*70)
print("CHECK 9: DOMAIN RANDOMIZATION")
print("="*70)

env_clean = make_improved_tmd_env(test_files[0], max_force=50000)
env_noisy = make_improved_tmd_env(
    test_files[0],
    max_force=50000,
    sensor_noise_std=0.10,
    actuator_noise_std=0.05,
    latency_steps=2,
    dropout_prob=0.08
)

print(f"\nClean environment:")
print(f"  Sensor noise: {env_clean.sensor_noise_std}")
print(f"  Actuator noise: {env_clean.actuator_noise_std}")
print(f"  Latency: {env_clean.latency_steps} steps")
print(f"  Dropout: {env_clean.dropout_prob}")

print(f"\nNoisy environment:")
print(f"  Sensor noise: {env_noisy.sensor_noise_std}")
print(f"  Actuator noise: {env_noisy.actuator_noise_std}")
print(f"  Latency: {env_noisy.latency_steps} steps")
print(f"  Dropout: {env_noisy.dropout_prob}")

# ===========================================================================
# ISSUE CHECK 10: Model loading and architecture
# ===========================================================================
print("\n" + "="*70)
print("CHECK 10: MODEL ARCHITECTURE")
print("="*70)

model = SAC.load("rl_cl_robust_models/stage1_50kN_final_robust.zip")

print(f"\nModel hyperparameters:")
print(f"  Learning rate: {model.learning_rate}")
print(f"  Buffer size: {model.buffer_size}")
print(f"  Batch size: {model.batch_size}")
print(f"  Tau: {model.tau}")
print(f"  Gamma: {model.gamma}")

# Check if observation space matches
model_obs_space = model.observation_space
env_obs_space = env.observation_space

print(f"\nObservation space match:")
print(f"  Model obs space: {model_obs_space}")
print(f"  Env obs space: {env_obs_space}")
print(f"  Match: {'✅ YES' if model_obs_space == env_obs_space else '❌ NO'}")

# ===========================================================================
# ISSUE CHECK 11: Episode termination logic
# ===========================================================================
print("\n" + "="*70)
print("CHECK 11: EPISODE TERMINATION")
print("="*70)

env = make_improved_tmd_env(test_files[0], max_force=50000)
print(f"\nMax steps: {env.max_steps}")
print(f"Expected duration: {env.max_steps * env.dt:.2f} s")

# Check if episode terminates correctly
obs, _ = env.reset()
done = False
steps = 0

while not done and steps < 2000:
    action = np.array([0.0])
    obs, reward, done, truncated, info = env.step(action)
    done = done or truncated
    steps += 1

print(f"Episode terminated at step: {steps}")
print(f"Expected termination: {env.max_steps}")
print(f"Termination correct: {'✅ YES' if steps == env.max_steps else '❌ NO'}")

print("\n" + "="*70)
print("COMPREHENSIVE INVESTIGATION COMPLETE")
print("="*70)
