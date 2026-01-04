"""
FINAL EXHAUSTIVE CHECK - Leave no stone unturned
"""
import sys
sys.path.insert(0, '/Users/Shared/dev/git/struct-engineer-ai')
import numpy as np
from rl.rl_cl.tmd_environment import make_improved_tmd_env
from stable_baselines3 import SAC

print("="*70)
print("FINAL EXHAUSTIVE CHECK")
print("="*70)

# ===========================================================================
# CHECK: Control force sign convention
# ===========================================================================
print("\n" + "="*70)
print("CHECK: CONTROL FORCE SIGN AND APPLICATION")
print("="*70)

test_file = "../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv"
env = make_improved_tmd_env(test_file, max_force=50000)
obs, _ = env.reset()

# Apply positive force for one step
action = np.array([1.0])
obs, reward, done, truncated, info = env.step(action)

print(f"\nAction: +1.0 (max positive)")
print(f"Control force: {info['control_force']} N")
print(f"Expected: +50000 N")

# Check the force application in the code
# Force should be: F_eq[11] -= control_force (roof gets -F)
#                  F_eq[12] += control_force (TMD gets +F)
# This is Newton's 3rd law: action-reaction pair

# Reset and check negative force
env.reset()
action = np.array([-1.0])
obs, reward, done, truncated, info = env.step(action)

print(f"\nAction: -1.0 (max negative)")
print(f"Control force: {info['control_force']} N")
print(f"Expected: -50000 N")

# ===========================================================================
# CHECK: Reward calculation matches code
# ===========================================================================
print("\n" + "="*70)
print("CHECK: REWARD CALCULATION VERIFICATION")
print("="*70)

env = make_improved_tmd_env(test_file, max_force=50000)
obs, _ = env.reset()

# Take one step
action = np.array([0.0])
obs, reward, done, truncated, info = env.step(action)

# Manually calculate what reward should be
roof_disp = info['roof_displacement']
roof_vel = info['roof_velocity']
roof_accel = info['roof_acceleration']
control_force = info['control_force']

displacement_penalty = -1.0 * abs(roof_disp)
velocity_penalty = -0.3 * abs(roof_vel)
force_penalty = 0.0  # Disabled
smoothness_penalty = -0.005 * (abs(control_force - 0) / 50000)
acceleration_penalty = -0.1 * abs(roof_accel)
dcr_penalty = info['reward_breakdown']['dcr']

manual_reward = (displacement_penalty + velocity_penalty + force_penalty + 
                smoothness_penalty + acceleration_penalty + dcr_penalty)

print(f"\nManual calculation:")
print(f"  Displacement: {displacement_penalty:.6f}")
print(f"  Velocity: {velocity_penalty:.6f}")
print(f"  Force: {force_penalty:.6f}")
print(f"  Smoothness: {smoothness_penalty:.6f}")
print(f"  Acceleration: {acceleration_penalty:.6f}")
print(f"  DCR: {dcr_penalty:.6f}")
print(f"  TOTAL (manual): {manual_reward:.6f}")
print(f"  TOTAL (actual): {reward:.6f}")
print(f"  Match: {'✅ YES' if abs(manual_reward - reward) < 0.001 else '❌ NO'}")

# ===========================================================================
# CHECK: Episode metrics calculation
# ===========================================================================
print("\n" + "="*70)
print("CHECK: EPISODE METRICS (DCR, etc)")
print("="*70)

env = make_improved_tmd_env(test_file, max_force=50000)
obs, _ = env.reset()

# Run full episode
done = False
while not done:
    action = np.array([0.0])
    obs, reward, done, truncated, info = env.step(action)
    done = done or truncated

# Get episode metrics
metrics = env.get_episode_metrics()

print(f"\nEpisode metrics:")
print(f"  Peak displacement: {metrics['peak_displacement']*100:.2f} cm")
print(f"  Peak drift: {metrics['peak_drift']*100:.4f} cm")
print(f"  DCR: {metrics['dcr']:.3f}")
print(f"  Max drift floor: {metrics['max_drift_floor']}")

# Verify DCR calculation
sorted_peaks = np.sort(env.peak_drift_per_floor)
percentile_75 = np.percentile(sorted_peaks, 75)
max_peak = np.max(env.peak_drift_per_floor)
manual_dcr = max_peak / percentile_75 if percentile_75 > 1e-10 else 0.0

print(f"\nDCR verification:")
print(f"  75th percentile: {percentile_75*100:.4f} cm")
print(f"  Max drift: {max_peak*100:.4f} cm")
print(f"  Manual DCR: {manual_dcr:.3f}")
print(f"  Reported DCR: {metrics['dcr']:.3f}")
print(f"  Match: {'✅ YES' if abs(manual_dcr - metrics['dcr']) < 0.01 else '❌ NO'}")

# ===========================================================================
# CHECK: TMD spring force calculation
# ===========================================================================
print("\n" + "="*70)
print("CHECK: TMD SPRING FORCE")
print("="*70)

env = make_improved_tmd_env(test_file, max_force=50000)
obs, _ = env.reset()

# Run a few steps
for _ in range(10):
    action = np.array([0.0])
    obs, reward, done, truncated, info = env.step(action)

# Check TMD spring force
tmd_disp = env.displacement[12]
roof_disp = env.displacement[11]
tmd_relative = tmd_disp - roof_disp

spring_force = env.tmd_k * tmd_relative
damping_force = env.tmd_c * (env.velocity[12] - env.velocity[11])

print(f"\nTMD state:")
print(f"  TMD absolute displacement: {tmd_disp*100:.4f} cm")
print(f"  Roof displacement: {roof_disp*100:.4f} cm")
print(f"  TMD relative displacement: {tmd_relative*100:.4f} cm")
print(f"  TMD spring force: {spring_force:.2f} N")
print(f"  TMD damping force: {damping_force:.2f} N")
print(f"  TMD stiffness: {env.tmd_k} N/m")

# ===========================================================================
# CHECK: Trained model policy behavior
# ===========================================================================
print("\n" + "="*70)
print("CHECK: TRAINED MODEL POLICY BEHAVIOR")
print("="*70)

model = SAC.load("rl_cl_robust_models/stage1_50kN_final_robust.zip")
env = make_improved_tmd_env(test_file, max_force=50000)
obs, _ = env.reset()

# Get actions for different observations
test_observations = [
    np.array([0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0]),  # Positive roof disp
    np.array([-0.1, 0.0, -0.1, 0.0, -0.1, 0.0, 0.0, 0.0]),  # Negative roof disp
    np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0]),  # Positive velocity
    np.array([0.0, -0.5, 0.0, -0.5, 0.0, -0.5, 0.0, 0.0]),  # Negative velocity
]

print(f"\nModel policy responses (deterministic):")
for i, test_obs in enumerate(test_observations):
    action, _ = model.predict(test_obs, deterministic=True)
    print(f"  Test {i+1}: obs[0]={test_obs[0]:+.2f}, obs[1]={test_obs[1]:+.2f} → action={action[0]:+.3f}")

# ===========================================================================
# CHECK: SAC hyperparameters
# ===========================================================================
print("\n" + "="*70)
print("CHECK: SAC HYPERPARAMETERS")
print("="*70)

print(f"\nModel hyperparameters:")
print(f"  Learning rate: {model.learning_rate}")
print(f"  Buffer size: {model.buffer_size:,}")
print(f"  Batch size: {model.batch_size}")
print(f"  Tau (soft update): {model.tau}")
print(f"  Gamma (discount): {model.gamma}")
print(f"  Entropy coefficient (alpha): {model.ent_coef}")

if model.learning_rate < 1e-5:
    print(f"  ⚠️  Learning rate may be too low: {model.learning_rate}")
if model.gamma < 0.9:
    print(f"  ⚠️  Discount factor may be too low: {model.gamma}")

# ===========================================================================
# CHECK: Compare with MATLAB expected values
# ===========================================================================
print("\n" + "="*70)
print("CHECK: COMPARISON WITH EXPECTED VALUES")
print("="*70)

# Expected passive TMD performance
print(f"\nExpected baseline (from earlier analysis):")
print(f"  Uncontrolled (no TMD): 21 cm")
print(f"  Passive TMD (k=3765): 20 cm")
print(f"  Passive TMD (k=50000): 21 cm")

# Actual trained model
model = SAC.load("rl_cl_robust_models/stage1_50kN_final_robust.zip")
env = make_improved_tmd_env(test_file, max_force=50000)
obs, _ = env.reset()

done = False
peak = 0
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    peak = max(peak, abs(info['roof_displacement']))
    done = done or truncated

print(f"\nActual trained model (current TMD k=3765):")
print(f"  Peak displacement: {peak*100:.2f} cm")

# ===========================================================================
# CHECK: Soft TMD allows unbounded drift
# ===========================================================================
print("\n" + "="*70)
print("CHECK: TMD DRIFT ACCUMULATION")
print("="*70)

env = make_improved_tmd_env(test_file, max_force=50000)
obs, _ = env.reset()

# Apply constant force and see how much TMD drifts
tmd_drifts = []
for _ in range(100):
    action = np.array([0.5])  # Constant moderate force
    obs, reward, done, truncated, info = env.step(action)
    tmd_drift = env.displacement[12]
    tmd_drifts.append(abs(tmd_drift))
    if done or truncated:
        break

print(f"\nTMD drift with constant 25kN force (first 100 steps):")
print(f"  Initial: {tmd_drifts[0]*100:.2f} cm")
print(f"  After 50 steps: {tmd_drifts[49]*100:.2f} cm")
print(f"  After 100 steps: {tmd_drifts[99]*100:.2f} cm")
print(f"  Growth rate: {(tmd_drifts[99] - tmd_drifts[0]) / 0.1:.2f} m/s")

if tmd_drifts[99] > 1.0:  # More than 1 meter
    print(f"  ❌ TMD is accumulating drift (too soft!)")
else:
    print(f"  ✅ TMD drift is bounded")

print("\n" + "="*70)
print("EXHAUSTIVE CHECK COMPLETE")
print("="*70)
