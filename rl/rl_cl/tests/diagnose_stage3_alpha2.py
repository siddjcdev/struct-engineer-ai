"""
Emergency diagnostic: Why is Stage 3 giving 237cm (essentially uncontrolled)?
"""
import numpy as np
from stable_baselines3 import SAC
from tmd_environment import make_improved_tmd_env

print("="*70)
print("EMERGENCY DIAGNOSTIC: STAGE 3 FAILURE")
print("="*70)

test_file = "../../matlab/datasets/PEER_high_M7.4_PGA0.75g.csv"
model_path = "rl_cl_models_alpha_2/stage3_150kN_final_robust.zip"

# Load model
print(f"\n1. Loading model: {model_path}")
model = SAC.load(model_path)

# Create environment with SAME bounds as training
obs_bounds = {
    'disp': 5.0,
    'vel': 20.0,
    'tmd_disp': 15.0,
    'tmd_vel': 60.0
}

print(f"\n2. Creating test environment with obs_bounds: {obs_bounds}")
env = make_improved_tmd_env(test_file, max_force=150000.0, obs_bounds=obs_bounds)

# Run simulation and track everything
obs, _ = env.reset()
done = False
forces = []
observations = []
actions = []
displacements = []

print(f"\n3. Running simulation...")
print(f"   Initial observation: {obs}")
print(f"   Observation space: {env.observation_space}")

step = 0
while not done and step < 3000:
    # Get action
    action, _ = model.predict(obs, deterministic=True)

    # Store
    forces.append(float(action[0]) * 150000)
    observations.append(obs.copy())
    actions.append(float(action[0]))

    # Step
    obs, reward, done, truncated, info = env.step(action)
    displacements.append(info['roof_displacement'])
    done = done or truncated
    step += 1

forces = np.array(forces)
actions = np.array(actions)
displacements = np.array(displacements)
observations = np.array(observations)

print(f"\n4. RESULTS:")
print(f"   Peak displacement: {max(abs(displacements))*100:.2f} cm")
print(f"   Uncontrolled expected: 231.56 cm")
print(f"   Steps simulated: {step}")

print(f"\n5. CONTROL ANALYSIS:")
print(f"   Action statistics:")
print(f"      Mean: {np.mean(actions):.6f}")
print(f"      Std:  {np.std(actions):.6f}")
print(f"      Min:  {np.min(actions):.6f}")
print(f"      Max:  {np.max(actions):.6f}")
print(f"   Force statistics:")
print(f"      Mean abs: {np.mean(np.abs(forces))/1000:.2f} kN")
print(f"      Peak:     {np.max(np.abs(forces))/1000:.2f} kN")
print(f"      Max allowed: 150 kN")

if np.std(actions) < 0.001:
    print(f"\n   ❌ PROBLEM: Actions are nearly constant!")
    print(f"      Model is not responding to observations")
else:
    print(f"\n   ✓ Model is varying actions")

print(f"\n6. OBSERVATION ANALYSIS:")
print(f"   Roof displacement range: [{np.min(observations[:,0]):.3f}, {np.max(observations[:,0]):.3f}]")
print(f"   Roof velocity range:     [{np.min(observations[:,1]):.3f}, {np.max(observations[:,1]):.3f}]")
print(f"   TMD displacement range:  [{np.min(observations[:,6]):.3f}, {np.max(observations[:,6]):.3f}]")
print(f"   TMD velocity range:      [{np.min(observations[:,7]):.3f}, {np.max(observations[:,7]):.3f}]")

# Check if observations are clipped
obs_at_bounds = 0
for i in range(len(observations)):
    if np.any(np.abs(observations[i]) > 4.9):  # Close to ±5.0 bound
        obs_at_bounds += 1

if obs_at_bounds > 0:
    print(f"\n   ⚠️  {obs_at_bounds}/{len(observations)} observations near bounds (possible clipping)")
else:
    print(f"\n   ✓ No observations near bounds")

print(f"\n7. SAMPLE TIMESTEPS (first 10 peak displacement moments):")
peak_indices = np.argsort(np.abs(displacements))[-10:][::-1]
for idx in peak_indices:
    if idx < len(observations):
        print(f"   Step {idx}: disp={displacements[idx]*100:6.2f}cm, "
              f"action={actions[idx]:+.3f}, force={forces[idx]/1000:+6.1f}kN, "
              f"obs[0]={observations[idx,0]:+.3f}")

print("\n" + "="*70)
print("DIAGNOSIS:")
if np.std(actions) < 0.001:
    print("❌ Model outputs constant actions → Not learning!")
    print("   Possible causes:")
    print("   - Observation space mismatch between training and testing")
    print("   - Model didn't train properly")
    print("   - Wrong model file loaded")
elif np.mean(np.abs(forces)) < 1000:
    print("❌ Forces are too weak (< 1kN average)")
    print("   Model learned to do nothing")
elif obs_at_bounds > len(observations) * 0.1:
    print("❌ Observations are being clipped (>10% at bounds)")
    print("   Model is blind to true state")
else:
    print("⚠️  Model is responding but not effective")
    print("   May need more training or reward tuning")
print("="*70 + "\n")
