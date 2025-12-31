"""
Check observation clipping across all stages
"""
import sys
sys.path.insert(0, '/Users/Shared/dev/git/struct-engineer-ai')
import numpy as np
from rl.rl_cl.tmd_environment import make_improved_tmd_env
from stable_baselines3 import SAC

datasets = [
    ("../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv", "rl_cl_robust_models/stage1_50kN_final_robust.zip", 50000),
    ("../../matlab/datasets/PEER_moderate_M5.7_PGA0.35g.csv", "rl_cl_robust_models/stage2_100kN_final_robust.zip", 100000),
    ("../../matlab/datasets/PEER_high_M7.4_PGA0.75g.csv", "rl_cl_robust_models/stage3_150kN_final_robust.zip", 150000),
]

print("="*70)
print("OBSERVATION CLIPPING CHECK - ALL STAGES")
print("="*70)

for stage, (eq_file, model_path, max_force) in enumerate(datasets, 1):
    print(f"\n{'='*70}")
    print(f"STAGE {stage}")
    print(f"{'='*70}")
    
    model = SAC.load(model_path)
    env = make_improved_tmd_env(eq_file, max_force=max_force)
    obs, _ = env.reset()
    
    obs_min = obs.copy()
    obs_max = obs.copy()
    done = False
    clipped_count = 0
    total_steps = 0
    peak_disp = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        obs_min = np.minimum(obs_min, obs)
        obs_max = np.maximum(obs_max, obs)
        
        # Check if any observation is being clipped
        if np.any(obs <= env.observation_space.low) or np.any(obs >= env.observation_space.high):
            clipped_count += 1
        
        peak_disp = max(peak_disp, abs(info['roof_displacement']))
        total_steps += 1
        done = done or truncated
    
    print(f"\nPeak displacement: {peak_disp*100:.2f} cm")
    print(f"Clipped steps: {clipped_count}/{total_steps} ({clipped_count/total_steps*100:.1f}%)")
    
    obs_labels = ['roof_disp', 'roof_vel', 'floor8_disp', 'floor8_vel',
                  'floor6_disp', 'floor6_vel', 'tmd_disp', 'tmd_vel']
    
    print(f"\nObservation ranges:")
    for i, label in enumerate(obs_labels):
        exceeds = obs_min[i] < env.observation_space.low[i] or obs_max[i] > env.observation_space.high[i]
        status = "❌ CLIPPED" if exceeds else "✅ OK"
        print(f"  {label:12s}: [{obs_min[i]:+7.3f}, {obs_max[i]:+7.3f}]  "
              f"(bounds: [{env.observation_space.low[i]:+4.1f}, {env.observation_space.high[i]:+4.1f}])  {status}")
