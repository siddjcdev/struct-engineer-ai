"""
Investigate why DCR reward is so massive
"""
import sys
sys.path.insert(0, '/Users/Shared/dev/git/struct-engineer-ai')
import numpy as np
from rl.rl_cl.tmd_environment import make_improved_tmd_env

print("="*70)
print("DCR REWARD INVESTIGATION")
print("="*70)

test_file = "../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv"
env = make_improved_tmd_env(test_file, max_force=50000)
obs, _ = env.reset()

# Run first 50 steps
dcr_values = []
dcr_penalties = []
peak_drifts_history = []

done = False
step = 0

while not done and step < 50:
    action = np.array([0.0])
    obs, reward, done, truncated, info = env.step(action)
    
    # Extract DCR-related values
    dcr_penalty = info['reward_breakdown']['dcr']
    dcr_penalties.append(dcr_penalty)
    
    # Calculate current DCR
    if np.max(env.peak_drift_per_floor) > 1e-10:
        sorted_peaks = np.sort(env.peak_drift_per_floor)
        percentile_75 = np.percentile(sorted_peaks, 75)
        max_peak = np.max(env.peak_drift_per_floor)
        
        if percentile_75 > 1e-10:
            current_dcr = max_peak / percentile_75
        else:
            current_dcr = 0.0
    else:
        current_dcr = 0.0
    
    dcr_values.append(current_dcr)
    peak_drifts_history.append(env.peak_drift_per_floor.copy())
    
    done = done or truncated
    step += 1

print(f"\nDCR values over first 50 steps:")
print(f"  Min DCR: {min(dcr_values):.3f}")
print(f"  Max DCR: {max(dcr_values):.3f}")
print(f"  Final DCR: {dcr_values[-1]:.3f}")

print(f"\nDCR penalties over first 50 steps:")
print(f"  Min penalty: {min(dcr_penalties):.2f}")
print(f"  Max penalty: {max(dcr_penalties):.2f}")
print(f"  Final penalty: {dcr_penalties[-1]:.2f}")
print(f"  Average penalty: {np.mean(dcr_penalties):.2f}")

# Check the DCR penalty formula
print(f"\n" + "="*70)
print("DCR PENALTY FORMULA CHECK")
print("="*70)

# From the code:
# dcr_deviation = max(0, current_dcr - 1.0)
# dcr_penalty = -2.0 * (dcr_deviation ** 2)

final_dcr = dcr_values[-1]
dcr_deviation = max(0, final_dcr - 1.0)
expected_penalty = -2.0 * (dcr_deviation ** 2)

print(f"\nFinal step analysis:")
print(f"  Current DCR: {final_dcr:.3f}")
print(f"  DCR deviation: {dcr_deviation:.3f}")
print(f"  Expected penalty: {expected_penalty:.2f}")
print(f"  Actual penalty: {dcr_penalties[-1]:.2f}")

# Check peak drifts
print(f"\n" + "="*70)
print("PEAK DRIFT ANALYSIS")
print("="*70)

final_peaks = peak_drifts_history[-1]
print(f"\nFinal peak drifts per floor (first 50 steps):")
for i in range(12):
    print(f"  Floor {i+1:2d}: {final_peaks[i]*100:.4f} cm")

sorted_peaks = np.sort(final_peaks)
percentile_75 = np.percentile(sorted_peaks, 75)
max_peak = np.max(final_peaks)

print(f"\n75th percentile drift: {percentile_75*100:.4f} cm")
print(f"Max drift: {max_peak*100:.4f} cm")
print(f"DCR: {max_peak/percentile_75:.3f}")

# Plot DCR evolution
print(f"\n" + "="*70)
print("DCR EVOLUTION")
print("="*70)

print(f"\nStep-by-step DCR values (first 20 steps):")
for i in range(min(20, len(dcr_values))):
    print(f"  Step {i:2d}: DCR={dcr_values[i]:6.2f}  Penalty={dcr_penalties[i]:12.2f}")

# Check if there's a bug in the calculation
print(f"\n" + "="*70)
print("POTENTIAL BUG CHECK")
print("="*70)

# The penalty is WAY too large. Let's see if there's a calculation error
# Expected: dcr_penalty = -2.0 * (dcr_deviation ** 2)
# For DCR = 3.0: deviation = 2.0, penalty = -2.0 * 4.0 = -8.0

if abs(dcr_penalties[-1]) > 1000:
    print(f"\n❌ CRITICAL: DCR penalty is {abs(dcr_penalties[-1]):.0f}!")
    print(f"   Expected magnitude: ~10-100 for typical DCR values")
    print(f"   Actual magnitude: {abs(dcr_penalties[-1]):.0f}")
    print(f"\n   This suggests a BUG in the DCR calculation!")
    print(f"   The penalty is {abs(dcr_penalties[-1])/100:.0f}x larger than expected!")
