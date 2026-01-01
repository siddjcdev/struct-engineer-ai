"""
Quick check of uncontrolled baselines for all PEER earthquakes
"""
from tmd_environment_shaped_reward import make_improved_tmd_env
import numpy as np

earthquakes = [
    ("M4.5", "../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv"),
    ("M5.7", "../../matlab/datasets/PEER_moderate_M5.7_PGA0.35g.csv"),
    ("M7.4", "../../matlab/datasets/PEER_high_M7.4_PGA0.75g.csv"),
    ("M8.4", "../../matlab/datasets/PEER_insane_M8.4_PGA0.9g.csv"),
]

print("="*70)
print("  UNCONTROLLED BASELINE CHECK")
print("="*70)
print("\nTesting with NO control (zero force)...\n")

results = []

for name, eq_file in earthquakes:
    env = make_improved_tmd_env(eq_file, max_force=150000)
    obs, _ = env.reset()
    done = False
    peak = 0

    while not done:
        # Apply ZERO control
        obs, reward, done, truncated, info = env.step(np.array([0.0]))
        peak = max(peak, abs(info['roof_displacement']))
        done = done or truncated

    peak_cm = peak * 100
    results.append((name, peak_cm))

    print(f"   {name}: {peak_cm:.2f} cm (uncontrolled)")

print("\n" + "="*70)
print("  SUMMARY - UNCONTROLLED BASELINES")
print("="*70)
for name, peak_cm in results:
    print(f"   {name}: {peak_cm:.2f} cm (uncontrolled)")
print("="*70)

print("\n" + "="*70)
print("  COMPARISON WITH RL CURRICULUM RESULTS")
print("="*70)
print("\nRL Results from training:")
rl_results = [
    ("M4.5", 20.85),
    ("M5.7", 44.44),
    ("M7.4", 221.04),
    # M8.4 will be added after Stage 4 completes
]

for i, ((name_uncont, uncont_cm), (name_rl, rl_cm)) in enumerate(zip(results[:3], rl_results)):
    improvement = 100 * (uncont_cm - rl_cm) / uncont_cm
    status = "✓" if improvement > 0 else "✗"
    print(f"   {name_uncont}:")
    print(f"      Uncontrolled: {uncont_cm:.2f} cm")
    print(f"      RL (v4):      {rl_cm:.2f} cm")
    print(f"      Improvement:  {improvement:+.1f}% {status}\n")

print("="*70 + "\n")
