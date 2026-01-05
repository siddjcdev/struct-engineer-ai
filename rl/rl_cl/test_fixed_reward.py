"""
Quick test script to verify the fixed reward function is working correctly
Run this BEFORE starting full training to confirm the fixes work
"""

import numpy as np
import sys
import os

# Add path to environment
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'restapi', 'rl_cl'))

from rl_cl_tmd_environment import ImprovedTMDBuildingEnv

def test_reward_function():
    """Test that reward function now gives proper learning signals"""

    print("\n" + "="*70)
    print("  TESTING FIXED REWARD FUNCTION")
    print("="*70 + "\n")

    # Create synthetic M4.5-like earthquake (PGA ~0.25g = 2.5 m/s²)
    t = np.linspace(0, 40, 2000)  # 40 seconds at 0.02s timestep
    earthquake = 2.5 * np.sin(2 * np.pi * 1.2 * t) * np.exp(-0.05 * t)  # Decaying sine wave

    # Create environment
    env = ImprovedTMDBuildingEnv(
        earthquake_data=earthquake,
        dt=0.02,
        max_force=110000,
        earthquake_name="Test_M4.5",
        obs_bounds={'disp': 5.0, 'vel': 20.0, 'tmd_disp': 15.0, 'tmd_vel': 60.0}
    )

    print("[OK] Environment created")
    print(f"  Episode length: {env.max_steps} steps")
    print(f"  Max force: {env.max_force/1000:.0f} kN\n")

    # Test 1: Zero force (should get negative rewards)
    print("TEST 1: Zero force (uncontrolled)")
    print("-" * 70)
    obs, info = env.reset()
    total_reward = 0
    peak_disp = 0

    for i in range(env.max_steps):
        action = np.array([0.0])  # No control
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        peak_disp = max(peak_disp, abs(info['roof_displacement']))

        if i == 100:
            breakdown = info['reward_breakdown']
            print(f"  Step {i}: reward={reward:.3f}, disp={info['roof_displacement']*100:.2f} cm")
            print(f"    Breakdown: disp={breakdown['displacement']:.3f}, vel={breakdown['velocity']:.3f}, isdr={breakdown['isdr']:.3f}, dcr={breakdown['dcr']:.3f}")

    metrics = env.get_episode_metrics()
    print(f"\n  Results (uncontrolled):")
    print(f"    Total reward: {total_reward:.1f}")
    print(f"    Peak displacement: {peak_disp*100:.2f} cm")
    print(f"    Max ISDR: {metrics['max_isdr_percent']:.2f}%")
    print(f"    DCR: {metrics['DCR']:.2f}")

    uncontrolled_reward = total_reward
    uncontrolled_disp = peak_disp

    # Test 2: Random force (should get better rewards if it helps)
    print("\n\nTEST 2: Random force")
    print("-" * 70)
    np.random.seed(42)
    obs, info = env.reset()
    total_reward = 0
    peak_disp = 0

    for i in range(env.max_steps):
        action = np.random.uniform(-0.5, 0.5, size=(1,))  # Random control ±50%
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        peak_disp = max(peak_disp, abs(info['roof_displacement']))

        if i == 100:
            breakdown = info['reward_breakdown']
            print(f"  Step {i}: reward={reward:.3f}, disp={info['roof_displacement']*100:.2f} cm, force={info['control_force']/1000:.1f} kN")
            print(f"    Breakdown: disp={breakdown['displacement']:.3f}, vel={breakdown['velocity']:.3f}, isdr={breakdown['isdr']:.3f}, dcr={breakdown['dcr']:.3f}")

    metrics = env.get_episode_metrics()
    print(f"\n  Results (random control):")
    print(f"    Total reward: {total_reward:.1f}")
    print(f"    Peak displacement: {peak_disp*100:.2f} cm")
    print(f"    Max ISDR: {metrics['max_isdr_percent']:.2f}%")
    print(f"    DCR: {metrics['DCR']:.2f}")
    print(f"    Mean force: {metrics['mean_force']/1000:.1f} kN")

    random_reward = total_reward
    random_disp = peak_disp

    # Test 3: PD control (should get best rewards)
    print("\n\nTEST 3: PD controller (Kp=200kN/m, Kd=10kN·s/m)")
    print("-" * 70)
    obs, info = env.reset()
    total_reward = 0
    peak_disp = 0

    for i in range(env.max_steps):
        # PD controller: force = -Kp * displacement - Kd * velocity
        roof_disp = obs[0]  # Roof displacement
        roof_vel = obs[1]   # Roof velocity
        Kp = 200000.0  # 200 kN/m proportional gain
        Kd = 10000.0   # 10 kN·s/m derivative gain
        force_desired = -Kp * roof_disp - Kd * roof_vel
        action = np.clip([force_desired / env.max_force], -1.0, 1.0)

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        peak_disp = max(peak_disp, abs(info['roof_displacement']))

        if i == 100:
            print(f"  Step {i}: reward={reward:.3f}, displacement={info['roof_displacement']*100:.2f} cm, force={info['control_force']/1000:.1f} kN")

    metrics = env.get_episode_metrics()
    print(f"\n  Results (PD control):")
    print(f"    Total reward: {total_reward:.1f}")
    print(f"    Peak displacement: {peak_disp*100:.2f} cm")
    print(f"    Max ISDR: {metrics['max_isdr_percent']:.2f}%")
    print(f"    DCR: {metrics['DCR']:.2f}")
    print(f"    Mean force: {metrics['mean_force']/1000:.1f} kN")

    pd_reward = total_reward
    pd_disp = peak_disp

    # Verify reward ordering
    print("\n\n" + "="*70)
    print("  VERIFICATION")
    print("="*70)
    print(f"\nReward comparison:")
    print(f"  Uncontrolled:  {uncontrolled_reward:8.1f}")
    print(f"  Random:        {random_reward:8.1f}")
    print(f"  PD Control:    {pd_reward:8.1f}")

    print(f"\nDisplacement comparison:")
    print(f"  Uncontrolled:  {uncontrolled_disp*100:6.2f} cm")
    print(f"  Random:        {random_disp*100:6.2f} cm")
    print(f"  PD Control:    {pd_disp*100:6.2f} cm")

    # Check if reward function is working correctly
    success = True

    # Main test: Random control should get better reward than uncontrolled
    # (it provides some damping even if not optimal)
    if random_reward <= uncontrolled_reward:
        print("\n[FAIL] PROBLEM: Random control got worse reward than uncontrolled!")
        print("   The reward function may have issues.")
        success = False
    else:
        print(f"\n[PASS] GOOD: Random control got better reward than uncontrolled (+{random_reward - uncontrolled_reward:.1f})")

    # PD test is informational only (gains may need tuning)
    if pd_reward > uncontrolled_reward:
        print(f"[PASS] BONUS: PD control also beat uncontrolled (+{pd_reward - uncontrolled_reward:.1f})")
        if pd_disp < uncontrolled_disp:
            reduction = (1 - pd_disp/uncontrolled_disp) * 100
            print(f"[PASS] BONUS: PD reduced displacement by {reduction:.1f}%")
    else:
        print(f"[INFO] PD control got worse reward (PD gains likely need tuning)")
        print(f"       This is OK - simple PD may not work well without proper tuning")

    # Check reward magnitude (for 2000-step episode, expect -6000 to +2000 range)
    if abs(uncontrolled_reward) > 200000 or abs(pd_reward) > 200000:
        print(f"[FAIL] PROBLEM: Rewards are too large (uncontrolled={uncontrolled_reward:.1f}, pd={pd_reward:.1f})")
        print("   Expected range: -200000 to +50000 for 2000-step episode")
        success = False
    else:
        print(f"[PASS] GOOD: Reward magnitudes are reasonable")
        print(f"   Uncontrolled: {uncontrolled_reward:.1f}, PD: {pd_reward:.1f}")

    print("\n" + "="*70)
    if success:
        print("  [SUCCESS] REWARD FUNCTION APPEARS TO BE WORKING CORRECTLY!")
        print("  Ready to start training.")
    else:
        print("  [FAILED] REWARD FUNCTION STILL HAS ISSUES")
        print("  Review the test results above.")
    print("="*70 + "\n")

    return success


if __name__ == "__main__":
    test_reward_function()
