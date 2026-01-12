"""
Test V13 reward function to verify it produces positive baseline rewards.
"""
import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from restapi.rl_cl.tmd_environment_v13_rooftop import RooftopTMDEnv

def test_reward_function():
    """Test reward function with known states."""

    print("=" * 80)
    print("V13 REWARD FUNCTION VERIFICATION")
    print("=" * 80)

    # Load a test earthquake
    csv_path = r"c:\Dev\dAmpIng26\git\struct-engineer-ai\matlab\datasets\training\training_set_v2\TRAIN_M4.5_PGA0.25g_RMS0.073g_variant1.csv"
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    earthquake_data = data[:, 1]  # Second column is acceleration

    # Create environment
    env = RooftopTMDEnv(
        earthquake_data=earthquake_data,
        dt=0.02,
        max_force=300000.0,
        earthquake_name="Test"
    )

    # Reset to initial state
    obs, info = env.reset()

    print("\n1. Testing reward at ZERO state (start of episode):")
    print("-" * 80)

    # At start, everything should be near zero
    all_floor_drifts = np.zeros(12)  # All floors have zero drift
    control_force = 0.0

    reward = env._calculate_reward(control_force, all_floor_drifts)

    print(f"   State: displacement=0, ISDR=0, DCR=1.0, force=0")
    print(f"   Expected reward: ~8.5 to 9.5 (positive baseline)")
    print(f"   Actual reward: {reward:.3f}")

    if reward > 8.0:
        print("   [PASS] - Reward is positive baseline")
    else:
        print(f"   [FAIL] - Reward should be ~9.0, got {reward:.3f}")

    print("\n2. Testing reward with SMALL displacement (good control):")
    print("-" * 80)

    # Simulate small displacement (7 cm = half target)
    env.d[11] = 0.07  # 7 cm roof displacement
    all_floor_drifts = np.full(12, 0.001)  # 0.1 cm drift per floor
    control_force = 50000.0  # 50 kN

    reward = env._calculate_reward(control_force, all_floor_drifts)

    roof_disp_cm = env.d[11] * 100
    max_isdr_pct = (0.001 / env.story_height) * 100

    print(f"   State: displacement={roof_disp_cm:.1f}cm, max_ISDR={max_isdr_pct:.3f}%, force=50kN")
    print(f"   Expected reward: ~5.0 to 7.0 (good performance)")
    print(f"   Actual reward: {reward:.3f}")

    if reward > 3.0:
        print("   [PASS] - Reward is positive (good control)")
    else:
        print(f"   [FAIL] - Reward should be positive, got {reward:.3f}")

    print("\n3. Testing reward with LARGE displacement (poor control):")
    print("-" * 80)

    # Simulate large displacement (28 cm = 2x target)
    env.d[11] = 0.28  # 28 cm roof displacement
    all_floor_drifts = np.full(12, 0.008)  # 0.8 cm drift per floor (2x target ISDR)
    control_force = 200000.0  # 200 kN

    reward = env._calculate_reward(control_force, all_floor_drifts)

    roof_disp_cm = env.d[11] * 100
    max_isdr_pct = (0.008 / env.story_height) * 100

    print(f"   State: displacement={roof_disp_cm:.1f}cm, max_ISDR={max_isdr_pct:.3f}%, force=200kN")
    print(f"   Expected reward: -10.0 to -20.0 (poor performance)")
    print(f"   Actual reward: {reward:.3f}")

    if reward < -5.0:
        print("   [PASS] - Reward is negative (poor control)")
    else:
        print(f"   [FAIL] - Reward should be very negative, got {reward:.3f}")

    print("\n4. Testing reward gradient (does force penalty work?):")
    print("-" * 80)

    env.d[11] = 0.10  # 10 cm (moderate)
    all_floor_drifts = np.full(12, 0.003)  # Near target ISDR

    reward_no_force = env._calculate_reward(0.0, all_floor_drifts)
    reward_small_force = env._calculate_reward(50000.0, all_floor_drifts)
    reward_large_force = env._calculate_reward(300000.0, all_floor_drifts)

    print(f"   With force=0kN:   reward={reward_no_force:.3f}")
    print(f"   With force=50kN:  reward={reward_small_force:.3f}")
    print(f"   With force=300kN: reward={reward_large_force:.3f}")

    if reward_no_force > reward_small_force > reward_large_force:
        print("   [PASS] - Force penalty works (more force = lower reward)")
    else:
        print("   [FAIL] - Force penalty gradient incorrect")

    print("\n5. Run 10 steps and check reward evolution:")
    print("-" * 80)

    env.reset()

    for step in range(10):
        action = np.array([0.0])  # No control
        obs, reward, done, truncated, info = env.step(action)

        if step < 3:
            print(f"   Step {step}: reward={reward:.3f}, roof_disp={info.get('displacement_roof', 0)*100:.2f}cm")

    print(f"   ...")

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_reward_function()
