"""
TMD Tuning Diagnostic
=====================

This script calculates the optimal TMD parameters and compares with current values.
A properly tuned TMD should show significant improvement over uncontrolled.

Author: Claude Code
Date: January 2026
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath('../../restapi/rl_cl'))
from tmd_environment_adaptive_reward import ImprovedTMDBuildingEnv


def load_earthquake(filepath):
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    if data.shape[1] >= 2:
        return data[:, 1], float(np.mean(np.diff(data[:, 0])))
    return data.flatten(), 0.02


def calculate_building_frequency(env):
    """Calculate fundamental frequency of the building"""
    # Eigenvalue analysis of M^-1 K (just the building, not TMD)
    n = env.n_floors
    M_bldg = env.M[:n, :n]
    K_bldg = env.K[:n, :n]

    eigenvalues = np.linalg.eigvals(np.linalg.solve(M_bldg, K_bldg))
    omega_squared = np.real(eigenvalues[eigenvalues > 1e-10])
    omega = np.sqrt(np.sort(omega_squared))

    f1 = omega[0] / (2 * np.pi) if len(omega) > 0 else 0
    return f1, omega[0] if len(omega) > 0 else 0


def run_test(env, name, control_fn=None):
    """Run a test and return results"""
    obs, _ = env.reset()
    done = False
    peak_disp = 0

    while not done:
        action = control_fn(obs) if control_fn else 0.0
        obs, _, done, truncated, info = env.step(np.array([action]))
        peak_disp = max(peak_disp, abs(info['roof_displacement']))
        done = done or truncated

    metrics = env.get_episode_metrics()
    return {
        'name': name,
        'disp_cm': peak_disp * 100,
        'isdr': metrics['max_isdr_percent'],
        'dcr': metrics['DCR']
    }


class OptimallyTunedEnv(ImprovedTMDBuildingEnv):
    """Environment with CORRECTLY tuned TMD"""

    def __init__(self, earthquake_data, dt, max_force, use_optimal_tuning=True):
        super().__init__(
            earthquake_data=earthquake_data,
            dt=dt,
            max_force=max_force,
            earthquake_name="DIAGNOSTIC",
            reward_scale=1.0
        )

        if use_optimal_tuning:
            # Calculate building frequency
            n = self.n_floors
            M_bldg = self.M[:n, :n]
            K_bldg = self.K[:n, :n]
            eigenvalues = np.linalg.eigvals(np.linalg.solve(M_bldg, K_bldg))
            omega_squared = np.real(eigenvalues[eigenvalues > 1e-10])
            omega_bldg = np.sqrt(np.min(omega_squared))

            # Optimal TMD tuning (Den Hartog)
            mu = self.tmd_mass / (self.n_floors * self.floor_mass)  # Mass ratio

            # Optimal frequency ratio
            f_ratio = 1 / (1 + mu)
            omega_tmd = f_ratio * omega_bldg

            # Optimal damping ratio for TMD
            zeta_opt = np.sqrt(3 * mu / (8 * (1 + mu)))

            # Calculate optimal k and c
            self.tmd_k = self.tmd_mass * omega_tmd**2
            self.tmd_c = 2 * zeta_opt * np.sqrt(self.tmd_k * self.tmd_mass)

            print(f"\n  OPTIMAL TMD TUNING:")
            print(f"    Building ω1: {omega_bldg:.3f} rad/s ({omega_bldg/(2*np.pi):.3f} Hz)")
            print(f"    Mass ratio μ: {mu*100:.2f}%")
            print(f"    Optimal TMD ω: {omega_tmd:.3f} rad/s ({omega_tmd/(2*np.pi):.3f} Hz)")
            print(f"    Optimal TMD k: {self.tmd_k/1000:.2f} kN/m")
            print(f"    Optimal TMD c: {self.tmd_c:.0f} N·s/m")
            print(f"    Optimal TMD ζ: {zeta_opt*100:.1f}%")

            # Rebuild matrices
            self.K = self._build_stiffness_matrix()
            self.C = self._build_damping_matrix()


def main():
    print("=" * 70)
    print("  TMD TUNING DIAGNOSTIC")
    print("=" * 70)

    # Load earthquake
    eq_file = "../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv"
    if not os.path.exists(eq_file):
        eq_file = "../../matlab/datasets/test/PEER_small_M4.5_PGA0.25g.csv"

    earthquake_data, dt = load_earthquake(eq_file)
    print(f"\nLoaded earthquake: {len(earthquake_data)} samples")

    max_force = 250000

    # Current (potentially detuned) TMD
    print("\n" + "=" * 70)
    print("  CURRENT TMD CONFIGURATION")
    print("=" * 70)

    env_current = ImprovedTMDBuildingEnv(
        earthquake_data=earthquake_data,
        dt=dt,
        max_force=max_force,
        earthquake_name="CURRENT",
        reward_scale=1.0
    )

    # Calculate current TMD frequency
    tmd_omega = np.sqrt(env_current.tmd_k / env_current.tmd_mass)
    tmd_freq = tmd_omega / (2 * np.pi)

    # Calculate building frequency
    bldg_freq, bldg_omega = calculate_building_frequency(env_current)

    print(f"\n  Building fundamental frequency: {bldg_freq:.3f} Hz ({bldg_omega:.3f} rad/s)")
    print(f"\n  Current TMD parameters:")
    print(f"    Mass:      {env_current.tmd_mass:.0f} kg ({env_current.tmd_mass/env_current.floor_mass*100:.1f}% of floor mass)")
    print(f"    Stiffness: {env_current.tmd_k/1000:.1f} kN/m")
    print(f"    Damping:   {env_current.tmd_c:.0f} N·s/m")
    print(f"    Frequency: {tmd_freq:.3f} Hz ({tmd_omega:.3f} rad/s)")

    freq_ratio = tmd_freq / bldg_freq
    print(f"\n  FREQUENCY RATIO (TMD/Building): {freq_ratio:.2f}")

    if freq_ratio > 1.5 or freq_ratio < 0.5:
        print(f"  ⚠️  WARNING: TMD is SEVERELY DETUNED!")
        print(f"      Optimal ratio should be ~0.96")
        print(f"      Current ratio is {freq_ratio:.2f}x off!")
    elif freq_ratio > 1.1 or freq_ratio < 0.85:
        print(f"  ⚠️  WARNING: TMD is moderately detuned")
    else:
        print(f"  ✅ TMD frequency is approximately correct")

    # Test with current TMD
    result_current = run_test(env_current, "Current TMD (passive)")
    env_current.close()

    # Optimal TMD
    print("\n" + "=" * 70)
    print("  OPTIMALLY TUNED TMD")
    print("=" * 70)

    env_optimal = OptimallyTunedEnv(
        earthquake_data=earthquake_data,
        dt=dt,
        max_force=max_force,
        use_optimal_tuning=True
    )

    result_optimal = run_test(env_optimal, "Optimal TMD (passive)")
    env_optimal.close()

    # No TMD baseline
    print("\n" + "=" * 70)
    print("  NO TMD BASELINE")
    print("=" * 70)

    env_no_tmd = ImprovedTMDBuildingEnv(
        earthquake_data=earthquake_data,
        dt=dt,
        max_force=max_force,
        earthquake_name="NO_TMD",
        reward_scale=1.0
    )
    # Disable TMD
    env_no_tmd.tmd_mass = 1.0
    env_no_tmd.tmd_k = 1.0
    env_no_tmd.tmd_c = 1.0
    env_no_tmd.M = env_no_tmd._build_mass_matrix()
    env_no_tmd.K = env_no_tmd._build_stiffness_matrix()
    env_no_tmd.C = env_no_tmd._build_damping_matrix()

    result_no_tmd = run_test(env_no_tmd, "No TMD")
    env_no_tmd.close()

    # Summary
    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)

    print(f"\n  {'Configuration':<25} {'Disp (cm)':<12} {'ISDR%':<10} {'DCR':<8} {'vs No TMD':<10}")
    print(f"  {'-' * 65}")

    base = result_no_tmd['disp_cm']
    for r in [result_no_tmd, result_current, result_optimal]:
        improve = 100 * (base - r['disp_cm']) / base
        print(f"  {r['name']:<25} {r['disp_cm']:<12.2f} {r['isdr']:<10.2f} {r['dcr']:<8.2f} {improve:+.1f}%")

    print("\n" + "=" * 70)
    print("  DIAGNOSIS")
    print("=" * 70)

    current_improve = 100 * (base - result_current['disp_cm']) / base
    optimal_improve = 100 * (base - result_optimal['disp_cm']) / base

    if optimal_improve > current_improve + 5:
        print(f"""
  THE TMD IS DETUNED!

  Current TMD improvement:  {current_improve:+.1f}%
  Optimal TMD improvement:  {optimal_improve:+.1f}%

  FIX: Update tmd_k in tmd_environment_adaptive_reward.py to:
       self.tmd_k = {env_optimal.tmd_k:.0f}  # {env_optimal.tmd_k/1000:.2f} kN/m
       self.tmd_c = {env_optimal.tmd_c:.0f}  # N·s/m
""")
    else:
        print(f"""
  TMD tuning appears correct.
  Current improvement: {current_improve:+.1f}%
  Optimal improvement: {optimal_improve:+.1f}%
""")


if __name__ == "__main__":
    main()
