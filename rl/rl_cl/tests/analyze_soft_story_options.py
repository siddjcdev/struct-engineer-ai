"""
Soft Story Analysis: Comparing TMD effectiveness under different scenarios
==========================================================================

This script tests:
1. Current configuration (0.6 soft story factor - SEVERE)
2. Reduced soft story (0.7 factor - MODERATE)
3. Inter-story dampers only (no rooftop TMD)
4. TMD + Inter-story dampers (hybrid)
5. No intervention baseline

Goal: Prove TMDs work in soft-story conditions for science fair

Author: Claude Code
Date: January 2026
"""

import sys
import os
import numpy as np
from typing import Dict, Tuple

# Add restapi path
sys.path.insert(0, os.path.abspath('../../restapi/rl_cl'))

from tmd_environment_adaptive_reward import ImprovedTMDBuildingEnv


def load_earthquake(filepath: str) -> Tuple[np.ndarray, float]:
    """Load earthquake data from CSV"""
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    if data.shape[1] >= 2:
        times = data[:, 0]
        accelerations = data[:, 1]
        dt = float(np.mean(np.diff(times)))
    else:
        accelerations = data.flatten()
        dt = 0.02
    return accelerations, dt


class ConfigurableTMDEnv(ImprovedTMDBuildingEnv):
    """
    Environment with configurable soft story factor and inter-story damping.

    Instead of overriding matrix building, we modify parameters and rebuild.
    """

    def __init__(
        self,
        earthquake_data: np.ndarray,
        dt: float = 0.02,
        max_force: float = 250000.0,
        earthquake_name: str = "Unknown",
        reward_scale: float = 1.0,
        soft_story_factor: float = 0.6,
        interstory_damping: float = 0.0,
        tmd_enabled: bool = True
    ):
        # Store config before parent init
        self._soft_story_factor = soft_story_factor
        self._interstory_damping = interstory_damping
        self._tmd_enabled = tmd_enabled

        # Call parent init
        super().__init__(
            earthquake_data=earthquake_data,
            dt=dt,
            max_force=max_force,
            earthquake_name=earthquake_name,
            reward_scale=reward_scale
        )

        # Apply custom soft story factor
        k_typical = 2.0e7  # 20 MN/m (must match parent)
        self.story_stiffness[7] = self._soft_story_factor * k_typical

        # Disable TMD if requested (for baseline comparison)
        if not self._tmd_enabled:
            self.tmd_mass = 1.0  # Negligible mass
            self.tmd_k = 1.0     # Negligible stiffness
            self.tmd_c = 1.0     # Negligible damping

        # Rebuild matrices with new parameters
        self.M = self._build_mass_matrix()
        self.K = self._build_stiffness_matrix()
        self.C = self._build_damping_matrix()

        # Add inter-story damping if specified
        if self._interstory_damping > 0:
            self._add_interstory_damping()

    def _add_interstory_damping(self):
        """Add viscous dampers between floors"""
        c = self._interstory_damping

        # Inter-story damping matrix contribution
        # Damper between floor i and i+1 adds:
        #   C[i,i] += c, C[i+1,i+1] += c, C[i,i+1] -= c, C[i+1,i] -= c
        for i in range(self.n_floors - 1):
            self.C[i, i] += c
            self.C[i + 1, i + 1] += c
            self.C[i, i + 1] -= c
            self.C[i + 1, i] -= c

        # Also add damper at ground level (floor 0 to ground)
        self.C[0, 0] += c


def run_simulation(env, control_fn=None) -> Dict[str, float]:
    """Run a simulation with optional control"""
    obs, _ = env.reset()
    done = False
    step = 0
    peak_disp = 0
    total_force = 0
    force_count = 0

    while not done:
        if control_fn is not None:
            action = control_fn(obs, step)
            action = np.clip(action, -1.0, 1.0)
        else:
            action = 0.0  # No active control

        obs, reward, done, truncated, info = env.step(np.array([action]))
        peak_disp = max(peak_disp, abs(info['roof_displacement']))
        total_force += abs(info['control_force'])
        force_count += 1
        step += 1
        done = done or truncated

    metrics = env.get_episode_metrics()
    return {
        'displacement_cm': peak_disp * 100,
        'isdr_percent': metrics['max_isdr_percent'],
        'dcr': metrics['DCR'],
        'mean_force_kN': (total_force / max(force_count, 1)) / 1000
    }


def create_pd_controller(Kp, Kd, max_force):
    """Create a PD controller"""
    def controller(obs, step):
        roof_disp = obs[0] * 5.0  # Denormalize (bounds = 5m)
        roof_vel = obs[1] * 20.0  # Denormalize (bounds = 20m/s)
        force = -Kp * roof_disp - Kd * roof_vel
        return force / max_force
    return controller


def find_best_pd_gains(env, max_force):
    """Search for best PD gains"""
    best_result = None
    best_gains = None

    # Test various gain combinations
    for Kp in [25000, 50000, 75000, 100000, 150000]:
        for Kd in [2500, 5000, 10000, 15000]:
            controller = create_pd_controller(Kp, Kd, max_force)
            result = run_simulation(env, controller)

            if best_result is None or result['displacement_cm'] < best_result['displacement_cm']:
                best_result = result
                best_gains = (Kp, Kd)

    return best_result, best_gains


def main():
    print("=" * 80)
    print("  SOFT STORY ANALYSIS FOR SCIENCE FAIR")
    print("  Testing TMD effectiveness under different configurations")
    print("=" * 80)

    # Load earthquake
    earthquake_file = "../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv"
    if not os.path.exists(earthquake_file):
        earthquake_file = "../../matlab/datasets/test/PEER_small_M4.5_PGA0.25g.csv"

    if not os.path.exists(earthquake_file):
        print("Earthquake file not found!")
        return

    earthquake_data, dt = load_earthquake(earthquake_file)
    print(f"\nLoaded M4.5 earthquake: {len(earthquake_data)} samples, dt={dt:.3f}s")

    max_force = 250000  # 250 kN

    results = {}

    # =========================================================================
    # TEST 1: NO INTERVENTION (bare building with soft story)
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("  TEST 1: NO INTERVENTION BASELINE")
    print("  Building with 0.6 soft story, NO TMD, NO dampers")
    print("=" * 80)

    env_bare = ConfigurableTMDEnv(
        earthquake_data=earthquake_data,
        dt=dt,
        max_force=max_force,
        earthquake_name="BARE",
        soft_story_factor=0.6,
        interstory_damping=0.0,
        tmd_enabled=False  # No TMD
    )

    result_bare = run_simulation(env_bare, control_fn=None)
    results['bare'] = result_bare
    print(f"\n  Results (no intervention):")
    print(f"    Peak Displacement: {result_bare['displacement_cm']:.2f} cm")
    print(f"    Max ISDR:          {result_bare['isdr_percent']:.2f}%")
    print(f"    DCR:               {result_bare['dcr']:.2f}")
    env_bare.close()

    # =========================================================================
    # TEST 2: PASSIVE TMD ONLY (0.6 soft story)
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("  TEST 2: PASSIVE TMD (no active control)")
    print("  Building with 0.6 soft story + passive TMD")
    print("=" * 80)

    env_passive = ConfigurableTMDEnv(
        earthquake_data=earthquake_data,
        dt=dt,
        max_force=max_force,
        earthquake_name="PASSIVE_TMD",
        soft_story_factor=0.6,
        tmd_enabled=True
    )

    result_passive = run_simulation(env_passive, control_fn=None)
    results['passive_tmd'] = result_passive
    improvement = 100 * (result_bare['displacement_cm'] - result_passive['displacement_cm']) / result_bare['displacement_cm']
    print(f"\n  Results (passive TMD):")
    print(f"    Peak Displacement: {result_passive['displacement_cm']:.2f} cm ({improvement:+.1f}% vs bare)")
    print(f"    Max ISDR:          {result_passive['isdr_percent']:.2f}%")
    print(f"    DCR:               {result_passive['dcr']:.2f}")
    env_passive.close()

    # =========================================================================
    # TEST 3: ACTIVE TMD (0.6 soft story)
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("  TEST 3: ACTIVE TMD CONTROL (0.6 soft story)")
    print("  Building with 0.6 soft story + active TMD with optimal PD control")
    print("=" * 80)

    env_active_06 = ConfigurableTMDEnv(
        earthquake_data=earthquake_data,
        dt=dt,
        max_force=max_force,
        earthquake_name="ACTIVE_TMD_06",
        soft_story_factor=0.6,
        tmd_enabled=True
    )

    print("\n  Searching for optimal PD gains...")
    result_active_06, best_gains = find_best_pd_gains(env_active_06, max_force)
    results['active_tmd_06'] = result_active_06

    improvement = 100 * (result_bare['displacement_cm'] - result_active_06['displacement_cm']) / result_bare['displacement_cm']
    print(f"\n  Results (active TMD, best PD: Kp={best_gains[0]/1000:.0f}k, Kd={best_gains[1]/1000:.1f}k):")
    print(f"    Peak Displacement: {result_active_06['displacement_cm']:.2f} cm ({improvement:+.1f}% vs bare)")
    print(f"    Max ISDR:          {result_active_06['isdr_percent']:.2f}%")
    print(f"    DCR:               {result_active_06['dcr']:.2f}")
    env_active_06.close()

    # =========================================================================
    # TEST 4: ACTIVE TMD (0.7 soft story - LESS SEVERE)
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("  TEST 4: ACTIVE TMD CONTROL (0.7 soft story - less severe)")
    print("  Building with 0.7 soft story + active TMD")
    print("=" * 80)

    env_active_07 = ConfigurableTMDEnv(
        earthquake_data=earthquake_data,
        dt=dt,
        max_force=max_force,
        earthquake_name="ACTIVE_TMD_07",
        soft_story_factor=0.7,
        tmd_enabled=True
    )

    # Also get bare building with 0.7 for fair comparison
    env_bare_07 = ConfigurableTMDEnv(
        earthquake_data=earthquake_data,
        dt=dt,
        max_force=max_force,
        earthquake_name="BARE_07",
        soft_story_factor=0.7,
        tmd_enabled=False
    )
    result_bare_07 = run_simulation(env_bare_07, control_fn=None)
    results['bare_07'] = result_bare_07
    env_bare_07.close()

    result_active_07, best_gains_07 = find_best_pd_gains(env_active_07, max_force)
    results['active_tmd_07'] = result_active_07

    improvement_vs_bare = 100 * (result_bare['displacement_cm'] - result_active_07['displacement_cm']) / result_bare['displacement_cm']
    improvement_vs_bare_07 = 100 * (result_bare_07['displacement_cm'] - result_active_07['displacement_cm']) / result_bare_07['displacement_cm']

    print(f"\n  Bare building (0.7 SS): {result_bare_07['displacement_cm']:.2f} cm, {result_bare_07['isdr_percent']:.2f}% ISDR")
    print(f"\n  Results (active TMD, 0.7 SS, best PD: Kp={best_gains_07[0]/1000:.0f}k, Kd={best_gains_07[1]/1000:.1f}k):")
    print(f"    Peak Displacement: {result_active_07['displacement_cm']:.2f} cm ({improvement_vs_bare_07:+.1f}% vs bare 0.7)")
    print(f"    Max ISDR:          {result_active_07['isdr_percent']:.2f}%")
    print(f"    DCR:               {result_active_07['dcr']:.2f}")
    env_active_07.close()

    # =========================================================================
    # TEST 5: INTER-STORY DAMPERS ONLY (no TMD)
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("  TEST 5: INTER-STORY DAMPERS ONLY (no TMD)")
    print("  Testing various damping levels")
    print("=" * 80)

    damping_levels = [10000, 25000, 50000, 100000, 200000]  # N·s/m per connection

    print(f"\n  {'Damping (kN·s/m)':<18} {'Disp (cm)':<12} {'ISDR%':<10} {'DCR':<8} {'vs Bare':<10}")
    print(f"  {'-' * 60}")

    best_isd_result = None
    best_isd_damping = 0

    for damping in damping_levels:
        env_isd = ConfigurableTMDEnv(
            earthquake_data=earthquake_data,
            dt=dt,
            max_force=0,  # No active control
            earthquake_name=f"ISD_{damping}",
            soft_story_factor=0.6,
            interstory_damping=damping,
            tmd_enabled=False  # No TMD
        )

        result = run_simulation(env_isd, control_fn=None)
        improvement = 100 * (result_bare['displacement_cm'] - result['displacement_cm']) / result_bare['displacement_cm']
        print(f"  {damping/1000:<18.0f} {result['displacement_cm']:<12.2f} {result['isdr_percent']:<10.2f} {result['dcr']:<8.2f} {improvement:+.1f}%")

        if best_isd_result is None or result['displacement_cm'] < best_isd_result['displacement_cm']:
            best_isd_result = result
            best_isd_damping = damping

        env_isd.close()

    results['interstory_dampers'] = best_isd_result

    print(f"\n  Best damping level: {best_isd_damping/1000:.0f} kN·s/m per connection")

    # =========================================================================
    # TEST 6: HYBRID (TMD + Inter-story dampers)
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("  TEST 6: HYBRID SYSTEM (Active TMD + Inter-story dampers)")
    print("=" * 80)

    env_hybrid = ConfigurableTMDEnv(
        earthquake_data=earthquake_data,
        dt=dt,
        max_force=max_force,
        earthquake_name="HYBRID",
        soft_story_factor=0.6,
        interstory_damping=best_isd_damping,
        tmd_enabled=True
    )

    result_hybrid, hybrid_gains = find_best_pd_gains(env_hybrid, max_force)
    results['hybrid'] = result_hybrid

    improvement = 100 * (result_bare['displacement_cm'] - result_hybrid['displacement_cm']) / result_bare['displacement_cm']
    print(f"\n  Results (hybrid system):")
    print(f"    Peak Displacement: {result_hybrid['displacement_cm']:.2f} cm ({improvement:+.1f}% vs bare)")
    print(f"    Max ISDR:          {result_hybrid['isdr_percent']:.2f}%")
    print(f"    DCR:               {result_hybrid['dcr']:.2f}")
    env_hybrid.close()

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("  SUMMARY: COMPARISON OF ALL CONFIGURATIONS")
    print("=" * 80)

    print(f"\n  {'Configuration':<35} {'Disp(cm)':<10} {'ISDR%':<8} {'DCR':<8} {'vs Bare':<10}")
    print(f"  {'-' * 75}")

    configs = [
        ("No intervention (0.6 SS)", results['bare']),
        ("Passive TMD only (0.6 SS)", results['passive_tmd']),
        ("Active TMD (0.6 SS)", results['active_tmd_06']),
        ("Active TMD (0.7 SS)", results['active_tmd_07']),
        ("Inter-story dampers only", results['interstory_dampers']),
        ("Hybrid (TMD + dampers)", results['hybrid']),
    ]

    for name, result in configs:
        improvement = 100 * (results['bare']['displacement_cm'] - result['displacement_cm']) / results['bare']['displacement_cm']
        print(f"  {name:<35} {result['displacement_cm']:<10.2f} {result['isdr_percent']:<8.2f} {result['dcr']:<8.2f} {improvement:+.1f}%")

    # =========================================================================
    # SCIENCE FAIR CONCLUSIONS
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("  CONCLUSIONS FOR SCIENCE FAIR HYPOTHESIS")
    print("=" * 80)

    passive_improve = 100 * (results['bare']['displacement_cm'] - results['passive_tmd']['displacement_cm']) / results['bare']['displacement_cm']
    active_improve = 100 * (results['bare']['displacement_cm'] - results['active_tmd_06']['displacement_cm']) / results['bare']['displacement_cm']
    isd_improve = 100 * (results['bare']['displacement_cm'] - results['interstory_dampers']['displacement_cm']) / results['bare']['displacement_cm']
    hybrid_improve = 100 * (results['bare']['displacement_cm'] - results['hybrid']['displacement_cm']) / results['bare']['displacement_cm']

    print(f"""
  HYPOTHESIS: TMDs are effective for seismic control in soft-story buildings

  FINDINGS:

  1. TMD EFFECTIVENESS:
     - Passive TMD: {passive_improve:+.1f}% displacement reduction
     - Active TMD:  {active_improve:+.1f}% displacement reduction
     - VERDICT: {'TMDs ARE EFFECTIVE' if active_improve > 5 else 'TMDs have LIMITED effectiveness'}

  2. COMPARISON WITH ALTERNATIVES:
     - Inter-story dampers: {isd_improve:+.1f}% reduction
     - Hybrid system:       {hybrid_improve:+.1f}% reduction
     - BEST APPROACH: {'Hybrid' if hybrid_improve > max(active_improve, isd_improve) else ('Active TMD' if active_improve > isd_improve else 'Inter-story dampers')}

  3. SOFT STORY IMPACT:
     - 0.6 factor (severe):   ISDR = {results['active_tmd_06']['isdr_percent']:.2f}%
     - 0.7 factor (moderate): ISDR = {results['active_tmd_07']['isdr_percent']:.2f}%
     - Reducing severity by 10% {'helps' if results['active_tmd_07']['isdr_percent'] < results['active_tmd_06']['isdr_percent'] else 'does not help'} ISDR

  4. KEY INSIGHT:
     - TMDs effectively reduce GLOBAL displacement (roof movement)
     - LOCAL drift (ISDR at soft story) is harder to control from roof
     - For maximum protection, combine TMD with local dampers

  TARGET ACHIEVEMENT (0.6 soft story):
     Displacement: {results['active_tmd_06']['displacement_cm']:.2f} cm (target ≤18) - {'ACHIEVED' if results['active_tmd_06']['displacement_cm'] <= 18 else 'NOT ACHIEVED'}
     ISDR: {results['active_tmd_06']['isdr_percent']:.2f}% (target ≤0.5%) - {'ACHIEVED' if results['active_tmd_06']['isdr_percent'] <= 0.5 else 'NOT ACHIEVED'}
     DCR: {results['active_tmd_06']['dcr']:.2f} (target ≤1.1) - {'ACHIEVED' if results['active_tmd_06']['dcr'] <= 1.1 else 'NOT ACHIEVED'}
""")


if __name__ == "__main__":
    main()
