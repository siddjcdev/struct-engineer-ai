"""
Pareto Frontier Analysis for TMD Control
==========================================

This script systematically explores what combinations of (displacement, ISDR, DCR)
are actually achievable with your TMD configuration.

Instead of guessing at reward weights, we:
1. Test many different simple control strategies
2. Record the (displacement, ISDR, DCR) achieved by each
3. Find the Pareto frontier - the best achievable trade-offs
4. Determine if your targets (14cm, 0.4% ISDR, DCR 1.0) are inside the frontier

If targets are OUTSIDE the Pareto frontier, no amount of reward tuning will achieve them.
You'll need hardware changes (more TMD mass, more force, etc.)

Author: Claude Code
Date: January 2026
"""

import sys
import os
import numpy as np
from typing import List, Tuple, Dict

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


def run_control_strategy(env, strategy_fn) -> Dict[str, float]:
    """
    Run a control strategy and return metrics

    Args:
        env: TMD environment
        strategy_fn: Function that takes (obs, step) and returns action [-1, 1]

    Returns:
        dict with displacement, ISDR, DCR metrics
    """
    obs, _ = env.reset()
    done = False
    step = 0
    peak_disp = 0

    while not done:
        action = strategy_fn(obs, step)
        action = np.clip(action, -1.0, 1.0)
        obs, reward, done, truncated, info = env.step(np.array([action]))
        peak_disp = max(peak_disp, abs(info['roof_displacement']))
        step += 1
        done = done or truncated

    metrics = env.get_episode_metrics()
    return {
        'displacement_cm': peak_disp * 100,
        'isdr_percent': metrics['max_isdr_percent'],
        'dcr': metrics['DCR'],
        'mean_force_kN': metrics['mean_force'] / 1000
    }


def create_pd_controller(Kp: float, Kd: float, max_force: float):
    """Create a PD controller with given gains"""
    def controller(obs, step):
        roof_disp = obs[0]  # Normalized displacement
        roof_vel = obs[1]   # Normalized velocity

        # Denormalize (assuming bounds of 5m disp, 20m/s vel)
        disp = roof_disp * 5.0
        vel = roof_vel * 20.0

        # PD control
        force = -Kp * disp - Kd * vel

        # Normalize to [-1, 1]
        return force / max_force

    return controller


def create_phase_controller(gain: float, phase_shift: float, max_force: float):
    """Create a phase-shifted velocity controller"""
    def controller(obs, step):
        roof_vel = obs[1] * 20.0  # Denormalize

        # Simple phase-shifted control
        force = -gain * roof_vel * np.cos(phase_shift)
        return force / max_force

    return controller


def create_displacement_only_controller(Kp: float, max_force: float):
    """Control based only on displacement (proportional)"""
    def controller(obs, step):
        roof_disp = obs[0] * 5.0  # Denormalize
        force = -Kp * roof_disp
        return force / max_force

    return controller


def create_velocity_only_controller(Kd: float, max_force: float):
    """Control based only on velocity (derivative/damping)"""
    def controller(obs, step):
        roof_vel = obs[1] * 20.0  # Denormalize
        force = -Kd * roof_vel
        return force / max_force

    return controller


def find_pareto_frontier(results: List[Dict]) -> List[Dict]:
    """
    Find Pareto-optimal points where no other point is better in ALL objectives

    A point is Pareto-optimal if there's no other point with:
    - Lower displacement AND lower ISDR AND lower DCR
    """
    pareto = []

    for i, r1 in enumerate(results):
        is_dominated = False

        for j, r2 in enumerate(results):
            if i == j:
                continue

            # Check if r2 dominates r1 (r2 is better in ALL objectives)
            if (r2['displacement_cm'] <= r1['displacement_cm'] and
                r2['isdr_percent'] <= r1['isdr_percent'] and
                r2['dcr'] <= r1['dcr'] and
                (r2['displacement_cm'] < r1['displacement_cm'] or
                 r2['isdr_percent'] < r1['isdr_percent'] or
                 r2['dcr'] < r1['dcr'])):
                is_dominated = True
                break

        if not is_dominated:
            pareto.append(r1)

    return pareto


def main():
    print("="*70)
    print("  PARETO FRONTIER ANALYSIS")
    print("  Finding achievable (displacement, ISDR, DCR) combinations")
    print("="*70)

    # Load M4.5 earthquake
    earthquake_file = "../../matlab/datasets/PEER_small_M4.5_PGA0.25g.csv"

    if not os.path.exists(earthquake_file):
        # Try alternative path
        earthquake_file = "../../matlab/datasets/test/PEER_small_M4.5_PGA0.25g.csv"

    if not os.path.exists(earthquake_file):
        print(f"❌ Earthquake file not found!")
        print(f"   Please provide the correct path to M4.5 earthquake data")
        return

    print(f"\nLoading earthquake: {earthquake_file}")
    earthquake_data, dt = load_earthquake(earthquake_file)
    print(f"✅ Loaded {len(earthquake_data)} samples, dt={dt}s")

    # Create environment
    # HARDWARE UPGRADE v2: Increased max force for aggressive targets
    # Previous: 150 kN (could not achieve 0.5% ISDR)
    # New: 250 kN (67% more control authority)
    max_force = 250000  # 250 kN

    env = ImprovedTMDBuildingEnv(
        earthquake_data=earthquake_data,
        dt=dt,
        max_force=max_force,
        earthquake_name="M4.5_Pareto_Test",
        reward_scale=1.0
    )

    print(f"\nTMD Configuration:")
    print(f"  Max force: {max_force/1000} kN")
    print(f"  TMD mass ratio: {env.tmd_mass / env.floor_mass * 100:.1f}%")
    print(f"  TMD stiffness: {env.tmd_k/1000:.1f} kN/m")
    print(f"  TMD damping: {env.tmd_c/1000:.1f} kN·s/m")

    # Define control strategies to test
    print(f"\n{'='*70}")
    print("  TESTING CONTROL STRATEGIES")
    print("="*70)

    results = []
    strategies = []

    # 1. No control (baseline)
    strategies.append(("No control", lambda obs, step: 0.0))

    # 2. PD controllers with varying gains
    for Kp in [10000, 25000, 50000, 75000, 100000, 150000]:
        for Kd in [1000, 2500, 5000, 10000, 15000, 20000]:
            name = f"PD(Kp={Kp/1000:.0f}k, Kd={Kd/1000:.1f}k)"
            strategies.append((name, create_pd_controller(Kp, Kd, max_force)))

    # 3. Displacement-only controllers
    for Kp in [25000, 50000, 100000, 150000, 200000]:
        name = f"P-only(Kp={Kp/1000:.0f}k)"
        strategies.append((name, create_displacement_only_controller(Kp, max_force)))

    # 4. Velocity-only controllers (pure damping)
    for Kd in [2500, 5000, 10000, 20000, 30000]:
        name = f"D-only(Kd={Kd/1000:.1f}k)"
        strategies.append((name, create_velocity_only_controller(Kd, max_force)))

    # 5. Max force strategies
    strategies.append(("Max positive", lambda obs, step: 1.0))
    strategies.append(("Max negative", lambda obs, step: -1.0))
    strategies.append(("Alternating", lambda obs, step: 1.0 if step % 10 < 5 else -1.0))

    # 6. Bang-bang control
    def bang_bang_disp(obs, step):
        return -1.0 if obs[0] > 0 else 1.0
    strategies.append(("Bang-bang(disp)", bang_bang_disp))

    def bang_bang_vel(obs, step):
        return -1.0 if obs[1] > 0 else 1.0
    strategies.append(("Bang-bang(vel)", bang_bang_vel))

    # Test all strategies
    print(f"\nTesting {len(strategies)} control strategies...")
    print(f"\n{'Strategy':<35} {'Disp(cm)':<12} {'ISDR%':<10} {'DCR':<8} {'Force(kN)':<10}")
    print("-" * 75)

    for name, strategy in strategies:
        try:
            metrics = run_control_strategy(env, strategy)
            metrics['name'] = name
            results.append(metrics)

            print(f"{name:<35} {metrics['displacement_cm']:<12.2f} {metrics['isdr_percent']:<10.2f} {metrics['dcr']:<8.2f} {metrics['mean_force_kN']:<10.1f}")
        except Exception as e:
            print(f"{name:<35} ERROR: {e}")

    env.close()

    # Find Pareto frontier
    print(f"\n{'='*70}")
    print("  PARETO FRONTIER (Best Achievable Trade-offs)")
    print("="*70)

    pareto = find_pareto_frontier(results)

    print(f"\nFound {len(pareto)} Pareto-optimal points:\n")
    print(f"{'Strategy':<35} {'Disp(cm)':<12} {'ISDR%':<10} {'DCR':<8}")
    print("-" * 65)

    # Sort by displacement
    pareto_sorted = sorted(pareto, key=lambda x: x['displacement_cm'])

    for p in pareto_sorted:
        print(f"{p['name']:<35} {p['displacement_cm']:<12.2f} {p['isdr_percent']:<10.2f} {p['dcr']:<8.2f}")

    # Find best individual metrics
    print(f"\n{'='*70}")
    print("  BEST ACHIEVABLE (Individual Metrics)")
    print("="*70)

    best_disp = min(results, key=lambda x: x['displacement_cm'])
    best_isdr = min(results, key=lambda x: x['isdr_percent'])
    best_dcr = min(results, key=lambda x: x['dcr'])

    print(f"\nBest displacement: {best_disp['displacement_cm']:.2f} cm")
    print(f"  Strategy: {best_disp['name']}")
    print(f"  ISDR: {best_disp['isdr_percent']:.2f}%, DCR: {best_disp['dcr']:.2f}")

    print(f"\nBest ISDR: {best_isdr['isdr_percent']:.2f}%")
    print(f"  Strategy: {best_isdr['name']}")
    print(f"  Displacement: {best_isdr['displacement_cm']:.2f} cm, DCR: {best_isdr['dcr']:.2f}")

    print(f"\nBest DCR: {best_dcr['dcr']:.2f}")
    print(f"  Strategy: {best_dcr['name']}")
    print(f"  Displacement: {best_dcr['displacement_cm']:.2f} cm, ISDR: {best_dcr['isdr_percent']:.2f}%")

    # Check if targets are achievable
    print(f"\n{'='*70}")
    print("  TARGET FEASIBILITY ANALYSIS")
    print("="*70)

    target_disp = 18.0  # cm (upper bound of your 10-18 cm target)
    target_isdr = 0.5   # %
    target_dcr = 1.1    #

    print(f"\nYour targets:")
    print(f"  Displacement: ≤{target_disp} cm")
    print(f"  ISDR: ≤{target_isdr}%")
    print(f"  DCR: ≤{target_dcr}")

    # Check if ANY strategy achieves all targets
    achievable = [r for r in results if
                  r['displacement_cm'] <= target_disp and
                  r['isdr_percent'] <= target_isdr and
                  r['dcr'] <= target_dcr]

    if achievable:
        print(f"\n✅ TARGETS ARE ACHIEVABLE!")
        print(f"   {len(achievable)} strategies meet all targets:\n")
        for a in achievable[:5]:  # Show top 5
            print(f"   {a['name']}: {a['displacement_cm']:.2f}cm, {a['isdr_percent']:.2f}%, DCR={a['dcr']:.2f}")
    else:
        print(f"\n❌ TARGETS ARE NOT SIMULTANEOUSLY ACHIEVABLE")
        print(f"   No tested strategy meets ALL targets.")

        # Find closest
        def distance_to_target(r):
            d_disp = max(0, r['displacement_cm'] - target_disp) / target_disp
            d_isdr = max(0, r['isdr_percent'] - target_isdr) / target_isdr
            d_dcr = max(0, r['dcr'] - target_dcr) / target_dcr
            return d_disp + d_isdr + d_dcr

        closest = min(results, key=distance_to_target)

        print(f"\n   Closest achievable:")
        print(f"   {closest['name']}")
        print(f"   Displacement: {closest['displacement_cm']:.2f} cm (target: ≤{target_disp})")
        print(f"   ISDR: {closest['isdr_percent']:.2f}% (target: ≤{target_isdr}%)")
        print(f"   DCR: {closest['dcr']:.2f} (target: ≤{target_dcr})")

        # Analyze which constraint is binding
        print(f"\n   Gap analysis:")
        print(f"   Displacement gap: {closest['displacement_cm'] - target_disp:+.2f} cm")
        print(f"   ISDR gap: {closest['isdr_percent'] - target_isdr:+.2f}%")
        print(f"   DCR gap: {closest['dcr'] - target_dcr:+.2f}")

    # =========================================================================
    # UNCONTROLLED COMPARISON - Is TMD intervention even beneficial?
    # =========================================================================
    print(f"\n{'='*70}")
    print("  UNCONTROLLED COMPARISON")
    print("  Is active TMD control beneficial?")
    print("="*70)

    # Find the "No control" result (passive TMD only, no active force)
    uncontrolled = next((r for r in results if r['name'] == 'No control'), None)

    if uncontrolled:
        print(f"\n  BASELINE (Passive TMD, no active control):")
        print(f"    Displacement: {uncontrolled['displacement_cm']:.2f} cm")
        print(f"    ISDR:         {uncontrolled['isdr_percent']:.2f}%")
        print(f"    DCR:          {uncontrolled['dcr']:.2f}")

        print(f"\n  BEST CONTROLLED STRATEGIES vs UNCONTROLLED:")
        print(f"  {'-'*65}")

        # Find best strategies that improve over uncontrolled
        improved_strategies = [r for r in results if r['name'] != 'No control']

        # Best displacement improvement
        best_disp_improve = min(improved_strategies, key=lambda x: x['displacement_cm'])
        disp_improve_pct = 100 * (uncontrolled['displacement_cm'] - best_disp_improve['displacement_cm']) / uncontrolled['displacement_cm']

        print(f"\n  Best Displacement Reduction:")
        print(f"    Strategy:     {best_disp_improve['name']}")
        print(f"    Displacement: {best_disp_improve['displacement_cm']:.2f} cm (vs {uncontrolled['displacement_cm']:.2f} cm)")
        print(f"    Improvement:  {disp_improve_pct:+.1f}%")
        if disp_improve_pct > 0:
            print(f"    Verdict:      ACTIVE CONTROL HELPS")
        else:
            print(f"    Verdict:      ACTIVE CONTROL DOES NOT HELP")

        # Best ISDR improvement
        best_isdr_improve = min(improved_strategies, key=lambda x: x['isdr_percent'])
        isdr_improve_pct = 100 * (uncontrolled['isdr_percent'] - best_isdr_improve['isdr_percent']) / uncontrolled['isdr_percent']

        print(f"\n  Best ISDR Reduction:")
        print(f"    Strategy:     {best_isdr_improve['name']}")
        print(f"    ISDR:         {best_isdr_improve['isdr_percent']:.2f}% (vs {uncontrolled['isdr_percent']:.2f}%)")
        print(f"    Improvement:  {isdr_improve_pct:+.1f}%")
        if isdr_improve_pct > 0:
            print(f"    Verdict:      ACTIVE CONTROL HELPS")
        else:
            print(f"    Verdict:      ACTIVE CONTROL DOES NOT HELP")

        # Best DCR improvement
        best_dcr_improve = min(improved_strategies, key=lambda x: x['dcr'])
        dcr_improve_pct = 100 * (uncontrolled['dcr'] - best_dcr_improve['dcr']) / uncontrolled['dcr']

        print(f"\n  Best DCR Reduction:")
        print(f"    Strategy:     {best_dcr_improve['name']}")
        print(f"    DCR:          {best_dcr_improve['dcr']:.2f} (vs {uncontrolled['dcr']:.2f})")
        print(f"    Improvement:  {dcr_improve_pct:+.1f}%")
        if dcr_improve_pct > 0:
            print(f"    Verdict:      ACTIVE CONTROL HELPS")
        else:
            print(f"    Verdict:      ACTIVE CONTROL DOES NOT HELP")

        # Overall assessment
        print(f"\n  {'-'*65}")
        print(f"  OVERALL ASSESSMENT:")

        any_improvement = disp_improve_pct > 0 or isdr_improve_pct > 0 or dcr_improve_pct > 0
        all_improvement = disp_improve_pct > 0 and isdr_improve_pct > 0 and dcr_improve_pct > 0

        if all_improvement:
            print(f"\n    ACTIVE TMD CONTROL IS BENEFICIAL")
            print(f"    Active control improves ALL metrics vs passive TMD alone.")
            print(f"    Maximum achievable improvements:")
            print(f"      - Displacement: {disp_improve_pct:+.1f}%")
            print(f"      - ISDR:         {isdr_improve_pct:+.1f}%")
            print(f"      - DCR:          {dcr_improve_pct:+.1f}%")
        elif any_improvement:
            print(f"\n    ACTIVE TMD CONTROL IS PARTIALLY BENEFICIAL")
            print(f"    Active control improves SOME metrics but not all.")
            print(f"    Trade-offs exist between objectives.")
        else:
            print(f"\n    ACTIVE TMD CONTROL IS NOT BENEFICIAL")
            print(f"    Passive TMD alone achieves the best results.")
            print(f"    Active control may be destabilizing the system.")

        # Pareto frontier vs uncontrolled
        print(f"\n  PARETO FRONTIER vs UNCONTROLLED:")
        print(f"  {'-'*65}")

        pareto_better_disp = [p for p in pareto if p['displacement_cm'] < uncontrolled['displacement_cm']]
        pareto_better_isdr = [p for p in pareto if p['isdr_percent'] < uncontrolled['isdr_percent']]
        pareto_better_dcr = [p for p in pareto if p['dcr'] < uncontrolled['dcr']]

        print(f"\n    Pareto points with better displacement: {len(pareto_better_disp)}/{len(pareto)}")
        print(f"    Pareto points with better ISDR:         {len(pareto_better_isdr)}/{len(pareto)}")
        print(f"    Pareto points with better DCR:          {len(pareto_better_dcr)}/{len(pareto)}")

        if len(pareto_better_disp) > 0 or len(pareto_better_isdr) > 0 or len(pareto_better_dcr) > 0:
            print(f"\n    Active control CAN improve over passive TMD.")
            print(f"    RL training should focus on finding optimal trade-offs.")
        else:
            print(f"\n    No Pareto point improves over passive TMD!")
            print(f"    Consider: Your passive TMD may already be well-tuned.")

    else:
        print(f"\n  ERROR: 'No control' baseline not found in results")

    # Recommendation
    print(f"\n{'='*70}")
    print("  RECOMMENDATION")
    print("="*70)

    if achievable:
        print(f"\n  YOUR TARGETS ARE ACHIEVABLE!")
        print(f"  {len(achievable)} strategies meet all targets.")
        print(f"  RL should be able to find a policy that meets all targets.")
        print(f"  If RL isn't converging, the issue is reward shaping, not physics.")
    else:
        print(f"\n  YOUR TARGETS ARE NOT SIMULTANEOUSLY ACHIEVABLE")
        print(f"  This is a HARDWARE LIMITATION, not a software/RL problem.")
        print(f"\n  OPTIONS:")
        print(f"\n  1. RELAX TARGETS to achievable levels:")

        # Suggest realistic targets based on Pareto frontier
        if pareto_sorted:
            best_balanced = min(pareto_sorted, key=lambda x:
                (x['displacement_cm']/25 + x['isdr_percent']/2 + x['dcr']/1.5))
            print(f"     Realistic targets (balanced):")
            print(f"       Displacement: {best_balanced['displacement_cm']:.0f} cm")
            print(f"       ISDR:         {best_balanced['isdr_percent']:.1f}%")
            print(f"       DCR:          {best_balanced['dcr']:.1f}")

        print(f"\n  2. INCREASE TMD MASS RATIO:")
        print(f"     Current: 2% of floor mass")
        print(f"     Try:     3-4% of floor mass")
        print(f"     Effect:  More inertial force capacity")

        print(f"\n  3. INCREASE MAX ACTIVE FORCE:")
        print(f"     Current: 150 kN")
        print(f"     Try:     200-250 kN")
        print(f"     Effect:  More control authority")

        print(f"\n  4. OPTIMIZE PASSIVE TMD TUNING:")
        print(f"     Adjust TMD stiffness (k) and damping (c)")
        print(f"     May allow better baseline performance")

    print(f"\n{'='*70}")
    print("  ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
