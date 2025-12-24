"""
TEST AND EVALUATE RL MODEL
==========================

Test trained RL model and compare against passive/PD/fuzzy baselines

Author: Siddharth
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from tmd_environment import make_improved_tmd_env
import os


def test_rl_model(
    model_path,
    earthquake_file,
    render=False,
    verbose=True
):
    """
    Test RL model on a single earthquake
    
    Args:
        model_path: Path to saved .zip model
        earthquake_file: Path to earthquake CSV
        render: Whether to plot results
        verbose: Print detailed output
        
    Returns:
        dict with performance metrics
    """
    
    # Load model
    if verbose:
        print(f"Loading model from {model_path}...")
    model = SAC.load(model_path)
    
    # Create environment
    env = make_improved_tmd_env(earthquake_file)
    
    # Run episode
    obs, info = env.reset()
    done = False
    truncated = False
    
    # Storage
    displacements = []
    velocities = []
    forces = []
    times = []
    rewards = []
    
    total_reward = 0
    timestep = 0
    
    while not (done or truncated):
        # Get action from policy
        action, _states = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        
        # Record
        displacements.append(obs[0] * 100)  # Convert to cm
        velocities.append(obs[1])
        forces.append(info['control_force'] / 1000)  # Convert to kN
        times.append(timestep * env.dt)
        rewards.append(reward)
        total_reward += reward
        
        timestep += 1
    
    # Calculate metrics
    peak_displacement = max(abs(np.array(displacements)))
    rms_displacement = np.sqrt(np.mean(np.array(displacements)**2))
    mean_force = np.mean(np.abs(forces))
    max_force = max(abs(np.array(forces)))
    
    results = {
        'peak_displacement_cm': peak_displacement,
        'rms_displacement_cm': rms_displacement,
        'mean_force_kN': mean_force,
        'max_force_kN': max_force,
        'total_reward': total_reward,
        'timesteps': timestep,
        'times': np.array(times),
        'displacements': np.array(displacements),
        'velocities': np.array(velocities),
        'forces': np.array(forces),
        'rewards': np.array(rewards)
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  RESULTS")
        print(f"{'='*60}")
        print(f"Peak displacement: {peak_displacement:.2f} cm")
        print(f"RMS displacement:  {rms_displacement:.2f} cm")
        print(f"Mean |force|:      {mean_force:.2f} kN")
        print(f"Max |force|:       {max_force:.2f} kN")
        print(f"Total reward:      {total_reward:.2f}")
        print(f"{'='*60}\n")
    
    if render:
        plot_episode_results(results, earthquake_file)
    
    return results


def plot_episode_results(results, earthquake_name=""):
    """Plot time series results from episode"""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    times = results['times']
    
    # Displacement
    axes[0].plot(times, results['displacements'], 'b-', linewidth=1.5)
    axes[0].axhline(y=results['peak_displacement_cm'], color='r', 
                   linestyle='--', alpha=0.5, label=f'Peak: {results["peak_displacement_cm"]:.2f} cm')
    axes[0].axhline(y=-results['peak_displacement_cm'], color='r', 
                   linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Roof Displacement (cm)', fontsize=11)
    axes[0].set_title(f'RL TMD Controller Performance - {earthquake_name}', 
                     fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Velocity
    axes[1].plot(times, results['velocities'], 'g-', linewidth=1.5)
    axes[1].set_ylabel('Roof Velocity (m/s)', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # Control force
    axes[2].plot(times, results['forces'], 'r-', linewidth=1.5)
    axes[2].axhline(y=100, color='k', linestyle='--', alpha=0.3, label='Force limit (±100 kN)')
    axes[2].axhline(y=-100, color='k', linestyle='--', alpha=0.3)
    axes[2].set_ylabel('Control Force (kN)', fontsize=11)
    axes[2].set_xlabel('Time (s)', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig


def compare_controllers(
    rl_model_path,
    earthquake_file,
    passive_peak=None,
    fuzzy_peak=None,
    pd_peak=None
):
    """
    Compare RL against other controllers
    
    Args:
        rl_model_path: Path to RL model
        earthquake_file: Earthquake to test on
        passive_peak: Peak displacement from passive TMD (cm)
        fuzzy_peak: Peak displacement from fuzzy controller (cm)
        pd_peak: Peak displacement from PD controller (cm)
    """
    
    # Test RL
    results = test_rl_model(rl_model_path, earthquake_file, verbose=True)
    rl_peak = results['peak_displacement_cm']
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    controllers = []
    peaks = []
    colors = []
    
    if passive_peak is not None:
        controllers.append('Passive\nTMD')
        peaks.append(passive_peak)
        colors.append('gray')
    
    if pd_peak is not None:
        controllers.append('PD\nControl')
        peaks.append(pd_peak)
        colors.append('steelblue')
    
    if fuzzy_peak is not None:
        controllers.append('Fuzzy\nLogic')
        peaks.append(fuzzy_peak)
        colors.append('orange')
    
    controllers.append('RL\n(SAC)')
    peaks.append(rl_peak)
    colors.append('green')
    
    # Bar chart
    bars = ax.bar(range(len(controllers)), peaks, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, peak) in enumerate(zip(bars, peaks)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{peak:.2f} cm',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Calculate improvements
    if passive_peak is not None:
        baseline = passive_peak
        for i, (name, peak) in enumerate(zip(controllers, peaks)):
            if name != 'Passive\nTMD':
                improvement = (baseline - peak) / baseline * 100
                ax.text(i, peak/2, f'{improvement:+.1f}%',
                       ha='center', va='center', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_ylabel('Peak Roof Displacement (cm)', fontsize=12)
    ax.set_xlabel('Control Strategy', fontsize=12)
    ax.set_title('TMD Controller Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(controllers)))
    ax.set_xticklabels(controllers)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("  CONTROLLER COMPARISON SUMMARY")
    print("="*60)
    for name, peak in zip(controllers, peaks):
        if passive_peak is not None and name != 'Passive\nTMD':
            improvement = (passive_peak - peak) / passive_peak * 100
            print(f"{name.replace(chr(10), ' '):15s}: {peak:6.2f} cm ({improvement:+.1f}% vs passive)")
        else:
            print(f"{name.replace(chr(10), ' '):15s}: {peak:6.2f} cm")
    print("="*60 + "\n")


def batch_evaluate(
    rl_model_path,
    earthquake_files,
    save_results=True
):
    """
    Evaluate RL model on multiple earthquakes
    
    Args:
        rl_model_path: Path to RL model
        earthquake_files: List of earthquake CSV files
        save_results: Save results to file
        
    Returns:
        DataFrame with results
    """
    
    print(f"\n{'='*60}")
    print(f"  BATCH EVALUATION - {len(earthquake_files)} EARTHQUAKES")
    print(f"{'='*60}\n")
    
    results = []
    
    for i, eq_file in enumerate(earthquake_files):
        print(f"[{i+1}/{len(earthquake_files)}] Testing {os.path.basename(eq_file)}...")
        
        result = test_rl_model(rl_model_path, eq_file, verbose=False)
        result['earthquake'] = os.path.basename(eq_file)
        results.append(result)
        
        print(f"   Peak: {result['peak_displacement_cm']:.2f} cm, "
              f"RMS: {result['rms_displacement_cm']:.2f} cm")
    
    # Summary statistics
    peaks = [r['peak_displacement_cm'] for r in results]
    rms_vals = [r['rms_displacement_cm'] for r in results]
    
    print(f"\n{'='*60}")
    print(f"  SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Peak displacement: {np.mean(peaks):.2f} ± {np.std(peaks):.2f} cm")
    print(f"RMS displacement:  {np.mean(rms_vals):.2f} ± {np.std(rms_vals):.2f} cm")
    print(f"{'='*60}\n")
    
    if save_results:
        import pandas as pd
        df = pd.DataFrame([{
            'earthquake': r['earthquake'],
            'peak_disp_cm': r['peak_displacement_cm'],
            'rms_disp_cm': r['rms_displacement_cm'],
            'mean_force_kN': r['mean_force_kN'],
            'max_force_kN': r['max_force_kN']
        } for r in results])
        
        df.to_csv('rl_evaluation_results.csv', index=False)
        print(f"✅ Results saved to rl_evaluation_results.csv\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test RL TMD controller')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.zip file)')
    parser.add_argument('--earthquake', type=str,
                       help='Single earthquake file to test')
    parser.add_argument('--batch', nargs='+',
                       help='Multiple earthquake files for batch evaluation')
    parser.add_argument('--compare', action='store_true',
                       help='Compare against other controllers')
    parser.add_argument('--passive-peak', type=float,
                       help='Passive TMD peak displacement (cm) for comparison')
    parser.add_argument('--fuzzy-peak', type=float,
                       help='Fuzzy controller peak displacement (cm)')
    parser.add_argument('--pd-peak', type=float,
                       help='PD controller peak displacement (cm)')
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch evaluation
        batch_evaluate(args.model, args.batch)
    elif args.earthquake:
        # Single test
        if args.compare:
            compare_controllers(
                args.model,
                args.earthquake,
                passive_peak=args.passive_peak,
                fuzzy_peak=args.fuzzy_peak,
                pd_peak=args.pd_peak
            )
        else:
            test_rl_model(args.model, args.earthquake, render=True)
    else:
        print("❌ Error: Specify either --earthquake or --batch")
        print("\nExamples:")
        print("  python test_rl_model.py --model rl_models/tmd_sac_final.zip --earthquake datasets/TEST3*.csv")
        print("  python test_rl_model.py --model rl_models/tmd_sac_final.zip --batch datasets/TEST*.csv")
