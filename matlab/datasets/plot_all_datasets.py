"""
PLOT ALL EARTHQUAKE DATASETS - TIME vs ACCELERATION
====================================================

Visualizes all training and test datasets to verify baseline correction.
Creates separate plots for each magnitude class showing all variants.

Usage:
    python plot_all_datasets.py

Author: Siddharth
Date: December 2025
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import glob
import os

def plot_earthquake_datasets():
    """
    Plot all earthquake datasets grouped by magnitude
    """

    # Define magnitude groups
    magnitude_groups = {
        'M4.5 (PGA 0.25g)': {
            'test': 'PEER_small_M4.5_PGA0.25g.csv',
            'train': glob.glob('training_set/TRAIN_M4.5*.csv')
        },
        'M5.7 (PGA 0.35g)': {
            'test': 'PEER_moderate_M5.7_PGA0.35g.csv',
            'train': glob.glob('training_set/TRAIN_M5.7*.csv')
        },
        'M7.4 (PGA 0.75g)': {
            'test': 'PEER_high_M7.4_PGA0.75g.csv',
            'train': glob.glob('training_set/TRAIN_M7.4*.csv')
        },
        'M8.4 (PGA 0.9g)': {
            'test': 'PEER_insane_M8.4_PGA0.9g.csv',
            'train': glob.glob('training_set/TRAIN_M8.4*.csv')
        }
    }

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (magnitude, files) in enumerate(magnitude_groups.items()):
        ax = axes[idx]

        # Plot test file
        if os.path.exists(files['test']):
            data = np.loadtxt(files['test'], delimiter=',', skiprows=1)
            times = data[:, 0]
            accel = data[:, 1]
            ax.plot(times, accel, 'r-', linewidth=2, label='PEER Test', alpha=0.8)

            # Mark final value
            ax.plot(times[-1], accel[-1], 'ro', markersize=8,
                   label=f'Final: {accel[-1]:.2e} m/s²')

        # Plot training files
        for i, train_file in enumerate(sorted(files['train']), 1):
            if os.path.exists(train_file):
                data = np.loadtxt(train_file, delimiter=',', skiprows=1)
                times = data[:, 0]
                accel = data[:, 1]
                ax.plot(times, accel, linewidth=1.5, alpha=0.6,
                       label=f'Train Var {i}')

                # Mark final value for first variant
                if i == 1:
                    ax.plot(times[-1], accel[-1], 'o', markersize=6,
                           label=f'Train Final: {accel[-1]:.2e} m/s²')

        # Formatting
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Acceleration (m/s²)', fontsize=11)
        ax.set_title(magnitude, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)

        # Add zero line
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

        # Highlight last 5 seconds (taper region)
        if os.path.exists(files['test']):
            data = np.loadtxt(files['test'], delimiter=',', skiprows=1)
            times = data[:, 0]
            taper_start = max(0, times[-1] - 5.0)
            ax.axvspan(taper_start, times[-1], alpha=0.1, color='yellow',
                      label='Taper Region (5s)')

    plt.suptitle('Earthquake Datasets - Baseline Corrected (V2)\n' +
                 'Final Acceleration = 0.0 for all datasets',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save figure
    output_file = 'all_datasets_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved visualization to: {output_file}")
    plt.close()


def plot_zoomed_endings():
    """
    Create zoomed plots showing the last 10 seconds to verify tapering
    """

    # All files
    test_files = [
        'PEER_small_M4.5_PGA0.25g.csv',
        'PEER_moderate_M5.7_PGA0.35g.csv',
        'PEER_high_M7.4_PGA0.75g.csv',
        'PEER_insane_M8.4_PGA0.9g.csv'
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, test_file in enumerate(test_files):
        ax = axes[idx]

        if not os.path.exists(test_file):
            continue

        # Load data
        data = np.loadtxt(test_file, delimiter=',', skiprows=1)
        times = data[:, 0]
        accel = data[:, 1]

        # Get last 10 seconds
        last_10s_idx = np.where(times >= (times[-1] - 10.0))[0]
        times_zoomed = times[last_10s_idx]
        accel_zoomed = accel[last_10s_idx]

        # Plot
        ax.plot(times_zoomed, accel_zoomed, 'b-', linewidth=2)

        # Mark taper region (last 5 seconds)
        taper_start = times[-1] - 5.0
        ax.axvspan(taper_start, times[-1], alpha=0.2, color='yellow',
                  label='Cosine Taper (5s)')

        # Mark final value
        ax.plot(times[-1], accel[-1], 'ro', markersize=10,
               label=f'Final: {accel[-1]:.2e} m/s²')

        # Zero line
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)

        # Formatting
        magnitude = test_file.split('_')[1] if '_' in test_file else 'Unknown'
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Acceleration (m/s²)', fontsize=11)
        ax.set_title(f'{test_file}\nLast 10 Seconds (Showing Taper to Zero)',
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)

        # Print final stats
        print(f"\n{test_file}:")
        print(f"  Duration: {times[-1]:.1f}s")
        print(f"  Final accel: {accel[-1]:.10f} m/s²")
        print(f"  PGA: {np.max(np.abs(accel)):.4f} m/s² ({np.max(np.abs(accel))/9.81:.3f}g)")

    plt.suptitle('Earthquake Dataset Endings - Verifying Zero Final Acceleration',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save
    output_file = 'dataset_endings_zoomed.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved zoomed endings to: {output_file}")
    plt.close()


def plot_comparison_before_after():
    """
    Compare V1 (broken) vs V2 (fixed) for PEER M7.4
    """

    v1_file = 'PEER_high_M7.4_PGA0.75g.csv.v1'
    v2_file = 'PEER_high_M7.4_PGA0.75g.csv'

    if not os.path.exists(v1_file):
        print(f"\n⚠️  V1 backup not found: {v1_file}")
        print("   Skipping before/after comparison")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Load both versions
    data_v1 = np.loadtxt(v1_file, delimiter=',', skiprows=1)
    times_v1 = data_v1[:, 0]
    accel_v1 = data_v1[:, 1]

    data_v2 = np.loadtxt(v2_file, delimiter=',', skiprows=1)
    times_v2 = data_v2[:, 0]
    accel_v2 = data_v2[:, 1]

    # Plot 1: Full acceleration comparison
    axes[0, 0].plot(times_v1, accel_v1, 'r-', linewidth=1.5, alpha=0.7, label='V1 (Broken)')
    axes[0, 0].plot(times_v2, accel_v2, 'g-', linewidth=1.5, alpha=0.7, label='V2 (Fixed)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Acceleration (m/s²)')
    axes[0, 0].set_title('Full Acceleration Time History')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Plot 2: Zoomed ending
    last_10s_idx_v1 = np.where(times_v1 >= (times_v1[-1] - 10.0))[0]
    last_10s_idx_v2 = np.where(times_v2 >= (times_v2[-1] - 10.0))[0]

    axes[0, 1].plot(times_v1[last_10s_idx_v1], accel_v1[last_10s_idx_v1],
                    'r-', linewidth=2, label='V1 (Broken)')
    axes[0, 1].plot(times_v2[last_10s_idx_v2], accel_v2[last_10s_idx_v2],
                    'g-', linewidth=2, label='V2 (Fixed)')
    axes[0, 1].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    axes[0, 1].plot(times_v1[-1], accel_v1[-1], 'ro', markersize=10,
                    label=f'V1 Final: {accel_v1[-1]:.4f}')
    axes[0, 1].plot(times_v2[-1], accel_v2[-1], 'go', markersize=10,
                    label=f'V2 Final: {accel_v2[-1]:.10f}')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Acceleration (m/s²)')
    axes[0, 1].set_title('Last 10 Seconds (Showing Fix)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Plot 3: Velocity comparison
    dt = np.mean(np.diff(times_v1))
    vel_v1 = np.cumsum(accel_v1) * dt
    vel_v2 = np.cumsum(accel_v2) * dt

    axes[1, 0].plot(times_v1, vel_v1, 'r-', linewidth=1.5, alpha=0.7, label='V1 (Broken)')
    axes[1, 0].plot(times_v2, vel_v2, 'g-', linewidth=1.5, alpha=0.7, label='V2 (Fixed)')
    axes[1, 0].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 0].plot(times_v1[-1], vel_v1[-1], 'ro', markersize=8,
                    label=f'V1 Final Vel: {vel_v1[-1]:.4f} m/s')
    axes[1, 0].plot(times_v2[-1], vel_v2[-1], 'go', markersize=8,
                    label=f'V2 Final Vel: {vel_v2[-1]:.4f} m/s')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Velocity (m/s)')
    axes[1, 0].set_title('Ground Velocity (Integrated Acceleration)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Plot 4: Displacement comparison
    disp_v1 = np.cumsum(vel_v1) * dt
    disp_v2 = np.cumsum(vel_v2) * dt

    axes[1, 1].plot(times_v1, disp_v1, 'r-', linewidth=1.5, alpha=0.7, label='V1 (Broken)')
    axes[1, 1].plot(times_v2, disp_v2, 'g-', linewidth=1.5, alpha=0.7, label='V2 (Fixed)')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 1].plot(times_v1[-1], disp_v1[-1], 'ro', markersize=8,
                    label=f'V1 Drift: {disp_v1[-1]:.4f} m')
    axes[1, 1].plot(times_v2[-1], disp_v2[-1], 'go', markersize=8,
                    label=f'V2 Drift: {disp_v2[-1]:.4f} m')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Displacement (m)')
    axes[1, 1].set_title('Ground Displacement (Baseline Drift)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.suptitle('PEER M7.4: Baseline Correction V1 (Broken) vs V2 (Fixed)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save
    output_file = 'baseline_correction_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved comparison to: {output_file}")

    # Print summary
    print(f"\n{'='*70}")
    print("PEER M7.4 COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"\nV1 (Broken):")
    print(f"  Final acceleration: {accel_v1[-1]:.6f} m/s²")
    print(f"  Final velocity: {vel_v1[-1]:.6f} m/s")
    print(f"  Baseline drift: {disp_v1[-1]:.6f} m")
    print(f"\nV2 (Fixed):")
    print(f"  Final acceleration: {accel_v2[-1]:.10f} m/s²")
    print(f"  Final velocity: {vel_v2[-1]:.6f} m/s")
    print(f"  Baseline drift: {disp_v2[-1]:.6f} m")
    print(f"\n✅ Improvement:")
    print(f"  Final accel: {abs(accel_v1[-1] - accel_v2[-1]):.6f} m/s² reduction")
    print(f"  Drift reduction: {abs(disp_v1[-1]) - abs(disp_v2[-1]):.6f} m")
    plt.close()


def main():
    """
    Main function - create all visualizations
    """

    print("\n" + "="*70)
    print("EARTHQUAKE DATASET VISUALIZATION")
    print("="*70)
    print("\nGenerating plots...")

    # Change to datasets directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Create all plots
    print("\n1. Plotting all datasets grouped by magnitude...")
    plot_earthquake_datasets()

    print("\n2. Plotting zoomed endings to verify tapering...")
    plot_zoomed_endings()

    print("\n3. Plotting before/after comparison...")
    plot_comparison_before_after()

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  1. all_datasets_visualization.png")
    print("  2. dataset_endings_zoomed.png")
    print("  3. baseline_correction_comparison.png")
    print("\n")


if __name__ == "__main__":
    main()
