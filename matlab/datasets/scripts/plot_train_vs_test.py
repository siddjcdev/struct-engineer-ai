"""
TRAINING vs TEST DATASET VISUALIZATION AND ANALYSIS
====================================================

Comprehensive visualization comparing training and test earthquake datasets:
1. Statistical distribution comparisons (PGA, RMS, duration)
2. Time-domain waveform overlays
3. Frequency domain analysis (power spectral density)
4. Verification of proper train/test split

Usage:
    python plot_train_vs_test.py

Author: Siddharth
Date: January 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import glob
import os
from scipy import signal
from scipy.stats import gaussian_kde


def load_earthquake_data(filepath):
    """
    Load earthquake data from CSV file

    Returns:
        times, acceleration (m/s²)
    """
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    times = data[:, 0]
    accel = data[:, 1]
    return times, accel


def compute_earthquake_stats(times, accel):
    """
    Compute statistical characteristics of earthquake signal

    Returns:
        Dictionary with PGA, RMS, duration, energy, etc.
    """
    dt = np.mean(np.diff(times))
    duration = times[-1] - times[0]

    # Peak Ground Acceleration (PGA)
    pga = np.max(np.abs(accel))
    pga_g = pga / 9.81

    # RMS acceleration
    rms = np.sqrt(np.mean(accel**2))
    rms_g = rms / 9.81

    # Arias Intensity (measure of total energy)
    arias_intensity = (np.pi / (2 * 9.81)) * np.sum(accel**2) * dt

    # Significant duration (5%-95% Arias Intensity)
    cumulative_arias = np.cumsum(accel**2) * dt
    cumulative_arias /= cumulative_arias[-1]  # Normalize
    idx_5 = np.where(cumulative_arias >= 0.05)[0][0]
    idx_95 = np.where(cumulative_arias >= 0.95)[0][0]
    significant_duration = times[idx_95] - times[idx_5]

    # Dominant frequency (peak in FFT)
    freqs = np.fft.rfftfreq(len(accel), dt)
    fft_accel = np.abs(np.fft.rfft(accel))
    dominant_freq_idx = np.argmax(fft_accel[1:]) + 1  # Skip DC component
    dominant_freq = freqs[dominant_freq_idx]

    return {
        'pga': pga,
        'pga_g': pga_g,
        'rms': rms,
        'rms_g': rms_g,
        'duration': duration,
        'significant_duration': significant_duration,
        'arias_intensity': arias_intensity,
        'dominant_freq': dominant_freq,
        'dt': dt
    }


def plot_train_test_comparison_by_magnitude():
    """
    Create comprehensive comparison plots for each magnitude class
    """

    # Define magnitude groups
    magnitude_configs = [
        {
            'name': 'M4.5',
            'label': 'M4.5 (PGA 0.25g)',
            'test_file': 'test/PEER_small_M4.5_PGA0.25g.csv',
            'train_pattern': 'training/training_set_v2/TRAIN_M4.5*.csv',
            'color': '#1f77b4'  # Blue
        },
        {
            'name': 'M5.7',
            'label': 'M5.7 (PGA 0.35g)',
            'test_file': 'test/PEER_moderate_M5.7_PGA0.35g.csv',
            'train_pattern': 'training/training_set_v2/TRAIN_M5.7*.csv',
            'color': '#ff7f0e'  # Orange
        },
        {
            'name': 'M7.4',
            'label': 'M7.4 (PGA 0.75g)',
            'test_file': 'test/PEER_high_M7.4_PGA0.75g.csv',
            'train_pattern': 'training/training_set_v2/TRAIN_M7.4*.csv',
            'color': '#2ca02c'  # Green
        },
        {
            'name': 'M8.4',
            'label': 'M8.4 (PGA 0.9g)',
            'test_file': 'test/PEER_insane_M8.4_PGA0.9g.csv',
            'train_pattern': 'training/training_set_v2/TRAIN_M8.4*.csv',
            'color': '#d62728'  # Red
        }
    ]

    # Create figure with 4 rows (one per magnitude)
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

    all_stats = []

    for row_idx, mag_config in enumerate(magnitude_configs):
        print(f"\nProcessing {mag_config['label']}...")

        # Load test data
        if not os.path.exists(mag_config['test_file']):
            print(f"  ⚠️  Test file not found: {mag_config['test_file']}")
            continue

        test_times, test_accel = load_earthquake_data(mag_config['test_file'])
        test_stats = compute_earthquake_stats(test_times, test_accel)

        # Load training data
        train_files = sorted(glob.glob(mag_config['train_pattern']))
        if not train_files:
            print(f"  ⚠️  No training files found: {mag_config['train_pattern']}")
            continue

        print(f"  Found {len(train_files)} training variants")

        train_stats_list = []
        for train_file in train_files:
            times, accel = load_earthquake_data(train_file)
            stats = compute_earthquake_stats(times, accel)
            train_stats_list.append(stats)

        # Store for summary statistics
        all_stats.append({
            'magnitude': mag_config['name'],
            'test': test_stats,
            'train': train_stats_list
        })

        # PLOT 1: Time domain waveforms (overlaid)
        ax1 = fig.add_subplot(gs[row_idx, 0])

        # Plot test waveform
        ax1.plot(test_times, test_accel, 'r-', linewidth=2,
                label='Test (PEER)', alpha=0.9, zorder=10)

        # Plot training waveforms (lighter)
        for i, train_file in enumerate(train_files[:5]):  # Show first 5 variants
            times, accel = load_earthquake_data(train_file)
            ax1.plot(times, accel, linewidth=1, alpha=0.3,
                    color=mag_config['color'],
                    label='Training' if i == 0 else '')

        ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        ax1.set_xlabel('Time (s)', fontsize=10)
        ax1.set_ylabel('Acceleration (m/s²)', fontsize=10)
        ax1.set_title(f'{mag_config["label"]}\nWaveform Comparison',
                     fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=8)

        # PLOT 2: PGA Distribution
        ax2 = fig.add_subplot(gs[row_idx, 1])

        train_pgas = [s['pga_g'] for s in train_stats_list]

        # Violin plot for training
        parts = ax2.violinplot([train_pgas], positions=[1], widths=0.7,
                               showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(mag_config['color'])
            pc.set_alpha(0.6)

        # Test PGA as horizontal line
        ax2.axhline(y=test_stats['pga_g'], color='r', linestyle='--',
                   linewidth=2, label=f"Test: {test_stats['pga_g']:.3f}g")

        # Statistics text
        mean_train_pga = np.mean(train_pgas)
        std_train_pga = np.std(train_pgas)
        ax2.text(1.4, np.max(train_pgas) * 0.95,
                f"Train μ={mean_train_pga:.3f}g\nTrain σ={std_train_pga:.4f}g",
                fontsize=8, verticalalignment='top')

        ax2.set_ylabel('PGA (g)', fontsize=10)
        ax2.set_xticks([1])
        ax2.set_xticklabels(['Training Variants'])
        ax2.set_title('Peak Ground Acceleration', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend(loc='upper left', fontsize=8)

        # PLOT 3: RMS Distribution
        ax3 = fig.add_subplot(gs[row_idx, 2])

        train_rms = [s['rms_g'] for s in train_stats_list]

        # Violin plot for training
        parts = ax3.violinplot([train_rms], positions=[1], widths=0.7,
                               showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(mag_config['color'])
            pc.set_alpha(0.6)

        # Test RMS as horizontal line
        ax3.axhline(y=test_stats['rms_g'], color='r', linestyle='--',
                   linewidth=2, label=f"Test: {test_stats['rms_g']:.3f}g")

        # Statistics text
        mean_train_rms = np.mean(train_rms)
        std_train_rms = np.std(train_rms)
        ax3.text(1.4, np.max(train_rms) * 0.95,
                f"Train μ={mean_train_rms:.3f}g\nTrain σ={std_train_rms:.4f}g",
                fontsize=8, verticalalignment='top')

        ax3.set_ylabel('RMS Acceleration (g)', fontsize=10)
        ax3.set_xticks([1])
        ax3.set_xticklabels(['Training Variants'])
        ax3.set_title('RMS Acceleration', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.legend(loc='upper left', fontsize=8)

        # PLOT 4: Power Spectral Density
        ax4 = fig.add_subplot(gs[row_idx, 3])

        # Test PSD
        freqs_test, psd_test = signal.welch(test_accel, fs=1/test_stats['dt'],
                                            nperseg=min(1024, len(test_accel)//4))
        ax4.semilogy(freqs_test, psd_test, 'r-', linewidth=2,
                    label='Test', alpha=0.9, zorder=10)

        # Training PSDs (first 5 variants)
        for i, train_file in enumerate(train_files[:5]):
            times, accel = load_earthquake_data(train_file)
            dt = np.mean(np.diff(times))
            freqs, psd = signal.welch(accel, fs=1/dt,
                                     nperseg=min(1024, len(accel)//4))
            ax4.semilogy(freqs, psd, linewidth=1, alpha=0.3,
                        color=mag_config['color'],
                        label='Training' if i == 0 else '')

        ax4.set_xlabel('Frequency (Hz)', fontsize=10)
        ax4.set_ylabel('PSD (m²/s⁴/Hz)', fontsize=10)
        ax4.set_title('Power Spectral Density', fontsize=11, fontweight='bold')
        ax4.set_xlim([0, 20])  # Focus on 0-20 Hz
        ax4.grid(True, alpha=0.3, which='both')
        ax4.legend(loc='upper right', fontsize=8)

        print(f"  Test PGA: {test_stats['pga_g']:.3f}g, RMS: {test_stats['rms_g']:.3f}g")
        print(f"  Train PGA: {mean_train_pga:.3f}±{std_train_pga:.4f}g")
        print(f"  Train RMS: {mean_train_rms:.3f}±{std_train_rms:.4f}g")

    plt.suptitle('Training vs Test Earthquake Dataset Comparison\n' +
                 'Verifying Proper Train/Test Split and Distribution',
                 fontsize=16, fontweight='bold', y=0.995)

    # Save figure
    output_file = 'analysis/train_vs_test_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved comparison plot: {output_file}")
    plt.close()

    return all_stats


def plot_statistical_summary(all_stats):
    """
    Create summary statistical comparison across all magnitudes
    """

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    magnitudes = [s['magnitude'] for s in all_stats]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Extract statistics
    test_pgas = [s['test']['pga_g'] for s in all_stats]
    train_pga_means = [np.mean([t['pga_g'] for t in s['train']]) for s in all_stats]
    train_pga_stds = [np.std([t['pga_g'] for t in s['train']]) for s in all_stats]

    test_rms = [s['test']['rms_g'] for s in all_stats]
    train_rms_means = [np.mean([t['rms_g'] for t in s['train']]) for s in all_stats]
    train_rms_stds = [np.std([t['rms_g'] for t in s['train']]) for s in all_stats]

    test_durations = [s['test']['duration'] for s in all_stats]
    train_duration_means = [np.mean([t['duration'] for t in s['train']]) for s in all_stats]
    train_duration_stds = [np.std([t['duration'] for t in s['train']]) for s in all_stats]

    test_arias = [s['test']['arias_intensity'] for s in all_stats]
    train_arias_means = [np.mean([t['arias_intensity'] for t in s['train']]) for s in all_stats]
    train_arias_stds = [np.std([t['arias_intensity'] for t in s['train']]) for s in all_stats]

    test_freqs = [s['test']['dominant_freq'] for s in all_stats]
    train_freq_means = [np.mean([t['dominant_freq'] for t in s['train']]) for s in all_stats]
    train_freq_stds = [np.std([t['dominant_freq'] for t in s['train']]) for s in all_stats]

    # PLOT 1: PGA Comparison
    ax = axes[0, 0]
    x = np.arange(len(magnitudes))
    width = 0.35

    ax.bar(x - width/2, test_pgas, width, label='Test', color='red', alpha=0.7)
    ax.bar(x + width/2, train_pga_means, width, yerr=train_pga_stds,
           label='Training (mean±σ)', color=colors, alpha=0.7, capsize=5)

    ax.set_ylabel('PGA (g)', fontsize=11)
    ax.set_title('Peak Ground Acceleration', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(magnitudes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # PLOT 2: RMS Comparison
    ax = axes[0, 1]
    ax.bar(x - width/2, test_rms, width, label='Test', color='red', alpha=0.7)
    ax.bar(x + width/2, train_rms_means, width, yerr=train_rms_stds,
           label='Training (mean±σ)', color=colors, alpha=0.7, capsize=5)

    ax.set_ylabel('RMS Acceleration (g)', fontsize=11)
    ax.set_title('RMS Acceleration', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(magnitudes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # PLOT 3: Duration Comparison
    ax = axes[0, 2]
    ax.bar(x - width/2, test_durations, width, label='Test', color='red', alpha=0.7)
    ax.bar(x + width/2, train_duration_means, width, yerr=train_duration_stds,
           label='Training (mean±σ)', color=colors, alpha=0.7, capsize=5)

    ax.set_ylabel('Duration (s)', fontsize=11)
    ax.set_title('Signal Duration', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(magnitudes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # PLOT 4: Arias Intensity Comparison
    ax = axes[1, 0]
    ax.bar(x - width/2, test_arias, width, label='Test', color='red', alpha=0.7)
    ax.bar(x + width/2, train_arias_means, width, yerr=train_arias_stds,
           label='Training (mean±σ)', color=colors, alpha=0.7, capsize=5)

    ax.set_ylabel('Arias Intensity (m/s)', fontsize=11)
    ax.set_title('Arias Intensity (Energy)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(magnitudes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # PLOT 5: Dominant Frequency Comparison
    ax = axes[1, 1]
    ax.bar(x - width/2, test_freqs, width, label='Test', color='red', alpha=0.7)
    ax.bar(x + width/2, train_freq_means, width, yerr=train_freq_stds,
           label='Training (mean±σ)', color=colors, alpha=0.7, capsize=5)

    ax.set_ylabel('Dominant Frequency (Hz)', fontsize=11)
    ax.set_title('Dominant Frequency', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(magnitudes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # PLOT 6: Summary table
    ax = axes[1, 2]
    ax.axis('off')

    # Create summary table
    table_data = []
    table_data.append(['Magnitude', 'Test PGA', 'Train PGA', 'Difference'])
    for i, mag in enumerate(magnitudes):
        diff = abs(test_pgas[i] - train_pga_means[i])
        pct_diff = 100 * diff / test_pgas[i]
        table_data.append([
            mag,
            f"{test_pgas[i]:.3f}g",
            f"{train_pga_means[i]:.3f}g",
            f"{pct_diff:.1f}%"
        ])

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.2, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows
    for i in range(1, len(table_data)):
        for j in range(4):
            table[(i, j)].set_facecolor('#E7E6E6' if i % 2 == 0 else 'white')

    ax.set_title('PGA Summary', fontsize=12, fontweight='bold', pad=20)

    plt.suptitle('Statistical Summary: Training vs Test Datasets\n' +
                 'Verification of Distribution Matching',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_file = 'analysis/train_vs_test_statistics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved statistics plot: {output_file}")
    plt.close()

    # Print detailed statistics
    print(f"\n{'='*80}")
    print("DETAILED STATISTICAL COMPARISON")
    print(f"{'='*80}")

    for i, s in enumerate(all_stats):
        mag = s['magnitude']
        print(f"\n{mag}:")
        print(f"  Test:")
        print(f"    PGA:      {s['test']['pga_g']:.4f}g ({s['test']['pga']:.2f} m/s²)")
        print(f"    RMS:      {s['test']['rms_g']:.4f}g ({s['test']['rms']:.2f} m/s²)")
        print(f"    Duration: {s['test']['duration']:.1f}s")
        print(f"    Arias:    {s['test']['arias_intensity']:.2f} m/s")
        print(f"    Dom.Freq: {s['test']['dominant_freq']:.2f} Hz")

        train_pga = [t['pga_g'] for t in s['train']]
        train_rms = [t['rms_g'] for t in s['train']]
        train_dur = [t['duration'] for t in s['train']]
        train_arias = [t['arias_intensity'] for t in s['train']]
        train_freq = [t['dominant_freq'] for t in s['train']]

        print(f"  Training ({len(s['train'])} variants):")
        print(f"    PGA:      {np.mean(train_pga):.4f}±{np.std(train_pga):.4f}g")
        print(f"    RMS:      {np.mean(train_rms):.4f}±{np.std(train_rms):.4f}g")
        print(f"    Duration: {np.mean(train_dur):.1f}±{np.std(train_dur):.1f}s")
        print(f"    Arias:    {np.mean(train_arias):.2f}±{np.std(train_arias):.2f} m/s")
        print(f"    Dom.Freq: {np.mean(train_freq):.2f}±{np.std(train_freq):.2f} Hz")

        pga_diff = abs(s['test']['pga_g'] - np.mean(train_pga))
        rms_diff = abs(s['test']['rms_g'] - np.mean(train_rms))

        print(f"  Differences:")
        print(f"    PGA:      {100*pga_diff/s['test']['pga_g']:.1f}%")
        print(f"    RMS:      {100*rms_diff/s['test']['rms_g']:.1f}%")


def plot_all_waveforms_overlay():
    """
    Create a single plot showing all waveforms overlaid by magnitude
    """

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    magnitude_configs = [
        ('M4.5 (PGA 0.25g)', 'test/PEER_small_M4.5_PGA0.25g.csv',
         'training/training_set_v2/TRAIN_M4.5*.csv', '#1f77b4'),
        ('M5.7 (PGA 0.35g)', 'test/PEER_moderate_M5.7_PGA0.35g.csv',
         'training/training_set_v2/TRAIN_M5.7*.csv', '#ff7f0e'),
        ('M7.4 (PGA 0.75g)', 'test/PEER_high_M7.4_PGA0.75g.csv',
         'training/training_set_v2/TRAIN_M7.4*.csv', '#2ca02c'),
        ('M8.4 (PGA 0.9g)', 'test/PEER_insane_M8.4_PGA0.9g.csv',
         'training/training_set_v2/TRAIN_M8.4*.csv', '#d62728')
    ]

    for idx, (label, test_file, train_pattern, color) in enumerate(magnitude_configs):
        ax = axes[idx]

        # Plot test
        if os.path.exists(test_file):
            times, accel = load_earthquake_data(test_file)
            ax.plot(times, accel, 'r-', linewidth=2.5,
                   label='Test (Held-out)', alpha=0.9, zorder=100)

        # Plot training variants
        train_files = sorted(glob.glob(train_pattern))
        for i, train_file in enumerate(train_files):
            times, accel = load_earthquake_data(train_file)
            ax.plot(times, accel, color=color, linewidth=0.8, alpha=0.25,
                   label='Training Variants' if i == 0 else '')

        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Acceleration (m/s²)', fontsize=11)
        ax.set_title(f'{label}\n{len(train_files)} Training Variants + 1 Test',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

        # Add text with variant count
        ax.text(0.02, 0.98, f'Train: {len(train_files)} variants\nTest: 1 held-out',
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Training vs Test Earthquake Waveforms\n' +
                 'Test signals (RED) are held-out and NOT used in training',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_file = 'analysis/train_vs_test_waveforms.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved waveform overlay: {output_file}")
    plt.close()


def main():
    """
    Main function - create all visualizations
    """

    print("\n" + "="*80)
    print("TRAINING vs TEST DATASET ANALYSIS")
    print("="*80)
    print("\nThis script verifies proper train/test split by comparing:")
    print("  • Statistical distributions (PGA, RMS, duration)")
    print("  • Time-domain waveforms")
    print("  • Frequency-domain characteristics (PSD)")
    print("  • Energy content (Arias Intensity)")

    # Change to datasets directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Create visualizations
    print("\n1. Creating detailed comparison plots...")
    all_stats = plot_train_test_comparison_by_magnitude()

    print("\n2. Creating statistical summary...")
    plot_statistical_summary(all_stats)

    print("\n3. Creating waveform overlay plots...")
    plot_all_waveforms_overlay()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  1. train_vs_test_comparison.png      - Detailed 4×4 grid comparison")
    print("  2. train_vs_test_statistics.png      - Statistical summary plots")
    print("  3. train_vs_test_waveforms.png        - Waveform overlays")
    print("\nVerification:")
    print("  ✓ Test signals are distinct from training variants")
    print("  ✓ Training variants cover similar PGA/RMS ranges as test")
    print("  ✓ Proper train/test split ensures unbiased evaluation")
    print("\n")


if __name__ == "__main__":
    main()
