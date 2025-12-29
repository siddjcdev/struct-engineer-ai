"""
GENERATE SYNTHETIC TRAINING EARTHQUAKES
========================================

Generates new earthquake time series with same magnitude/PGA as test data,
but different waveforms to prevent memorization.

Creates training set with identical statistical properties but different realizations:
- M4.5 (PGA 0.25g) - 3 variants for training
- M5.7 (PGA 0.35g) - 3 variants for training
- M7.4 (PGA 0.75g) - 3 variants for training
- M8.4 (PGA 0.9g) - 3 variants for training

Uses stochastic ground motion synthesis with controlled spectral characteristics.

Usage: python generate_training_earthquakes.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os


def generate_kanai_tajimi_earthquake(duration, dt, pga_target, omega_g, zeta_g, seed=None):
    """
    Generate synthetic earthquake using Kanai-Tajimi spectrum

    Parameters:
    - duration: earthquake duration (seconds)
    - dt: time step (seconds)
    - pga_target: target peak ground acceleration (m/s²)
    - omega_g: predominant frequency (rad/s)
    - zeta_g: damping ratio for ground
    - seed: random seed for reproducibility

    Returns:
    - time: time array
    - acceleration: ground acceleration (m/s²)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate white noise
    n_steps = int(duration / dt)
    white_noise = np.random.randn(n_steps)

    # Kanai-Tajimi filter (models frequency content of earthquakes)
    num = [1, 2*zeta_g*omega_g, omega_g**2]
    den = [1, 2*zeta_g*omega_g, omega_g**2]

    # Apply filter
    filtered = signal.lfilter(num, den, white_noise)

    # Apply envelope function (Jennings-type)
    time = np.arange(n_steps) * dt
    t1 = duration * 0.15  # Rise time
    t2 = duration * 0.7   # Decay start

    envelope = np.ones(n_steps)
    # Exponential rise
    rise_mask = time < t1
    envelope[rise_mask] = (time[rise_mask] / t1) ** 2

    # Exponential decay
    decay_mask = time > t2
    c = 0.4  # Decay parameter
    envelope[decay_mask] = np.exp(-c * (time[decay_mask] - t2))

    # Apply envelope
    acceleration = filtered * envelope

    # Scale to target PGA
    current_pga = np.max(np.abs(acceleration))
    if current_pga > 0:
        acceleration = acceleration * (pga_target / current_pga)

    return time, acceleration


def generate_clough_penzien_earthquake(duration, dt, pga_target, omega_g, zeta_g, omega_f, zeta_f, seed=None):
    """
    Generate synthetic earthquake using Clough-Penzien spectrum
    (More sophisticated - includes high-pass filter to remove low frequencies)

    Parameters:
    - omega_f: high-pass filter frequency (rad/s)
    - zeta_f: high-pass filter damping
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate white noise
    n_steps = int(duration / dt)
    white_noise = np.random.randn(n_steps)

    # Kanai-Tajimi filter
    num_kt = [omega_g**2]
    den_kt = [1, 2*zeta_g*omega_g, omega_g**2]

    kt_filtered = signal.lfilter(num_kt, den_kt, white_noise)

    # High-pass filter (Clough-Penzien)
    num_hp = [1, 0, 0]
    den_hp = [1, 2*zeta_f*omega_f, omega_f**2]

    cp_filtered = signal.lfilter(num_hp, den_hp, kt_filtered)

    # Apply envelope
    time = np.arange(n_steps) * dt
    t1 = duration * 0.15
    t2 = duration * 0.7

    envelope = np.ones(n_steps)
    rise_mask = time < t1
    envelope[rise_mask] = (time[rise_mask] / t1) ** 2

    decay_mask = time > t2
    c = 0.4
    envelope[decay_mask] = np.exp(-c * (time[decay_mask] - t2))

    acceleration = cp_filtered * envelope

    # Scale to target PGA
    current_pga = np.max(np.abs(acceleration))
    if current_pga > 0:
        acceleration = acceleration * (pga_target / current_pga)

    return time, acceleration


def generate_earthquake_variants(magnitude_name, pga_g, n_variants=3, duration=40.0, dt=0.02):
    """
    Generate multiple variants of earthquake with same PGA but different waveforms

    Parameters:
    - magnitude_name: e.g., "M4.5", "M5.7", "M7.4", "M8.4"
    - pga_g: target PGA in g (e.g., 0.25, 0.35, 0.75, 0.9)
    - n_variants: number of different realizations to generate
    - duration: earthquake duration (seconds)
    - dt: time step (seconds)

    Returns:
    - list of (time, acceleration) tuples
    """
    pga_mps2 = pga_g * 9.81  # Convert to m/s²

    # Frequency parameters based on magnitude
    # Larger earthquakes have lower frequency content
    freq_params = {
        "M4.5": {"omega_g": 2*np.pi*3.0, "zeta_g": 0.6, "omega_f": 2*np.pi*0.5, "zeta_f": 0.6},
        "M5.7": {"omega_g": 2*np.pi*2.5, "zeta_g": 0.6, "omega_f": 2*np.pi*0.4, "zeta_f": 0.6},
        "M7.4": {"omega_g": 2*np.pi*2.0, "zeta_g": 0.7, "omega_f": 2*np.pi*0.3, "zeta_f": 0.6},
        "M8.4": {"omega_g": 2*np.pi*1.5, "zeta_g": 0.7, "omega_f": 2*np.pi*0.2, "zeta_f": 0.6},
    }

    params = freq_params[magnitude_name]
    variants = []

    for i in range(n_variants):
        seed = hash(f"{magnitude_name}_{pga_g}_{i}") % (2**31)

        time, accel = generate_clough_penzien_earthquake(
            duration, dt, pga_mps2,
            params["omega_g"], params["zeta_g"],
            params["omega_f"], params["zeta_f"],
            seed=seed
        )

        variants.append((time, accel))

        print(f"   Generated {magnitude_name} variant {i+1}: PGA = {np.max(np.abs(accel))/9.81:.3f}g")

    return variants


def save_earthquake_csv(time, acceleration, filename):
    """Save earthquake data to CSV format matching PEER format"""
    # Create data array with time and acceleration
    data = np.column_stack([time, acceleration])

    # Save with header
    header = "Time (s),Acceleration (m/s^2)"
    np.savetxt(filename, data, delimiter=',', header=header, comments='', fmt='%.6f')

    print(f"   Saved: {filename}")


def plot_comparison(original_file, synthetic_variants, magnitude_name, pga_g):
    """Plot original vs synthetic earthquakes for visual verification"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{magnitude_name} (PGA {pga_g}g): Original vs Synthetic Training Data', fontsize=14)

    # Load original
    try:
        orig_data = np.loadtxt(original_file, delimiter=',', skiprows=1)
        orig_time = orig_data[:, 0]
        orig_accel = orig_data[:, 1]
    except:
        print(f"   Warning: Could not load {original_file}")
        return

    # Plot 1: Time histories
    ax = axes[0, 0]
    ax.plot(orig_time, orig_accel/9.81, 'k-', linewidth=2, label='Original (Test)', alpha=0.7)
    for i, (time, accel) in enumerate(synthetic_variants):
        ax.plot(time, accel/9.81, '--', label=f'Synthetic {i+1} (Train)', alpha=0.6)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (g)')
    ax.set_title('Time Histories')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Fourier spectra
    ax = axes[0, 1]
    dt = np.mean(np.diff(orig_time))
    freq = np.fft.rfftfreq(len(orig_accel), dt)

    orig_fft = np.abs(np.fft.rfft(orig_accel))
    ax.semilogy(freq, orig_fft, 'k-', linewidth=2, label='Original (Test)', alpha=0.7)

    for i, (time, accel) in enumerate(synthetic_variants):
        dt = np.mean(np.diff(time))
        freq = np.fft.rfftfreq(len(accel), dt)
        synth_fft = np.abs(np.fft.rfft(accel))
        ax.semilogy(freq, synth_fft, '--', label=f'Synthetic {i+1} (Train)', alpha=0.6)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Fourier Amplitude Spectra')
    ax.set_xlim([0, 10])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Peak values comparison
    ax = axes[1, 0]
    orig_pga = np.max(np.abs(orig_accel)) / 9.81
    synth_pgas = [np.max(np.abs(accel))/9.81 for _, accel in synthetic_variants]

    labels = ['Original'] + [f'Synth {i+1}' for i in range(len(synthetic_variants))]
    pgas = [orig_pga] + synth_pgas
    colors = ['black'] + ['C0', 'C1', 'C2'][:len(synthetic_variants)]

    ax.bar(labels, pgas, color=colors, alpha=0.7)
    ax.axhline(pga_g, color='r', linestyle='--', label=f'Target: {pga_g}g')
    ax.set_ylabel('PGA (g)')
    ax.set_title('Peak Ground Acceleration')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Cumulative energy
    ax = axes[1, 1]
    orig_energy = np.cumsum(orig_accel**2) / np.sum(orig_accel**2)
    ax.plot(orig_time, orig_energy, 'k-', linewidth=2, label='Original (Test)', alpha=0.7)

    for i, (time, accel) in enumerate(synthetic_variants):
        energy = np.cumsum(accel**2) / np.sum(accel**2)
        ax.plot(time, energy, '--', label=f'Synthetic {i+1} (Train)', alpha=0.6)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized Cumulative Energy')
    ax.set_title('Energy Build-up (checks duration/intensity)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = f'training_earthquakes_{magnitude_name}_comparison.png'
    plt.savefig(plot_file, dpi=150)
    print(f"   Comparison plot saved: {plot_file}")
    plt.close()


def main():
    """Generate training earthquake dataset"""

    print("="*70)
    print("  SYNTHETIC TRAINING EARTHQUAKE GENERATION")
    print("="*70)
    print("\nGenerating new earthquakes with same magnitudes as test set,")
    print("but different waveforms to prevent memorization.\n")

    # Define target earthquakes matching test set
    earthquakes = [
        {
            "magnitude": "M4.5",
            "pga_g": 0.25,
            "n_variants": 3,
            "original": "PEER_small_M4.5_PGA0.25g.csv"
        },
        {
            "magnitude": "M5.7",
            "pga_g": 0.35,
            "n_variants": 3,
            "original": "PEER_moderate_M5.7_PGA0.35g.csv"
        },
        {
            "magnitude": "M7.4",
            "pga_g": 0.75,
            "n_variants": 3,
            "original": "PEER_high_M7.4_PGA0.75g.csv"
        },
        {
            "magnitude": "M8.4",
            "pga_g": 0.9,
            "n_variants": 3,
            "original": "PEER_insane_M8.4_PGA0.9g.csv"
        }
    ]

    # Create output directory
    output_dir = "training_set"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}/\n")

    # Generate each magnitude class
    for eq in earthquakes:
        print(f"\n{eq['magnitude']} (PGA {eq['pga_g']}g):")
        print("-" * 50)

        # Generate variants
        variants = generate_earthquake_variants(
            eq["magnitude"],
            eq["pga_g"],
            n_variants=eq["n_variants"]
        )

        # Save each variant
        for i, (time, accel) in enumerate(variants, 1):
            filename = f"{output_dir}/TRAIN_{eq['magnitude']}_PGA{eq['pga_g']}g_variant{i}.csv"
            save_earthquake_csv(time, accel, filename)

        # Plot comparison with original
        original_path = eq["original"]
        if os.path.exists(original_path):
            plot_comparison(original_path, variants, eq["magnitude"], eq["pga_g"])
        else:
            print(f"   Warning: Original file not found: {original_path}")

    print("\n" + "="*70)
    print("  GENERATION COMPLETE!")
    print("="*70)
    print(f"\nGenerated {sum(eq['n_variants'] for eq in earthquakes)} training earthquakes")
    print(f"Location: {output_dir}/")
    print("\nNext steps:")
    print("1. Verify comparison plots to ensure synthetic data quality")
    print("2. Train model using: training_set/TRAIN_*.csv")
    print("3. Test model using: PEER_*.csv (original test set)")
    print("\nThis ensures training data ≠ testing data at the fundamental level!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
