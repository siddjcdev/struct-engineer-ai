"""
GENERATE SYNTHETIC TRAINING EARTHQUAKES - STABLE VERSION
=========================================================

Generates new earthquake time series with same magnitude/PGA as test data,
but different waveforms to prevent memorization.

Uses FREQUENCY-DOMAIN approach for numerical stability.

Creates training set with identical statistical properties but different realizations:
- M4.5 (PGA 0.25g) - 3 variants for training
- M5.7 (PGA 0.35g) - 3 variants for training
- M7.4 (PGA 0.75g) - 3 variants for training
- M8.4 (PGA 0.9g) - 3 variants for training

Usage: python generate_training_earthquakes.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def kanai_tajimi_spectrum(freq, omega_g, zeta_g, S0=1.0):
    """
    Kanai-Tajimi power spectral density function

    S(ω) = S0 * (1 + 4*ζ_g^2*(ω/ω_g)^2) / ((1 - (ω/ω_g)^2)^2 + 4*ζ_g^2*(ω/ω_g)^2)

    Parameters:
    - freq: frequency array (Hz)
    - omega_g: predominant circular frequency (rad/s)
    - zeta_g: damping ratio
    - S0: spectral intensity

    Returns:
    - S: power spectral density at each frequency
    """
    omega = 2 * np.pi * freq
    omega_g = omega_g

    # Avoid division by zero
    omega_ratio = np.divide(omega, omega_g, out=np.zeros_like(omega), where=omega_g!=0)

    numerator = 1 + 4 * zeta_g**2 * omega_ratio**2
    denominator = (1 - omega_ratio**2)**2 + 4 * zeta_g**2 * omega_ratio**2

    # Prevent division by zero
    S = np.divide(S0 * numerator, denominator, out=np.zeros_like(denominator), where=denominator>1e-10)

    return S


def generate_earthquake_frequency_domain(duration, dt, pga_target, omega_g, zeta_g, seed=None):
    """
    Generate synthetic earthquake using FREQUENCY DOMAIN approach (stable!)

    Method:
    1. Generate random phase spectrum
    2. Apply Kanai-Tajimi amplitude spectrum
    3. IFFT to time domain
    4. Apply temporal envelope
    5. Scale to target PGA

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

    # Time array
    n_steps = int(duration / dt)
    time = np.arange(n_steps) * dt

    # Frequency array (positive frequencies only for rfft)
    freq = np.fft.rfftfreq(n_steps, dt)

    # Generate random phase (uniformly distributed 0 to 2π)
    random_phase = np.random.uniform(0, 2*np.pi, len(freq))

    # Kanai-Tajimi spectrum (amplitude)
    f_g = omega_g / (2 * np.pi)  # Convert to Hz
    S0 = 1.0  # Will scale later
    amplitude_spectrum = np.sqrt(kanai_tajimi_spectrum(freq, omega_g, zeta_g, S0))

    # Construct complex spectrum with random phase
    complex_spectrum = amplitude_spectrum * np.exp(1j * random_phase)

    # Ensure DC component is zero (no constant offset)
    complex_spectrum[0] = 0

    # IFFT to time domain
    acceleration = np.fft.irfft(complex_spectrum, n=n_steps)

    # Apply temporal envelope (Jennings-type)
    t1 = duration * 0.15  # Rise time (15% of duration)
    t2 = duration * 0.70  # Start of decay (70% of duration)

    envelope = np.ones(n_steps)

    # Exponential rise
    rise_mask = time < t1
    envelope[rise_mask] = (time[rise_mask] / t1) ** 2

    # Exponential decay
    decay_mask = time > t2
    c = 0.5  # Decay parameter
    envelope[decay_mask] = np.exp(-c * (time[decay_mask] - t2))

    # Apply envelope
    acceleration = acceleration * envelope

    # Scale to target PGA
    current_pga = np.max(np.abs(acceleration))
    if current_pga > 1e-6:  # Avoid division by very small numbers
        acceleration = acceleration * (pga_target / current_pga)
    else:
        print(f"Warning: Generated acceleration too small, regenerating...")
        return generate_earthquake_frequency_domain(duration, dt, pga_target, omega_g, zeta_g, seed=seed+1 if seed else None)

    # Sanity check: verify no NaN or Inf values
    if np.any(np.isnan(acceleration)) or np.any(np.isinf(acceleration)):
        print(f"Warning: NaN or Inf detected, regenerating...")
        return generate_earthquake_frequency_domain(duration, dt, pga_target, omega_g, zeta_g, seed=seed+1 if seed else None)

    # Sanity check: verify realistic range (earthquakes don't exceed ~2g typically)
    max_accel_g = np.max(np.abs(acceleration)) / 9.81
    if max_accel_g > 2.0:
        print(f"Warning: Unrealistic PGA {max_accel_g:.2f}g, regenerating...")
        return generate_earthquake_frequency_domain(duration, dt, pga_target, omega_g, zeta_g, seed=seed+1 if seed else None)

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
    # Larger earthquakes have lower predominant frequencies
    freq_params = {
        "M4.5": {"omega_g": 2*np.pi*4.0, "zeta_g": 0.6},   # 4 Hz predominant
        "M5.7": {"omega_g": 2*np.pi*3.0, "zeta_g": 0.65},  # 3 Hz predominant
        "M7.4": {"omega_g": 2*np.pi*2.0, "zeta_g": 0.7},   # 2 Hz predominant
        "M8.4": {"omega_g": 2*np.pi*1.5, "zeta_g": 0.75},  # 1.5 Hz predominant
    }

    params = freq_params[magnitude_name]
    variants = []

    for i in range(n_variants):
        # Use deterministic seed for reproducibility
        seed = hash(f"{magnitude_name}_{pga_g}_{i}_v2") % (2**31)

        time, accel = generate_earthquake_frequency_domain(
            duration, dt, pga_mps2,
            params["omega_g"], params["zeta_g"],
            seed=seed
        )

        variants.append((time, accel))

        actual_pga_g = np.max(np.abs(accel)) / 9.81
        print(f"   Generated {magnitude_name} variant {i+1}: PGA = {actual_pga_g:.3f}g (target: {pga_g:.3f}g)")

    return variants


def save_earthquake_csv(time, acceleration, filename):
    """Save earthquake data to CSV format matching PEER format"""
    # Sanity check before saving
    if np.any(np.isnan(acceleration)) or np.any(np.isinf(acceleration)):
        raise ValueError(f"Cannot save {filename}: contains NaN or Inf values!")

    if np.max(np.abs(acceleration)) > 100:  # > 10g is unrealistic
        raise ValueError(f"Cannot save {filename}: unrealistic acceleration values!")

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
        orig_time = synthetic_variants[0][0]
        orig_accel = np.zeros_like(synthetic_variants[0][1])

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
    ax.set_ylim([-pga_g*1.5, pga_g*1.5])  # Reasonable limits

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
    if np.sum(orig_accel**2) > 0:
        orig_energy = np.cumsum(orig_accel**2) / np.sum(orig_accel**2)
        ax.plot(orig_time, orig_energy, 'k-', linewidth=2, label='Original (Test)', alpha=0.7)

    for i, (time, accel) in enumerate(synthetic_variants):
        if np.sum(accel**2) > 0:
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
    print("  SYNTHETIC TRAINING EARTHQUAKE GENERATION (STABLE VERSION)")
    print("="*70)
    print("\nGenerating new earthquakes with same magnitudes as test set,")
    print("but different waveforms to prevent memorization.\n")
    print("Using FREQUENCY-DOMAIN method for numerical stability.\n")

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
        plot_comparison(original_path, variants, eq["magnitude"], eq["pga_g"])

    print("\n" + "="*70)
    print("  GENERATION COMPLETE!")
    print("="*70)
    print(f"\nGenerated {sum(eq['n_variants'] for eq in earthquakes)} training earthquakes")
    print(f"Location: {output_dir}/")
    print("\nNext steps:")
    print("1. Verify comparison plots - ensure synthetic data looks realistic")
    print("2. Check that PGAs match targets (within ±5%)")
    print("3. Train model using: training_set/TRAIN_*.csv")
    print("4. Test model using: PEER_*.csv (held-out test set)")
    print("\nThis ensures training data ≠ testing data!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
