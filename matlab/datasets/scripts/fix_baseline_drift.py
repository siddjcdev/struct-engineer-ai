"""
EARTHQUAKE BASELINE CORRECTION SCRIPT
======================================

Fixes baseline drift in earthquake accelerograms using standard seismology methods:
1. High-pass Butterworth filter (0.1 Hz cutoff) to remove low-frequency drift
2. Linear detrending to remove any remaining trend
3. Ensures zero final velocity and displacement
4. Scales to match target PGA

This eliminates the 30+ meter baseline drift that causes catastrophic displacements.

Usage:
    python fix_baseline_drift.py

Author: Siddharth
Date: December 2025
"""

import numpy as np
import os
from scipy import signal
import glob


def baseline_correct_acceleration(accel, dt, target_pga_g=None):
    """
    Apply baseline correction to earthquake acceleration time series

    Args:
        accel: Acceleration array (m/s^2)
        dt: Time step (s)
        target_pga_g: Target PGA in g (optional, for scaling)

    Returns:
        corrected_accel: Baseline-corrected acceleration (m/s^2)
    """

    # Step 1: High-pass Butterworth filter (0.1 Hz cutoff)
    # Removes low-frequency drift and instrument noise
    fs = 1.0 / dt  # Sampling frequency
    fc = 0.1  # Cutoff frequency (Hz) - standard for earthquake engineering
    order = 4  # Filter order

    nyquist = fs / 2.0
    normalized_cutoff = fc / nyquist

    # Design Butterworth high-pass filter
    b, a = signal.butter(order, normalized_cutoff, btype='high', analog=False)

    # Apply filter (forward and backward to avoid phase shift)
    accel_filtered = signal.filtfilt(b, a, accel)

    # Step 2: Linear detrending
    # Remove any remaining linear trend
    accel_detrended = signal.detrend(accel_filtered, type='linear')

    # Step 3: Ensure zero final velocity
    # Integrate to get velocity
    vel = np.cumsum(accel_detrended) * dt

    # Subtract linear trend from velocity to force final velocity = 0
    # This ensures the ground returns to rest
    t = np.arange(len(vel)) * dt
    vel_slope = vel[-1] / t[-1] if t[-1] > 0 else 0
    vel_corrected = vel - vel_slope * t

    # Differentiate corrected velocity back to acceleration
    accel_corrected = np.diff(vel_corrected) / dt
    accel_corrected = np.append(accel_corrected, accel_corrected[-1])  # Match length

    # Step 4: Scale to target PGA if specified
    if target_pga_g is not None:
        current_pga = np.max(np.abs(accel_corrected))
        target_pga = target_pga_g * 9.81  # Convert g to m/s^2

        if current_pga > 0:
            scale_factor = target_pga / current_pga
            accel_corrected = accel_corrected * scale_factor

    return accel_corrected


def verify_correction(accel_original, accel_corrected, dt, name):
    """
    Verify baseline correction by checking drift metrics

    Args:
        accel_original: Original acceleration
        accel_corrected: Corrected acceleration
        dt: Time step
        name: Dataset name for reporting
    """

    # Integrate original
    vel_orig = np.cumsum(accel_original) * dt
    disp_orig = np.cumsum(vel_orig) * dt

    # Integrate corrected
    vel_corr = np.cumsum(accel_corrected) * dt
    disp_corr = np.cumsum(vel_corr) * dt

    print(f"\n{'='*70}")
    print(f"VERIFICATION: {name}")
    print(f"{'='*70}")

    print(f"\nOriginal:")
    print(f"  PGA: {np.max(np.abs(accel_original)):.4f} m/s² ({np.max(np.abs(accel_original))/9.81:.3f}g)")
    print(f"  Final velocity: {vel_orig[-1]:.4f} m/s")
    print(f"  Final displacement: {disp_orig[-1]:.4f} m")
    print(f"  ⚠️  Baseline drift: {abs(disp_orig[-1]):.2f} m")

    print(f"\nCorrected:")
    print(f"  PGA: {np.max(np.abs(accel_corrected)):.4f} m/s² ({np.max(np.abs(accel_corrected))/9.81:.3f}g)")
    print(f"  Final velocity: {vel_corr[-1]:.6f} m/s")
    print(f"  Final displacement: {disp_corr[-1]:.6f} m")
    print(f"  ✓ Baseline drift: {abs(disp_corr[-1]):.6f} m (should be ~0)")

    # Check if correction was successful
    if abs(disp_corr[-1]) < 0.1:
        print(f"\n✓ SUCCESS: Baseline drift reduced from {abs(disp_orig[-1]):.1f}m to {abs(disp_corr[-1]):.4f}m")
    else:
        print(f"\n⚠️  WARNING: Baseline drift still significant ({abs(disp_corr[-1]):.2f}m)")


def process_dataset(input_file, output_file, target_pga_g=None):
    """
    Load, correct, and save earthquake dataset

    Args:
        input_file: Path to input CSV
        output_file: Path to output CSV
        target_pga_g: Target PGA in g (optional)
    """

    print(f"\n{'='*70}")
    print(f"Processing: {os.path.basename(input_file)}")
    print(f"{'='*70}")

    # Load data
    data = np.loadtxt(input_file, delimiter=',', skiprows=1)
    times = data[:, 0]
    accel_original = data[:, 1]
    dt = float(np.mean(np.diff(times)))

    print(f"Loaded {len(accel_original)} samples (dt={dt:.4f}s, duration={times[-1]:.1f}s)")

    # Apply baseline correction
    accel_corrected = baseline_correct_acceleration(accel_original, dt, target_pga_g)

    # Verify correction
    verify_correction(accel_original, accel_corrected, dt, os.path.basename(input_file))

    # Save corrected data
    header = "time_s,acceleration_ms2"
    corrected_data = np.column_stack([times, accel_corrected])
    np.savetxt(output_file, corrected_data, delimiter=',', header=header, comments='', fmt='%.10f')

    print(f"\n✓ Saved corrected data to: {output_file}")

    return accel_corrected


def main():
    """
    Process all earthquake datasets
    """

    print("\n" + "="*70)
    print("EARTHQUAKE BASELINE CORRECTION")
    print("="*70)
    print("\nThis script will fix baseline drift in all earthquake datasets.")
    print("Original files will be backed up with .original extension.")

    # Define datasets to process
    datasets = [
        # PEER test set (held-out)
        {
            'file': 'PEER_high_M7.4_PGA0.75g.csv',
            'target_pga': 0.75,
            'description': 'M7.4 PEER Test Set'
        },
        {
            'file': 'PEER_small_M4.5_PGA0.25g.csv',
            'target_pga': 0.25,
            'description': 'M4.5 PEER Test Set'
        },
        {
            'file': 'PEER_moderate_M5.7_PGA0.35g.csv',
            'target_pga': 0.35,
            'description': 'M5.7 PEER Test Set'
        },
        {
            'file': 'PEER_insane_M8.4_PGA0.9g.csv',
            'target_pga': 0.9,
            'description': 'M8.4 PEER Test Set'
        },
    ]

    # Add training set files
    training_files = glob.glob('training_set/TRAIN_*.csv')
    for train_file in training_files:
        # Extract target PGA from filename
        if 'M4.5' in train_file:
            target_pga = 0.25
        elif 'M5.7' in train_file:
            target_pga = 0.35
        elif 'M7.4' in train_file:
            target_pga = 0.75
        elif 'M8.4' in train_file:
            target_pga = 0.9
        else:
            target_pga = None

        datasets.append({
            'file': train_file,
            'target_pga': target_pga,
            'description': os.path.basename(train_file)
        })

    print(f"\nFound {len(datasets)} datasets to process.")
    print("\nStarting correction...")

    # Process each dataset
    success_count = 0
    for dataset in datasets:
        input_file = dataset['file']

        if not os.path.exists(input_file):
            print(f"\n⚠️  Skipping {input_file} (not found)")
            continue

        # Backup original file
        backup_file = input_file + '.original'
        if not os.path.exists(backup_file):
            import shutil
            shutil.copy2(input_file, backup_file)
            print(f"✓ Backed up to: {backup_file}")

        # Process
        try:
            process_dataset(
                input_file=input_file,
                output_file=input_file,  # Overwrite original
                target_pga_g=dataset['target_pga']
            )
            success_count += 1
        except Exception as e:
            print(f"\n❌ ERROR processing {input_file}: {str(e)}")

    # Summary
    print("\n" + "="*70)
    print("CORRECTION COMPLETE")
    print("="*70)
    print(f"\n✓ Successfully processed {success_count}/{len(datasets)} datasets")
    print(f"\n📁 Original files backed up with .original extension")
    print(f"📁 Corrected files saved in place")
    print(f"\nYou can now re-run training with corrected data!")
    print(f"\nTo restore originals: rm *.csv training_set/*.csv && mv *.original *.csv && mv training_set/*.original training_set/*.csv")


if __name__ == "__main__":
    main()
