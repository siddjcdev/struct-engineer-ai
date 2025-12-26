"""
Generate synthetic earthquake ground motions based on PEER data characteristics

Creates realistic time series with:
- Appropriate frequency content
- Envelope functions for realistic buildup/decay
- Magnitude-appropriate durations
- Perturbations (noise, delay, dropout)

Uses only Python standard library (no numpy/pandas required)
"""

import math
import csv
import random
from pathlib import Path

def generate_earthquake(magnitude, pga_g, duration, dt=0.02,
                       noise_level=0.10, delay_ms=60, dropout_rate=0.08):
    """
    Generate synthetic earthquake ground motion

    Args:
        magnitude: Earthquake magnitude (Richter scale)
        pga_g: Peak ground acceleration in g (9.81 m/s²)
        duration: Total duration in seconds
        dt: Time step in seconds
        noise_level: Fraction of PGA to add as noise
        delay_ms: Communication delay in milliseconds
        dropout_rate: Fraction of samples to drop

    Returns:
        List of (time, acceleration) tuples
    """

    # Convert PGA to m/s²
    pga = pga_g * 9.81

    # Time array
    n_steps = int(duration / dt) + 1
    time = [i * dt for i in range(n_steps)]

    # Generate base acceleration using multiple frequency components
    # Real earthquakes have complex frequency content
    acc = [0.0] * n_steps

    # Dominant frequencies based on magnitude (larger = lower freq)
    if magnitude < 5.0:
        freqs = [2.0, 3.5, 5.0, 8.0, 12.0]  # Hz
        weights = [0.4, 0.3, 0.2, 0.1, 0.05]
    elif magnitude < 6.0:
        freqs = [1.5, 2.5, 4.0, 6.0, 9.0]
        weights = [0.35, 0.3, 0.2, 0.1, 0.05]
    elif magnitude < 7.0:
        freqs = [1.0, 2.0, 3.0, 5.0, 7.0]
        weights = [0.4, 0.25, 0.2, 0.1, 0.05]
    elif magnitude < 8.0:
        freqs = [0.5, 1.0, 2.0, 3.5, 5.0]
        weights = [0.4, 0.3, 0.15, 0.1, 0.05]
    else:  # Magnitude >= 8.0
        freqs = [0.3, 0.7, 1.2, 2.0, 3.0]
        weights = [0.45, 0.25, 0.15, 0.1, 0.05]

    # Add frequency components
    for freq, weight in zip(freqs, weights):
        phase = random.uniform(0, 2*math.pi)
        for i, t in enumerate(time):
            acc[i] += weight * math.sin(2 * math.pi * freq * t + phase)

    # Create realistic envelope (Saragoni-Hart type)
    # Buildup, strong motion, decay
    t_rise = 0.15 * duration  # 15% buildup
    t_strong = 0.5 * duration  # 50% strong motion

    envelope = [0.0] * n_steps
    for i, t in enumerate(time):
        if t < t_rise:
            # Exponential buildup
            envelope[i] = (t / t_rise) ** 2
        elif t < t_strong:
            # Strong motion plateau
            envelope[i] = 1.0
        else:
            # Exponential decay
            decay_time = t - t_strong
            envelope[i] = math.exp(-2.0 * decay_time / (duration - t_strong))

    # Apply envelope
    for i in range(n_steps):
        acc[i] = acc[i] * envelope[i]

    # Scale to target PGA
    current_pga = max(abs(a) for a in acc)
    if current_pga > 0:
        scale = pga / current_pga
        acc = [a * scale for a in acc]

    # Add white noise (sensor noise, site effects)
    for i in range(n_steps):
        noise = random.gauss(0, noise_level * pga)
        acc[i] += noise

    # Apply dropout (8% of samples)
    if dropout_rate > 0:
        for i in range(1, n_steps):
            if random.random() < dropout_rate:
                # Hold last valid value (common in real data acquisition)
                acc[i] = acc[i-1]

    # Return as list of tuples
    return [(time[i], acc[i]) for i in range(n_steps)]


def calculate_stats(data):
    """Calculate basic statistics"""
    accelerations = [row[1] for row in data]

    # PGA
    pga = max(abs(a) for a in accelerations)
    pga_g = pga / 9.81

    # RMS
    rms_squared = sum(a**2 for a in accelerations) / len(accelerations)
    rms = math.sqrt(rms_squared)

    return pga, pga_g, rms


def main():
    """Generate all earthquake scenarios"""

    # Output directory
    output_dir = Path('/home/user/struct-engineer-ai/matlab/data/earthquakes/peer_synthetic')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Earthquake scenarios with perturbations
    scenarios_perturbed = [
        {
            'name': 'PEER_small_M4.5_PGA0.25g',
            'description': 'Small Earthquake - Magnitude 4.5 (0.25g PGA)',
            'magnitude': 4.5,
            'pga_g': 0.25,
            'duration': 20.0,
            'perturbations': {'noise_level': 0.10, 'delay_ms': 60, 'dropout_rate': 0.08},
        },
        {
            'name': 'PEER_moderate_M5.7_PGA0.35g',
            'description': 'Moderate Earthquake - Magnitude 5.7 (0.35g PGA)',
            'magnitude': 5.7,
            'pga_g': 0.35,
            'duration': 40.0,
            'perturbations': {'noise_level': 0.10, 'delay_ms': 60, 'dropout_rate': 0.08},
        },
        {
            'name': 'PEER_high_M7.4_PGA0.75g',
            'description': 'High Magnitude Earthquake - 7.4 (0.75g PGA)',
            'magnitude': 7.4,
            'pga_g': 0.75,
            'duration': 80.0,
            'perturbations': {'noise_level': 0.10, 'delay_ms': 60, 'dropout_rate': 0.08},
        },
        {
            'name': 'PEER_insane_M8.4_PGA0.9g',
            'description': 'Insane Magnitude Earthquake - 8.4 (0.9g PGA)',
            'magnitude': 8.4,
            'pga_g': 0.90,
            'duration': 120.0,
            'perturbations': {'noise_level': 0.10, 'delay_ms': 60, 'dropout_rate': 0.08},
        },
    ]

    # Clean baseline (for perturbation testing in MATLAB)
    scenario_clean = {
        'name': 'PEER_moderate_M5.7_PGA0.35g_CLEAN',
        'description': 'Moderate Earthquake - CLEAN (no perturbations)',
        'magnitude': 5.7,
        'pga_g': 0.35,
        'duration': 40.0,
        'perturbations': {'noise_level': 0.0, 'delay_ms': 0, 'dropout_rate': 0.0},
    }

    print("Generating PEER-based synthetic earthquake ground motions...\n")

    # Combine all scenarios
    all_scenarios = scenarios_perturbed + [scenario_clean]

    for scenario in all_scenarios:
        print(f"Generating: {scenario['name']}")
        print(f"  Magnitude: {scenario['magnitude']}")
        print(f"  PGA: {scenario['pga_g']:.2f}g ({scenario['pga_g']*9.81:.2f} m/s²)")
        print(f"  Duration: {scenario['duration']:.1f}s")

        # Show perturbations
        perts = scenario['perturbations']
        if perts['noise_level'] > 0 or perts['dropout_rate'] > 0:
            print(f"  Perturbations: {perts['noise_level']*100}% noise, {perts['delay_ms']}ms delay, {perts['dropout_rate']*100}% dropout")
        else:
            print(f"  Perturbations: CLEAN (no perturbations)")

        # Generate earthquake
        data = generate_earthquake(
            magnitude=scenario['magnitude'],
            pga_g=scenario['pga_g'],
            duration=scenario['duration'],
            dt=0.02,
            **scenario['perturbations']
        )

        # Save to CSV
        output_file = output_dir / f"{scenario['name']}.csv"
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_s', 'acceleration_ms2'])
            writer.writerows(data)

        # Print statistics
        actual_pga, actual_pga_g, rms = calculate_stats(data)

        print(f"  Actual PGA: {actual_pga_g:.3f}g ({actual_pga:.2f} m/s²)")
        print(f"  RMS: {rms:.2f} m/s²")
        print(f"  Samples: {len(data)}")
        print(f"  Saved to: {output_file}")
        print()

    # Create README
    readme_path = output_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write("# PEER-Style Synthetic Earthquake Ground Motions\n\n")
        f.write("## Overview\n\n")
        f.write("Synthetic earthquake ground motions with realistic characteristics:\n\n")
        f.write("- Appropriate frequency content for magnitude\n")
        f.write("- Realistic envelope (buildup, strong motion, decay)\n")
        f.write("- Multiple frequency components\n")
        f.write("- Applied perturbations (noise, dropout, delay)\n\n")

        f.write("## Scenarios\n\n")
        f.write("### With Perturbations (10% noise, 60ms delay, 8% dropout)\n\n")
        for scenario in scenarios_perturbed:
            f.write(f"**{scenario['name']}**\n")
            f.write(f"- Description: {scenario['description']}\n")
            f.write(f"- Magnitude: {scenario['magnitude']}\n")
            f.write(f"- PGA: {scenario['pga_g']}g ({scenario['pga_g']*9.81:.2f} m/s²)\n")
            f.write(f"- Duration: {scenario['duration']}s\n\n")

        f.write("### Clean Baseline (No Perturbations)\n\n")
        f.write(f"**{scenario_clean['name']}**\n")
        f.write(f"- Description: {scenario_clean['description']}\n")
        f.write(f"- Magnitude: {scenario_clean['magnitude']}\n")
        f.write(f"- PGA: {scenario_clean['pga_g']}g ({scenario_clean['pga_g']*9.81:.2f} m/s²)\n")
        f.write(f"- Duration: {scenario_clean['duration']}s\n")
        f.write(f"- Use this for testing different perturbation configurations in MATLAB\n\n")

        f.write("## Perturbations\n\n")
        f.write("Perturbed datasets include:\n")
        f.write(f"- **Noise:** 10% of PGA (sensor noise + site effects)\n")
        f.write(f"- **Delay:** 60ms (communication latency)\n")
        f.write(f"- **Dropout:** 8% (packet loss with hold-last-value)\n\n")
        f.write("Clean dataset has NO perturbations for custom testing.\n\n")

        f.write("## CSV Format\n\n")
        f.write("Columns:\n")
        f.write("- `time_s`: Time in seconds\n")
        f.write("- `acceleration_ms2`: Ground acceleration in m/s²\n\n")

        f.write("## Usage in MATLAB\n\n")
        f.write("```matlab\n")
        f.write("% Load earthquake data\n")
        f.write("data = readtable('PEER_moderate_M5.7_PGA0.35g.csv');\n")
        f.write("ag = data.acceleration_ms2';\n")
        f.write("dt = 0.02;\n")
        f.write("```\n")

    print(f"✅ All earthquakes generated successfully!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 README created: {readme_path}")


if __name__ == "__main__":
    main()
