# PEER-Style Synthetic Earthquake Ground Motions

## Overview

Synthetic earthquake ground motions with realistic characteristics:

- Appropriate frequency content for magnitude
- Realistic envelope (buildup, strong motion, decay)
- Multiple frequency components
- Applied perturbations (noise, dropout, delay)

## Scenarios

### Base Earthquakes (with combined perturbations)

**PEER_small_M4.5_PGA0.25g**
- Description: Small Earthquake - Magnitude 4.5 (0.25g PGA)
- Magnitude: 4.5
- PGA: 0.25g (2.45 m/s²)
- Duration: 20.0s
- Perturbations: 10% noise, 60ms delay, 8% dropout

**PEER_moderate_M5.7_PGA0.35g**
- Description: Moderate Earthquake - Magnitude 5.7 (0.35g PGA)
- Magnitude: 5.7
- PGA: 0.35g (3.43 m/s²)
- Duration: 40.0s
- Perturbations: 10% noise, 60ms delay, 8% dropout

**PEER_high_M7.4_PGA0.75g**
- Description: High Magnitude Earthquake - 7.4 (0.75g PGA)
- Magnitude: 7.4
- PGA: 0.75g (7.36 m/s²)
- Duration: 80.0s
- Perturbations: 10% noise, 60ms delay, 8% dropout

**PEER_insane_M8.4_PGA0.9g**
- Description: Insane Magnitude Earthquake - 8.4 (0.9g PGA)
- Magnitude: 8.4
- PGA: 0.9g (8.83 m/s²)
- Duration: 120.0s
- Perturbations: 10% noise, 60ms delay, 8% dropout

### Moderate Earthquake Variants (different perturbations)

**PEER_moderate_10pct_noise**
- Description: Moderate (M5.7) + 10% Noise ONLY
- Perturbations: 10.0% noise

**PEER_moderate_60ms_latency**
- Description: Moderate (M5.7) + 60ms Latency ONLY
- Perturbations: 60ms delay

**PEER_moderate_8pct_dropout**
- Description: Moderate (M5.7) + 8% Dropout ONLY
- Perturbations: 8.0% dropout

**PEER_moderate_combined_stress**
- Description: Moderate (M5.7) + ALL Perturbations
- Perturbations: 10.0% noise, 60ms delay, 8.0% dropout

## Perturbation Types

- **Noise:** Random Gaussian noise added to ground motion (% of PGA)
- **Delay:** Simulated communication latency (milliseconds)
- **Dropout:** Random data packet loss with hold-last-value (% of samples)

Use moderate variants to isolate the effect of each perturbation type.

## CSV Format

Columns:
- `time_s`: Time in seconds
- `acceleration_ms2`: Ground acceleration in m/s²

## Usage in MATLAB

```matlab
% Load earthquake data
data = readtable('PEER_moderate_M5.7_PGA0.35g.csv');
ag = data.acceleration_ms2';
dt = 0.02;
```
