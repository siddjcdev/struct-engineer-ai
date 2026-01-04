# PEER-Style Synthetic Earthquake Ground Motions

## Overview

Synthetic earthquake ground motions with realistic characteristics:

- Appropriate frequency content for magnitude
- Realistic envelope (buildup, strong motion, decay)
- Multiple frequency components
- Applied perturbations (noise, dropout, delay)

## Scenarios

### Base Earthquakes (with combined perturbations)

### Moderate Earthquake Variants (different perturbations)

**PEER_moderate_60ms_latency**
- Description: Moderate (M4.5) + 40ms Latency ONLY
- Perturbations: 40ms delay

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
