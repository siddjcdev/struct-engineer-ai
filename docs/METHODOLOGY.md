# TMD Simulation Methodology

## Overview

This document describes the methodology used to optimize Tuned Mass Damper (TMD) parameters for a 12-story building under combined seismic and wind loading.

## 1. Building Model

### 1.1 Structural Configuration
- **Type**: 12-story steel frame building
- **Story Height**: 3.5 meters per floor
- **Total Height**: 42 meters
- **Design Standard**: ASCE 7-16

### 1.2 Structural Properties
The building is modeled as a multi-degree-of-freedom (MDOF) system with:
- Lumped masses at each floor level
- Elastic story stiffnesses
- Inherent structural damping

### 1.3 Governing Equations

The equation of motion for the building-TMD system is:

```
[M]{ü} + [C]{u̇} + [K]{u} = {F(t)}
```

Where:
- `[M]` = Mass matrix (building + TMD)
- `[C]` = Damping matrix
- `[K]` = Stiffness matrix
- `{u}` = Displacement vector
- `{F(t)}` = Time-varying force vector (seismic + wind)

## 2. TMD Design Parameters

### 2.1 Mass Ratio (μ)
```
μ = m_TMD / m_building
```

Tested range: 4% to 30% of building mass

**Theoretical Basis**: Larger mass ratios generally provide better control but increase costs and structural requirements.

### 2.2 Frequency Ratio
The TMD natural frequency is tuned to the building's fundamental frequency:

```
ω_TMD = ω_building / (1 + μ)
```

This is based on Den Hartog's optimal tuning formula.

### 2.3 Damping Ratio (ζ)
Tested range: 5% to 49% of critical damping

Optimal damping ratio from Den Hartog:
```
ζ_opt = √(3μ / (8(1 + μ)))
```

### 2.4 Location
TMDs are tested at multiple floor locations to find the optimal placement that maximizes energy dissipation.

## 3. Loading Conditions

### 3.1 Seismic Loading

**Earthquake Records Used**:
- El Centro (1940) - Imperial Valley earthquake
- Synthetic records matching design spectra

**Ground Motion Processing**:
1. Baseline correction
2. High-pass filtering (0.1 Hz cutoff)
3. Acceleration integration to velocity and displacement
4. Scaling to target Peak Ground Acceleration (PGA)

**Application**:
```
F_seismic(floor, t) = -m_floor * a_ground(t) * φ(floor)
```
Where φ(floor) is the mode shape coefficient.

### 3.2 Wind Loading

**Wind Speed Profile**:
```
V(z) = V_ref * (z / z_ref)^α
```
Where:
- α = 0.15 (terrain roughness exponent for suburban areas)
- z = height above ground
- V_ref = reference wind speed at z_ref

**Turbulence Generation**:
Wind velocity fluctuations are generated using the Kaimal spectrum:
```
S(f) = (200 * σ²_u * L_u / V) / (1 + 50 * f * L_u / V)^(5/3)
```

**Wind Force Calculation**:
```
F_wind(floor, t) = 0.5 * ρ * Cd * A_floor * V²(z, t)
```
Where:
- ρ = air density (1.225 kg/m³)
- Cd = drag coefficient (1.3 for rectangular buildings)
- A_floor = exposed area per floor

### 3.3 Combined Loading
For tests with both seismic and wind loading:
```
F_total(floor, t) = F_seismic(floor, t) + F_wind(floor, t)
```

## 4. Optimization Algorithm

### 4.1 Search Space
The algorithm searches over:
- **Floor locations**: Candidate floors based on modal analysis
- **Mass ratios**: 4%, 6%, 10%, 11%, 24%, 25%, 30%
- **Damping ratios**: 5%, 9%, 49%

### 4.2 Objective Function
The primary objective is to minimize the Demand-Capacity Ratio (DCR):

```
DCR = max(|drift_i| / drift_allowable)
```

Where drift_allowable is typically H/400 to H/500 (H = story height).

### 4.3 Candidate Floor Selection
Floors are selected based on mode shape analysis:
- Floors with high modal displacement
- Typically upper floors for first mode
- Adjusted for combined loading scenarios

### 4.4 Grid Search Procedure
1. For each candidate floor:
   2. For each mass ratio:
      3. For each damping ratio:
         4. Run time-history simulation
         5. Calculate DCR and other metrics
         6. Track best configuration
7. Return optimal TMD parameters

## 5. Numerical Integration

### 5.1 Time-History Analysis
The equations of motion are integrated using the Newmark-β method:

**Algorithm**:
```
u_{n+1} = u_n + Δt * u̇_n + (Δt²/2) * ((1-2β) * ü_n + 2β * ü_{n+1})
u̇_{n+1} = u̇_n + Δt * ((1-γ) * ü_n + γ * ü_{n+1})
```

**Parameters**:
- β = 1/4 (average acceleration method)
- γ = 1/2
- Δt = 0.02 seconds (50 Hz sampling rate)

### 5.2 Stability
The average acceleration method is unconditionally stable for linear systems.

## 6. Performance Metrics

### 6.1 Demand-Capacity Ratio (DCR)
```
DCR = max(drift_i / drift_capacity_i) for all stories
```

Primary metric for structural safety. DCR < 1.0 indicates elastic response.

### 6.2 Maximum Inter-Story Drift
```
drift_i = |u_i - u_{i-1}| / h_i
```

Critical for non-structural damage assessment.

### 6.3 Maximum Roof Displacement
```
u_roof = max(|u_{top_floor}(t)|)
```

Indicates overall building motion.

### 6.4 RMS Metrics
```
RMS_displacement = √(1/T ∫ u²(t) dt)
RMS_velocity = √(1/T ∫ u̇²(t) dt)
RMS_acceleration = √(1/T ∫ ü²(t) dt)
```

Provide average response characteristics.

## 7. Test Case Scenarios

### Test 1: Stationary Wind + Earthquake
- **Purpose**: Evaluate TMD under typical combined loading
- **Wind**: 12 m/s mean, low turbulence
- **Seismic**: El Centro (0.35g PGA)

### Test 2: Turbulent Wind + Earthquake
- **Purpose**: Assess performance under higher wind turbulence
- **Wind**: 25 m/s mean, high turbulence intensity
- **Seismic**: El Centro (0.35g PGA)

### Test 3: Small Earthquake
- **Purpose**: Evaluate TMD effectiveness for low-intensity seismic events
- **Magnitude**: M 4.5 (0.10g PGA)

### Test 4: Large Earthquake
- **Purpose**: Test TMD limits under major seismic event
- **Magnitude**: M 6.9 (0.40g PGA)

### Test 5: Extreme Combined Loading
- **Purpose**: Determine TMD performance ceiling
- **Wind**: 50 m/s (hurricane conditions)
- **Seismic**: M 6.7 (0.40g PGA)

### Test 6: Robustness Test
- **Purpose**: Verify algorithm stability with noisy data
- **Input**: 10% white noise added to ground motion

## 8. Validation and Verification

### 8.1 Code Verification
- Unit tests for structural dynamics functions
- Comparison with analytical solutions for SDOF systems
- Energy conservation checks

### 8.2 Physical Validation
- Results compared to published research on TMD effectiveness
- Frequency response functions verified against theory
- Modal analysis validated

## 9. Limitations and Assumptions

### 9.1 Assumptions
1. Linear elastic structural behavior
2. Rigid floor diaphragms
3. No soil-structure interaction
4. No torsional effects
5. TMD operates as ideal oscillator (no friction/stops)

### 9.2 Limitations
1. Does not model structural yielding or damage
2. Wind-structure interaction simplified
3. No consideration of construction constraints
4. Cost analysis not included

## 10. References

1. Den Hartog, J.P. (1956). *Mechanical Vibrations*. 4th Edition, McGraw-Hill.
2. Soong, T.T., & Dargush, G.F. (1997). *Passive Energy Dissipation Systems in Structural Engineering*. Wiley.
3. Warburton, G.B. (1982). "Optimum absorber parameters for various combinations of response and excitation parameters." *Earthquake Engineering & Structural Dynamics*, 10(3), 381-401.
4. Elias, S., & Matsagar, V. (2017). "Research developments in vibration control of structures using passive tuned mass dampers." *Annual Reviews in Control*, 44, 129-156.
5. ASCE 7-16. *Minimum Design Loads and Associated Criteria for Buildings and Other Structures*. American Society of Civil Engineers.
6. Simiu, E., & Scanlan, R.H. (1996). *Wind Effects on Structures*. 3rd Edition, Wiley.

---

**Document Version**: 1.0  
**Last Updated**: November 2025  
**Author**: 2026 Chester County Science Fair Project
