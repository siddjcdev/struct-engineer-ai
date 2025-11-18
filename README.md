# Struct Engineer AI: TMD Optimization for Multi-Hazard Loading

**2026 Chester County Science and Research Fair Project**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

## 🎯 Project Overview

This project investigates the effectiveness of **Tuned Mass Dampers (TMDs)** in reducing structural response under combined seismic and wind loading conditions. Using real earthquake data and synthetic wind profiles, we optimize TMD parameters (mass ratio, damping ratio, and floor location) to minimize structural demand.

### Research Question
**"How effective are passive TMDs in mitigating structural response under extreme multi-hazard loading scenarios?"**

## 🏗️ What is a Tuned Mass Damper?

A Tuned Mass Damper (TMD) is a passive vibration control device consisting of:
- A large mass (typically 1-10% of building mass)
- Spring and damper elements
- Tuned to the building's natural frequency

**How it works**: When the building vibrates, the TMD oscillates out-of-phase, absorbing energy and reducing structural motion.

## 📊 Experimental Design

### Test Cases
We evaluated 6 distinct loading scenarios:

| Test | Loading Type | Intensity | Key Metric |
|------|-------------|-----------|------------|
| **Test 1** | Stationary Wind + Earthquake | 12 m/s + 0.35g | Best performer |
| **Test 2** | Turbulent Wind + Earthquake | 25 m/s + 0.35g | Moderate performance |
| **Test 3** | Small Earthquake Only | M 4.5 (0.10g) | Limited improvement |
| **Test 4** | Large Earthquake Only | M 6.9 (0.40g) | Minimal improvement |
| **Test 5** | Extreme Combined | 50 m/s + 0.40g | TMD ineffective |
| **Test 6** | Noisy Data | 0.39g + 10% noise | Algorithm stability test |

### Performance Metrics
- **DCR (Demand-Capacity Ratio)**: Primary metric - measures structural safety
- **Max Inter-Story Drift**: Critical for non-structural damage
- **Max Roof Displacement**: Overall building motion
- **Average Acceleration**: Occupant comfort

## 🔬 Key Findings

### 1. Loading Intensity vs. TMD Effectiveness
```
Test 1 (Moderate):  17.6% DCR reduction ✅
Test 2 (High):       7.2% DCR reduction ⚠️
Test 5 (Extreme):    0.2% DCR reduction ❌
```

**Conclusion**: TMD effectiveness decreases exponentially with loading intensity.

### 2. Optimal TMD Placement Patterns
- **Low wind speeds** → Upper floors (Floor 9)
- **High wind speeds** → Lower floors (Floor 2)
- **Large earthquakes** → Mid-upper floors (Floor 8)

### 3. Trade-offs Observed
In some cases, **roof displacement increased** while **DCR decreased**:
- This is expected: TMDs absorb energy locally but protect critical structural elements
- DCR reduction confirms successful structural protection

## 📁 Repository Structure

```
struct-engineer-ai/
├── data/
│   ├── TEST1_stationary_wind_12ms.csv
│   ├── TEST2_turbulent_wind_25ms.csv
│   ├── TEST3_small_earthquake_M4.5.csv
│   ├── TEST4_large_earthquake_M6.9.csv
│   ├── TEST5_earthquake_M6.7.csv
│   ├── TEST5_hurricane_wind_50ms.csv
│   └── TEST6b_with_10pct_noise.csv
├── results/
│   ├── tmd_simulation_*.json (6 result files)
│   └── building_sim_latest.mat
├── notebooks/
│   └── TMD_Analysis.ipynb
├── scripts/
│   ├── visualize_results.py
│   ├── compare_tests.py
│   └── generate_report.py
├── docs/
│   ├── METHODOLOGY.md
│   ├── RESULTS.md
│   └── THEORY.md
├── README.md
└── requirements.txt
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scipy
```

### Run Visualization
```bash
python scripts/visualize_results.py
```

### Generate Comparison Report
```bash
python scripts/compare_tests.py
```

## 📈 Visualizations

Our analysis includes:
1. **DCR Reduction Comparison** - Bar chart across all tests
2. **TMD Mass vs. Performance** - Scatter plot showing optimization
3. **Time-History Plots** - Baseline vs. TMD response
4. **Floor Location Heatmap** - Optimal TMD placement by loading type
5. **Performance Trade-offs** - Drift reduction vs. displacement changes

## 🎓 Educational Value

This project demonstrates:
- **Physics**: Harmonic oscillators, resonance, energy dissipation
- **Engineering**: Structural dynamics, vibration control, multi-hazard design
- **Computer Science**: Optimization algorithms, data analysis, API development
- **Mathematics**: Differential equations, numerical integration, statistical analysis

## 🔮 Future Work

1. **Multiple TMDs**: Test 2-3 TMDs at different floors for extreme events
2. **Active Control**: Compare passive TMD to semi-active/active systems
3. **Cost-Benefit Analysis**: Economic feasibility study
4. **Real-Time Implementation**: Deploy on physical shake table

## 📚 References

1. Den Hartog, J.P. (1956). *Mechanical Vibrations*. McGraw-Hill.
2. Soong, T.T., & Dargush, G.F. (1997). *Passive Energy Dissipation Systems in Structural Engineering*. Wiley.
3. Elias, S., & Matsagar, V. (2017). Research developments in vibration control of structures using passive tuned mass dampers. *Annual Reviews in Control*.

## 👤 Author

**[Your Name]**  
Chester County Science and Research Fair 2026

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- El Centro earthquake data from PEER Ground Motion Database
- Synthetic wind profiles generated using Kaimal spectrum
- Building model based on ASCE 7-16 design standards

---

**⭐ If this project helps your research, please give it a star!**
