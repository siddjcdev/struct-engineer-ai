# struct-engineer-ai

2026 Chester County Science and Research Fair Project

## Overview

This repository contains MATLAB code used to explore model-updating / parameter-tuning methods for detecting and quantifying changes (damage) in simple multi-degree-of-freedom floor/structure systems using measured dynamic responses.

The codebase includes tools to create synthetic test datasets, convert real sensor data to a common sampling rate, run an automatic tuner that estimates changes in stiffness/damping, and an end-to-end runner that evaluates multiple test cases.

## Theory and approach

1. Structural dynamics background
   - Buildings and floor systems can be approximated as multi-degree-of-freedom (MDOF) mass-spring-damper systems. Each floor has an effective mass m, stiffness k, and damping c. External excitation (ambient vibration, impact, seismic, etc.) causes the structure to vibrate in characteristic ways.
   - The undamped natural frequency of a single-degree-of-freedom system is fn = (1/2π) * sqrt(k/m). For MDOF systems, the same principles apply: the system has a set of natural frequencies and mode shapes determined by mass and stiffness distributions.
   - Damage (e.g., cracking, loosened connections, stiffness loss) typically reduces local stiffness. That leads to measurable changes in modal properties: natural frequencies shift (usually downward), modal damping and mode shapes may change, and frequency-domain transfer functions or time-domain responses exhibit altered signatures.

2. Detection by model updating / tuning
   - Instead of attempting to directly invert noisy sensor signals for damage, this project uses a model-updating or parameter-tuning approach: define a parametric model of the structure (for example, stiffness reductions per floor) and search for the parameter set that makes the model's predicted response best match measured data.
   - The tuning algorithm implemented in the project compares measured and simulated dynamic responses (time histories, spectral amplitudes, or other features) and minimizes a cost function (commonly a least-squares difference or related metric). The resulting best-fit parameters indicate the likely location(s) and severity of stiffness changes.
   - This approach leverages physical insight (the mass-stiffness-damping model) while remaining robust to noise by fitting across the whole response rather than relying on a single feature.

3. Practical considerations used in the code
   - Resampling / common sampling frequency: real sensor datasets may come at different sampling rates. For consistent analysis, measured signals are resampled to a common rate (50 Hz in this project) — see convert_real_data_to_50Hz.m.
   - Synthetic test cases: create_all_6_test_datasets.m generates controlled datasets with known damage scenarios and noise levels. These allow validation of the tuner and estimation of detection limits.
   - Tuning function: thefunc_dcr_floor_tuner_v6.m contains the core parameter tuning / model updating algorithm. It parametrizes damage in terms of per-floor stiffness changes and searches (via an optimization routine) for parameter values that minimize the mismatch between measured and modeled responses.
   - End-to-end evaluation: RUN_ALL_6_TESTS_WITH_DATA.m runs the full experiment suite for six test scenarios, loads or generates datasets, runs the tuner, and collects results for comparison.

## Files of interest

- RUN_ALL_6_TESTS_WITH_DATA.m — end-to-end runner for the six test datasets; executes data loading/creation, preprocessing, tuning, and result aggregation.
- create_all_6_test_datasets.m — script that synthesizes the six test datasets (and optionally prepares or packages real data) used during evaluation.
- convert_real_data_to_50Hz.m — helper to resample real sensor data to a uniform 50 Hz sampling rate and apply basic preprocessing steps needed by the tuner.
- thefunc_dcr_floor_tuner_v6.m — core model-updating / tuning implementation; estimates per-floor stiffness/damping changes by minimizing a cost function between measured and modeled responses.

## Usage

1. Requirements
   - MATLAB (a recent release recommended)
   - Signal Processing Toolbox is helpful (for resampling, filtering, PSD)
   - The repository expects acceleration time-series data in the dataset format prepared by create_all_6_test_datasets.m / convert_real_data_to_50Hz.m.

2. Typical workflow
   - Use convert_real_data_to_50Hz.m to standardize raw recordings to 50 Hz if needed.
   - Run create_all_6_test_datasets.m to assemble datasets for testing.
   - Run RUN_ALL_6_TESTS_WITH_DATA.m to execute the six experiments and collect results.
   - Inspect outputs (figures, numeric results) saved by the scripts to evaluate identified modal properties and tuning outcomes.
1. Open MATLAB and add this repository folder to the MATLAB path (or change the working directory to the repository root).
2. (Optional) Convert any real data you want to use to 50 Hz with:
   - convert_real_data_to_50Hz
3. (Optional) Generate the synthetic test datasets (overwrites/creates files in data/ or local workspace):
   - create_all_6_test_datasets
4. Run the full test suite and see results:
   - RUN_ALL_6_TESTS_WITH_DATA
5. To inspect or tune behavior, open the tuning function and scripts and adjust parameters such as optimization options, noise levels, or which features are compared in the cost function.

## Notes and assumptions

- The methods demonstrated are aimed at small educational MDOF systems and controlled test cases. Real buildings are more complex, and additional considerations (nonlinearities, temperature effects, sensor placement, operational variability) are necessary for field deployment.
- Results depend strongly on model fidelity (how well the numerical model matches the true system) and signal quality.
- Toolboxes: some functions may use MATLAB built-in functions that require Signal Processing Toolbox or Optimization Toolbox. If you encounter missing functions, check which toolbox provides them or replace with custom equivalents.

## Contact / Next steps

If you want further help documenting functions in-line, adding plots or examples to the README, or preparing an experiment report, tell me which outputs (figures, sample datasets) you'd like included and I will update the repository accordingly.

# Training
Modified to work seamlessly with the aggregated file and provide helpful feedback when using it. Example workflow:

## Step 1: Create aggregated training file
This script takes 80% of data from all earthquake files and combines them into a single aggregated training file. Key features:
- Takes 80% from the beginning of each earthquake file
- Combines all data into a single CSV file
- Provides detailed statistics about the aggregation
- Default output: matlab/datasets aggregated_train_80pct.csv
Usage:
python rl/rl_cl/create_aggregated_training_file.py \
  --earthquakes matlab/datasets/*.csv \
  --output matlab/datasets/aggregated_train_80pct.csv

python rl/rl_cl/create_aggregated_training_file.py \
  --earthquakes matlab/datasets/TEST3_small_earthquake_M4.5.csv \
                matlab/datasets/TEST4_large_earthquake_M6.9.csv \
                matlab/datasets/TEST5_earthquake_M6.7.csv \
  --output matlab/datasets/aggregated_train_80pct.csv

## Step 2: Train on the aggregated file
python rl/rl_cl/train_rl_cl.py \
  --earthquakes matlab/datasets/aggregated_train_80pct.csv