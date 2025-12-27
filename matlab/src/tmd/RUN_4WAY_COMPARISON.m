% ============================================================================
% 4-WAY TMD CONTROLLER COMPARISON
% ============================================================================
%
% Compares FOUR controllers across 8 test scenarios:
%   1. Passive TMD (mechanical, optimized)
%   2. Fuzzy Logic Control (REST API - batch prediction)
%   3. RL Baseline (REST API - FULL SIMULATION)
%   4. RL with Curriculum Learning (REST API - FULL SIMULATION)
%
% IMPORTANT: Both RL controllers use /simulate endpoints which run the full
% simulation in their respective Python environments and return comprehensive
% metrics (DCR, RMS, max drift, forces). This ensures both RL controllers
% are tested with the same physics they were trained on.
%
% Test Scenarios:
%   - TEST3: Small Earthquake (M4.5)
%   - TEST4: Large Earthquake (M6.9)
%   - TEST5: Moderate Earthquake (M6.7)
%   - TEST6a: Baseline (clean)
%   - TEST6b: 10% sensor noise
%   - TEST6c: 50ms communication latency
%   - TEST6d: 5% data dropout
%   - TEST6e: Combined stress
%
% Prerequisites:
%   - Earthquake CSV files in '../../datasets/' folder
%   - REST API deployed with Fuzzy, RL, and RL_CL endpoints
%   - API URL configured below
%
% Author: Modified from tmd_comparision.m
% Date: December 2025
% ============================================================================

function RUN_4WAY_COMPARISON()

    clear all; close all; clc;

    fprintf('\n╔════════════════════════════════════════════════════════╗\n');
    fprintf('║  4-WAY TMD CONTROLLER COMPARISON TEST                  ║\n');
    fprintf('╚════════════════════════════════════════════════════════╝\n\n');

    % ========== UPDATE THIS WITH YOUR API URL ==========
    % IMPORTANT: Must end with trailing slash '/'
    % Examples:
    API_URL = 'http://localhost:8080/';
    %API_URL = 'https://perfect-rl-api-887344515766.us-east4.run.app/';
    % ===================================================

    % Auto-fix: Add trailing slash if missing
    if ~endsWith(API_URL, '/')
        API_URL = [API_URL '/'];
        fprintf('ℹ️  Added trailing slash to API_URL: %s\n', API_URL);
    end

    % Building parameters (matching your v7/v8 setup)
    N = 12;                      % 12 floors
    m0 = 2.0e5;                  % kg per floor
    k0 = 2.0e7;                  % N/m stiffness
    zeta_target = 0.015;         % 1.5% damping
    dt = 0.01;                   % 10ms time step
    soft_story_idx = 8;          % 8th floor is soft
    soft_story_factor = 0.60;    % 60% stiffness

    % TMD parameters
    mu_tmd = 0.02;               % 2% mass ratio
    tmd_mass = mu_tmd * m0;      % TMD mass

    % Test scenarios (using ../../datasets/ path)
    folder = fullfile('..', '..', 'datasets');

    % Check prerequisites
    fprintf('🔍 Checking prerequisites...\n\n');

    % Check API connection
    fprintf('  [1/2] Checking API connection... ');
    try
        options = weboptions('Timeout', 10);
        health = webread([API_URL 'health'], options);
        fprintf('✅\n');
        fprintf('        Status: %s\n', health.status);
    catch ME
        fprintf('❌ FAILED\n');
        fprintf('        Error: %s\n', ME.message);
        fprintf('\n⚠️  UPDATE API_URL in this script!\n\n');
        error('Cannot connect to API. Please fix and retry.');
    end

    % Check earthquake files
    fprintf('  [2/2] Checking earthquake files... ');
    if ~check_datasets_exist(folder)
        fprintf('❌ MISSING\n');
        error('Earthquake files missing in %s', folder);
    else
        fprintf('✅\n');
    end

    fprintf('\n✓ All prerequisites satisfied!\n\n');

    % Menu
    fprintf('Select test mode:\n');
    fprintf('  1. Quick demo (TEST3 only, ~1 min)\n');
    fprintf('  2. Run all 8 scenarios (~10 min)\n');
    fprintf('  3. Run specific scenario\n');
    fprintf('  4. Robustness tests only (TEST6a-e)\n\n');

    choice = input('Enter choice (1-4): ');

    switch choice
        case 1
            run_quick_demo(API_URL, folder, N, m0, k0, zeta_target, dt, soft_story_idx, soft_story_factor, tmd_mass);
        case 2
            run_all_scenarios(API_URL, folder, N, m0, k0, zeta_target, dt, soft_story_idx, soft_story_factor, tmd_mass);
        case 3
            run_specific_scenario(API_URL, folder, N, m0, k0, zeta_target, dt, soft_story_idx, soft_story_factor, tmd_mass);
        case 4
            run_robustness_tests(API_URL, folder, N, m0, k0, zeta_target, dt, soft_story_idx, soft_story_factor, tmd_mass);
        otherwise
            fprintf('Invalid choice. Running quick demo...\n');
            run_quick_demo(API_URL, folder, N, m0, k0, zeta_target, dt, soft_story_idx, soft_story_factor, tmd_mass);
    end

    fprintf('\n✓ All tests complete!\n\n');
end

%% ============================================================
%% QUICK DEMO (SMALL EARTHQUAKE ONLY)
%% ============================================================
function run_quick_demo(API_URL, folder, N, m0, k0, zeta_target, dt, soft_story_idx, soft_story_factor, tmd_mass)
    fprintf('\n═══ QUICK DEMO: SMALL EARTHQUAKE ONLY ═══\n\n');

    % Single scenario in proper row format (1x4 cell array)
    % Note: Perturbations already baked into PEER datasets (10% noise, 60ms delay, 8% dropout)
    scenarios = {
        'PEER_Small', 'Small (M4.5, 0.25g)', fullfile(folder, 'PEER_small_M4.5_PGA0.25g.csv'), struct('noise', 0, 'latency', 0, 'dropout', 0);
    };

    results = run_comparison(API_URL, scenarios, N, m0, k0, zeta_target, dt, soft_story_idx, soft_story_factor, tmd_mass);

    % Display and save
    display_results_table(results, scenarios);
    save_results(results, scenarios, API_URL, 'quick_demo');

    fprintf('\n═══ QUICK DEMO COMPLETE ═══\n');
end

%% ============================================================
%% RUN ALL 8 SCENARIOS (4 BASE + 4 MODERATE PERTURBATION TESTS)
%% ============================================================
function run_all_scenarios(API_URL, folder, N, m0, k0, zeta_target, dt, soft_story_idx, soft_story_factor, tmd_mass)
    fprintf('\n═══ COMPREHENSIVE 4-WAY COMPARISON (8 SCENARIOS) ═══\n\n');

   % Perturbations are BAKED INTO the CSV files (not applied in MATLAB)
    % This is required because /simulate endpoints run in Python and don't receive MATLAB perturbation parameters
    scenarios = {
        % === 4 Base Earthquakes (with baked-in perturbations) ===
        'PEER_Small',    'Small (M4.5, 0.25g)',          fullfile(folder, 'PEER_small_M4.5_PGA0.25g.csv'),         struct('noise', 0, 'latency', 0, 'dropout', 0);
        'PEER_Moderate', 'Moderate (M5.7, 0.35g)',       fullfile(folder, 'PEER_moderate_M5.7_PGA0.35g.csv'),     struct('noise', 0, 'latency', 0, 'dropout', 0);
        'PEER_High',     'High (M7.4, 0.75g)',           fullfile(folder, 'PEER_high_M7.4_PGA0.75g.csv'),         struct('noise', 0, 'latency', 0, 'dropout', 0);
        'PEER_Insane',   'Insane (M8.4, 0.9g)',          fullfile(folder, 'PEER_insane_M8.4_PGA0.9g.csv'),        struct('noise', 0, 'latency', 0, 'dropout', 0);

        % === 4 Moderate Tests (different perturbations baked into CSV files) ===
        'Mod_10Noise',   'Moderate + 10% Noise',         fullfile(folder, 'PEER_moderate_10pct_noise.csv'),        struct('noise', 0, 'latency', 0, 'dropout', 0);
        'Mod_60Latency', 'Moderate + 60ms Latency',      fullfile(folder, 'PEER_moderate_60ms_latency.csv'),       struct('noise', 0, 'latency', 0, 'dropout', 0);
        'Mod_8Dropout',  'Moderate + 8% Dropout',        fullfile(folder, 'PEER_moderate_8pct_dropout.csv'),       struct('noise', 0, 'latency', 0, 'dropout', 0);
        'Mod_Combined',  'Moderate + Combined Stress',   fullfile(folder, 'PEER_moderate_combined_stress.csv'),    struct('noise', 0, 'latency', 0, 'dropout', 0);
    };

    results = run_comparison(API_URL, scenarios, N, m0, k0, zeta_target, dt, soft_story_idx, soft_story_factor, tmd_mass);

    % Display and save
    display_results_table(results, scenarios);
    create_comparison_plots(results, scenarios);
    fprintf('  ✓ Saved: 4way_comparison_plots.png\n\n');
    create_analysis_plots(results, scenarios);
    fprintf('  ✓ Saved: 4way_analysis_plots.png\n\n');
    save_results(results, scenarios, API_URL, 'comprehensive');

    fprintf('\n═══ COMPREHENSIVE TEST COMPLETE ═══\n');
end

%% ============================================================
%% RUN SPECIFIC SCENARIO
%% ============================================================
function run_specific_scenario(API_URL, folder, N, m0, k0, zeta_target, dt, soft_story_idx, soft_story_factor, tmd_mass)
    fprintf('\n═══ SELECT SCENARIO ═══\n\n');
    fprintf('  BASE EARTHQUAKES:\n');
    fprintf('  1. Small: M4.5, PGA 0.25g (20s)\n');
    fprintf('  2. Moderate: M5.7, PGA 0.35g (40s)\n');
    fprintf('  3. High: M7.4, PGA 0.75g (80s)\n');
    fprintf('  4. Insane: M8.4, PGA 0.9g (120s)\n\n');
    fprintf('  MODERATE + PERTURBATIONS:\n');
    fprintf('  5. Moderate + 10%% Noise\n');
    fprintf('  6. Moderate + 60ms Latency\n');
    fprintf('  7. Moderate + 8%% Dropout\n');
    fprintf('  8. Moderate + Combined Stress\n\n');

    test_num = input('Enter test number (1-8): ');

    all_scenarios = {
        % === 4 Base Earthquakes (with baked-in perturbations) ===
        'PEER_Small',    'Small (M4.5, 0.25g)',          fullfile(folder, 'PEER_small_M4.5_PGA0.25g.csv'),         struct('noise', 0, 'latency', 0, 'dropout', 0);
        'PEER_Moderate', 'Moderate (M5.7, 0.35g)',       fullfile(folder, 'PEER_moderate_M5.7_PGA0.35g.csv'),     struct('noise', 0, 'latency', 0, 'dropout', 0);
        'PEER_High',     'High (M7.4, 0.75g)',           fullfile(folder, 'PEER_high_M7.4_PGA0.75g.csv'),         struct('noise', 0, 'latency', 0, 'dropout', 0);
        'PEER_Insane',   'Insane (M8.4, 0.9g)',          fullfile(folder, 'PEER_insane_M8.4_PGA0.9g.csv'),        struct('noise', 0, 'latency', 0, 'dropout', 0);

         % === 4 Moderate Tests (different perturbations baked into CSV files) ===
        'Mod_10Noise',   'Moderate + 10% Noise',         fullfile(folder, 'PEER_moderate_10pct_noise.csv'),        struct('noise', 0, 'latency', 0, 'dropout', 0);
        'Mod_60Latency', 'Moderate + 60ms Latency',      fullfile(folder, 'PEER_moderate_60ms_latency.csv'),       struct('noise', 0, 'latency', 0, 'dropout', 0);
        'Mod_8Dropout',  'Moderate + 8% Dropout',        fullfile(folder, 'PEER_moderate_8pct_dropout.csv'),       struct('noise', 0, 'latency', 0, 'dropout', 0);
        'Mod_Combined',  'Moderate + Combined Stress',   fullfile(folder, 'PEER_moderate_combined_stress.csv'),    struct('noise', 0, 'latency', 0, 'dropout', 0);
    };

    if test_num < 1 || test_num > 8
        fprintf('Invalid choice.\n');
        return;
    end

    scenarios = all_scenarios(test_num, :);

    results = run_comparison(API_URL, scenarios, N, m0, k0, zeta_target, dt, soft_story_idx, soft_story_factor, tmd_mass);

    display_results_table(results, scenarios);
    save_results(results, scenarios, API_URL, scenarios{1,1});
end

%% ============================================================
%% ROBUSTNESS TESTS (TEST6a-e)
%% ============================================================
function run_robustness_tests(API_URL, folder, N, m0, k0, zeta_target, dt, soft_story_idx, soft_story_factor, tmd_mass)
    fprintf('\n═══ ROBUSTNESS TESTS (TEST6a-e) ═══\n\n');

    scenarios = {
        'TEST6a', 'Baseline Clean', fullfile(folder, 'TEST6a_baseline_clean.csv'), struct('noise', 0, 'latency', 0, 'dropout', 0);
        'TEST6b', '10% Noise', fullfile(folder, 'TEST6b_with_10pct_noise.csv'), struct('noise', 0.10, 'latency', 0, 'dropout', 0);
        'TEST6c', '50ms Latency', fullfile(folder, 'TEST6c_with_50ms_latency.csv'), struct('noise', 0, 'latency', 0.050, 'dropout', 0);
        'TEST6d', '5% Dropout', fullfile(folder, 'TEST6d_with_5pct_dropout.csv'), struct('noise', 0, 'latency', 0, 'dropout', 0.05);
        'TEST6e', 'Combined Stress', fullfile(folder, 'TEST6e_combined_stress.csv'), struct('noise', 0.10, 'latency', 0.050, 'dropout', 0.05);
    };

    results = run_comparison(API_URL, scenarios, N, m0, k0, zeta_target, dt, soft_story_idx, soft_story_factor, tmd_mass);

    display_results_table(results, scenarios);
    save_results(results, scenarios, API_URL, 'robustness');

    fprintf('\n═══ ROBUSTNESS TESTS COMPLETE ═══\n');
end

%% ============================================================
%% MAIN COMPARISON ENGINE (imported from tmd_comparision.m)
%% ============================================================
function results = run_comparison(API_URL, scenarios, N, m0, k0, zeta_target, dt, soft_story_idx, soft_story_factor, tmd_mass)

    n_scenarios = size(scenarios, 1);

    % Build structural system
    fprintf('🏗️  Building structural system...\n');
    [M, K, C, om_sorted, V] = build_structure(N, m0, k0, zeta_target, soft_story_idx, soft_story_factor);
    fprintf('  Building: %d floors, First mode: %.3f s\n', N, 2*pi/om_sorted(1));
    fprintf('✓ System built\n\n');

    % Initialize results
    results = struct();
    controllers = {'Passive', 'Fuzzy', 'RL_Base', 'RL_CL'};

    for ctrl = controllers
        results.(ctrl{1}) = struct();
        results.(ctrl{1}).peak_roof = zeros(n_scenarios, 1);
        results.(ctrl{1}).max_drift = zeros(n_scenarios, 1);
        results.(ctrl{1}).DCR = zeros(n_scenarios, 1);
        results.(ctrl{1}).rms_roof = zeros(n_scenarios, 1);
        results.(ctrl{1}).peak_force = zeros(n_scenarios, 1);
        results.(ctrl{1}).mean_force = zeros(n_scenarios, 1);
        results.(ctrl{1}).time = zeros(n_scenarios, 1);
        results.(ctrl{1}).peak_disp_by_floor = zeros(N, n_scenarios);  % Peak displacement at each floor
        results.(ctrl{1}).rms_roof_accel = zeros(n_scenarios, 1);      % RMS roof acceleration
    end

    % Store earthquake metadata
    results.earthquake = struct();
    results.earthquake.PGA = zeros(n_scenarios, 1);
    results.earthquake.time = cell(n_scenarios, 1);
    results.earthquake.accel = cell(n_scenarios, 1);

    % Main test loop
    fprintf('🚀 Starting 4-way comparison...\n\n');

    for scenario_idx = 1:n_scenarios
        scenario_name = scenarios{scenario_idx, 1};
        scenario_desc = scenarios{scenario_idx, 2};
        eq_file = scenarios{scenario_idx, 3};
        perturbations = scenarios{scenario_idx, 4};

        fprintf('════════════════════════════════════════════════════════\n');
        fprintf('  SCENARIO %d/%d: %s - %s\n', scenario_idx, n_scenarios, scenario_name, scenario_desc);
        fprintf('════════════════════════════════════════════════════════\n\n');

        % Load earthquake data
        if ~exist(eq_file, 'file')
            fprintf('⚠️  Skipping (file not found): %s\n\n', eq_file);
            continue;
        end

        eq_data = readmatrix(eq_file);
        t = eq_data(:, 1);
        ag = eq_data(:, 2);
        Nt = length(t);

        % Ground motion forces
        r = ones(N, 1);
        Fg = -M * r * ag';

        fprintf('  Earthquake: %d steps, %.1f seconds\n', Nt, t(end));
        fprintf('  PGA: %.3f m/s² (%.2fg)\n\n', max(abs(ag)), max(abs(ag))/9.81);

        % Store earthquake data
        results.earthquake.PGA(scenario_idx) = max(abs(ag));
        results.earthquake.time{scenario_idx} = t;
        results.earthquake.accel{scenario_idx} = ag;

        % ========== TEST 1: PASSIVE TMD ==========
        fprintf('  [1/4] Testing Passive TMD... ');
        tic;
        [passive_results, x_passive, v_passive, M_passive, K_passive, C_passive] = test_passive_tmd(...
            M, K, C, Fg, t, om_sorted, tmd_mass, N, Nt);
        results.Passive.peak_roof(scenario_idx) = passive_results.peak_roof;
        results.Passive.max_drift(scenario_idx) = passive_results.max_drift;
        results.Passive.DCR(scenario_idx) = passive_results.DCR;
        results.Passive.rms_roof(scenario_idx) = passive_results.rms_roof;
        results.Passive.peak_disp_by_floor(:, scenario_idx) = passive_results.peak_disp_by_floor;
        results.Passive.rms_roof_accel(scenario_idx) = passive_results.rms_roof_accel;
        results.Passive.time(scenario_idx) = toc;
        fprintf('Peak: %.2f cm (%.2f s)\n', passive_results.peak_roof*100, results.Passive.time(scenario_idx));

        % ========== TEST 2: FUZZY LOGIC (using /fuzzy/simulate endpoint) ==========
        fprintf('  [2/4] Testing Fuzzy Logic (full simulation)... ');
        tic;
        fuzzy_results = test_fuzzy_simulate(ag, dt, API_URL);
        results.Fuzzy.peak_roof(scenario_idx) = fuzzy_results.peak_roof;
        results.Fuzzy.max_drift(scenario_idx) = fuzzy_results.max_drift;
        results.Fuzzy.DCR(scenario_idx) = fuzzy_results.DCR;
        results.Fuzzy.rms_roof(scenario_idx) = fuzzy_results.rms_roof;
        results.Fuzzy.peak_force(scenario_idx) = fuzzy_results.peak_force;
        results.Fuzzy.mean_force(scenario_idx) = fuzzy_results.mean_force;
        if isfield(fuzzy_results, 'peak_disp_by_floor')
            results.Fuzzy.peak_disp_by_floor(:, scenario_idx) = fuzzy_results.peak_disp_by_floor;
        end
        if isfield(fuzzy_results, 'rms_roof_accel')
            results.Fuzzy.rms_roof_accel(scenario_idx) = fuzzy_results.rms_roof_accel;
        end
        results.Fuzzy.time(scenario_idx) = toc;
        fprintf('Peak: %.2f cm, Force: %.1f kN (%.2f s)\n', ...
            fuzzy_results.peak_roof*100, fuzzy_results.mean_force, results.Fuzzy.time(scenario_idx));

        % ========== TEST 3: RL BASE (using /rl/simulate endpoint) ==========
        fprintf('  [3/4] Testing RL Base (full simulation)... ');
        tic;
        rl_results = test_rl_simulate(ag, dt, API_URL);
        results.RL_Base.peak_roof(scenario_idx) = rl_results.peak_roof;
        results.RL_Base.max_drift(scenario_idx) = rl_results.max_drift;
        results.RL_Base.DCR(scenario_idx) = rl_results.DCR;
        results.RL_Base.rms_roof(scenario_idx) = rl_results.rms_roof;
        results.RL_Base.peak_force(scenario_idx) = rl_results.peak_force;
        results.RL_Base.mean_force(scenario_idx) = rl_results.mean_force;
        if isfield(rl_results, 'peak_disp_by_floor')
            results.RL_Base.peak_disp_by_floor(:, scenario_idx) = rl_results.peak_disp_by_floor;
        end
        if isfield(rl_results, 'rms_roof_accel')
            results.RL_Base.rms_roof_accel(scenario_idx) = rl_results.rms_roof_accel;
        end
        results.RL_Base.time(scenario_idx) = toc;
        fprintf('Peak: %.2f cm, Force: %.1f kN (%.2f s)\n', ...
            rl_results.peak_roof*100, rl_results.mean_force, results.RL_Base.time(scenario_idx));

        % ========== TEST 4: RL_CL (using /rl-cl/simulate endpoint) ==========
        fprintf('  [4/4] Testing RL_CL (full simulation)... ');
        tic;
        rl_cl_results = test_rl_cl_simulate(ag, dt, API_URL);
        results.RL_CL.peak_roof(scenario_idx) = rl_cl_results.peak_roof;
        results.RL_CL.max_drift(scenario_idx) = rl_cl_results.max_drift;
        results.RL_CL.DCR(scenario_idx) = rl_cl_results.DCR;
        results.RL_CL.rms_roof(scenario_idx) = rl_cl_results.rms_roof;
        results.RL_CL.peak_force(scenario_idx) = rl_cl_results.peak_force;
        results.RL_CL.mean_force(scenario_idx) = rl_cl_results.mean_force;
        if isfield(rl_cl_results, 'peak_disp_by_floor')
            results.RL_CL.peak_disp_by_floor(:, scenario_idx) = rl_cl_results.peak_disp_by_floor;
        end
        if isfield(rl_cl_results, 'rms_roof_accel')
            results.RL_CL.rms_roof_accel(scenario_idx) = rl_cl_results.rms_roof_accel;
        end
        results.RL_CL.time(scenario_idx) = toc;
        fprintf('Peak: %.2f cm, Force: %.1f kN (%.2f s)\n\n', ...
            rl_cl_results.peak_roof*100, rl_cl_results.mean_force, results.RL_CL.time(scenario_idx));
    end

    % Calculate improvements
    fprintf('════════════════════════════════════════════════════════\n');
    fprintf('  CALCULATING IMPROVEMENTS\n');
    fprintf('════════════════════════════════════════════════════════\n\n');

    for ctrl = {'Fuzzy', 'RL_Base', 'RL_CL'}
        results.(ctrl{1}).improvement_roof = zeros(n_scenarios, 1);
        results.(ctrl{1}).improvement_drift = zeros(n_scenarios, 1);
        results.(ctrl{1}).improvement_DCR = zeros(n_scenarios, 1);

        for i = 1:n_scenarios
            if results.Passive.peak_roof(i) > 0
                results.(ctrl{1}).improvement_roof(i) = 100 * (results.Passive.peak_roof(i) - results.(ctrl{1}).peak_roof(i)) / results.Passive.peak_roof(i);
                results.(ctrl{1}).improvement_drift(i) = 100 * (results.Passive.max_drift(i) - results.(ctrl{1}).max_drift(i)) / results.Passive.max_drift(i);
                results.(ctrl{1}).improvement_DCR(i) = 100 * (results.Passive.DCR(i) - results.(ctrl{1}).DCR(i)) / results.Passive.DCR(i);
            end
        end
    end
end

%% ============================================================
%% HELPER FUNCTIONS (from tmd_comparision.m)
%% ============================================================

% Include all helper functions from tmd_comparision.m:
% - build_structure
% - test_passive_tmd
% - test_active_controller
% - display_results_table
% - create_comparison_plots
% - save_results
% - check_datasets_exist
% - get_forces_from_api
% - apply_perturbations
% - newmark_simulate
% - compute_interstory_drifts
% - compute_DCR
% - solve_rayleigh

function [M, K, C, om_sorted, V] = build_structure(N, m0, k0, zeta_target, soft_story_idx, soft_story_factor)
    % Build structural system matrices
    
    % Mass matrix
    m = m0 * ones(N, 1);
    M = diag(m);
    
    % Stiffness matrix (with soft story)
    k_story = k0 * ones(N, 1);
    k_story(soft_story_idx) = soft_story_factor * k0;
    
    K = zeros(N, N);
    K(1,1) = k_story(1) + k0;
    K(1,2) = -k_story(1);
    
    for i = 2:N-1
        K(i,i-1) = -k_story(i-1);
        K(i,i) = k_story(i-1) + k_story(i);
        K(i,i+1) = -k_story(i);
    end
    
    K(N,N-1) = -k_story(N-1);
    K(N,N) = k_story(N-1);
    
    % Modal analysis for damping
    [Vfull, Dfull] = eig(K, M);
    lam_full = diag(Dfull);
    [om_sorted, order] = sort(sqrt(lam_full));
    V = Vfull(:, order);
    
    % Rayleigh damping
    if numel(om_sorted) >= 2
        a0a1 = solve_rayleigh(om_sorted(1), om_sorted(2), zeta_target);
        C = a0a1(1)*M + a0a1(2)*K;
    else
        C = 2*zeta_target*sqrt(om_sorted(1))*M;
    end
end

function [passive_results, x_passive, v_passive, M_passive, K_passive, C_passive] = test_passive_tmd(...
    M, K, C, Fg, t, om_sorted, tmd_mass, N, Nt)
    % Test passive TMD controller

    % Find optimal passive TMD parameters
    [passive_floor, passive_freq, passive_damping] = optimize_passive_tmd(M, K, C, Fg, t, om_sorted(1), tmd_mass);

    % Build passive TMD system
    k_tmd_passive = tmd_mass * passive_freq^2;
    c_tmd_passive = 2 * passive_damping * sqrt(k_tmd_passive * tmd_mass);

    [M_passive, K_passive, C_passive, F_passive] = augment_with_TMD(M, K, C, Fg, N, tmd_mass, k_tmd_passive, c_tmd_passive, Nt, passive_floor);

    % Simulate
    [x_passive, v_passive, a_passive] = newmark_simulate(M_passive, C_passive, K_passive, F_passive, t);

    % Extract results
    roof_passive = x_passive(N, :);
    drift_passive = compute_interstory_drifts(x_passive(1:N, :));

    passive_results.peak_roof = max(abs(roof_passive));
    passive_results.max_drift = max(abs(drift_passive(:)));
    passive_results.DCR = compute_DCR(drift_passive);
    passive_results.rms_roof = rms(roof_passive);

    % Additional metrics for analysis plots
    passive_results.peak_disp_by_floor = max(abs(x_passive(1:N, :)), [], 2);  % Peak displacement at each floor
    passive_results.rms_roof_accel = rms(a_passive(N, :));  % RMS roof acceleration
end

function active_results = test_active_controller(x_passive, v_passive, M_passive, K_passive, C_passive,...
    Fg, t, N, Nt, API_URL, controller_type, perturbations, dt)
    % Test active controller (Fuzzy only - RL controllers use simulate endpoints)
    %
    % IMPORTANT: Fuzzy controller expects ABSOLUTE TMD states (not relative)
    % because it computes relative states internally

    % Get state history from passive simulation
    roof_disp = x_passive(N, :)';      % ABSOLUTE roof displacement
    roof_vel = v_passive(N, :)';       % ABSOLUTE roof velocity
    tmd_disp = x_passive(N+1, :)';     % ABSOLUTE TMD displacement (NOT relative!)
    tmd_vel = v_passive(N+1, :)';      % ABSOLUTE TMD velocity (NOT relative!)

    % Apply perturbations
    [roof_disp_pert, roof_vel_pert, tmd_disp_pert, tmd_vel_pert] = ...
        apply_perturbations(roof_disp, roof_vel, tmd_disp, tmd_vel, perturbations, dt);
    
    % Get forces from API
    try
        forces_kN = get_forces_from_api(roof_disp_pert, roof_vel_pert, tmd_disp_pert, tmd_vel_pert, API_URL, controller_type);
        forces = forces_kN * 1000;  % Convert kN to N

        % Diagnostic output (first 5 timesteps)
        % fprintf('  DEBUG %s - States: roof_d=%.4f, roof_v=%.4f, tmd_d=%.4f, tmd_v=%.4f\n', ...
        %     upper(controller_type), roof_disp_pert(1), roof_vel_pert(1), tmd_disp_pert(1), tmd_vel_pert(1));
        % fprintf('  DEBUG %s - Forces (kN): min=%.2f, max=%.2f, mean=%.2f, samples=[%.2f %.2f %.2f]\n', ...
        %     upper(controller_type), min(forces_kN), max(forces_kN), mean(abs(forces_kN)), ...
        %     forces_kN(1), forces_kN(2), forces_kN(3));
    catch ME
        fprintf('API FAILED: %s\n', ME.message);
        forces = zeros(Nt, 1);
        forces_kN = zeros(Nt, 1);
    end
    
    % Re-simulate with active control (CORRECT SIGNS)
    F_active = [Fg; zeros(1, Nt)];
    F_active(N, :) = F_active(N, :) - forces';      % Roof (reaction force)
    F_active(N+1, :) = F_active(N+1, :) + forces';  % TMD (actuator force)
    
    [x_active, v_active, a_active] = newmark_simulate(M_passive, C_passive, K_passive, F_active, t);
    
    % Extract results
    roof_active = x_active(N, :);
    drift_active = compute_interstory_drifts(x_active(1:N, :));
    
    active_results.peak_roof = max(abs(roof_active));
    active_results.max_drift = max(abs(drift_active(:)));
    active_results.DCR = compute_DCR(drift_active);
    active_results.rms_roof = rms(roof_active);
    active_results.peak_force = max(abs(forces_kN));
    active_results.mean_force = mean(abs(forces_kN));
end

function [floor, freq, damping] = optimize_passive_tmd(M, K, C, F, t, om1, tmd_mass)
    % Simple passive TMD optimization using Den Hartog formulas
    N = size(M, 1);
    floor = N;
    
    mu = tmd_mass / M(floor, floor);
    freq = om1 / (1 + mu);
    damping = sqrt(3*mu / (8*(1+mu)));
end

function [M2, K2, C2, F2] = augment_with_TMD(M, K, C, F, N, m_t, k_t, c_t, Nt, floor)
    % Add TMD to structure at specified floor
    M2 = blkdiag(M, m_t);
    K2 = blkdiag(K, 0);
    C2 = blkdiag(C, 0);
    
    K2(floor, floor) = K2(floor, floor) + k_t;
    K2(floor, N+1) = -k_t;
    K2(N+1, floor) = -k_t;
    K2(N+1, N+1) = k_t;
    
    C2(floor, floor) = C2(floor, floor) + c_t;
    C2(floor, N+1) = -c_t;
    C2(N+1, floor) = -c_t;
    C2(N+1, N+1) = c_t;
    
    F2 = [F; zeros(1, Nt)];
end

function [x, v, a] = newmark_simulate(M, C, K, F, t)
    % Newmark-beta time integration
    dt = t(2) - t(1);
    Nt = length(t);
    N = size(M, 1);
    
    beta = 1/4;
    gamma = 1/2;
    
    x = zeros(N, Nt);
    v = zeros(N, Nt);
    a = zeros(N, Nt);
    
    a(:,1) = M \ (F(:,1) - C*v(:,1) - K*x(:,1));
    
    Khat = K + gamma/(beta*dt)*C + M/(beta*dt^2);
    [L, U, P] = lu(Khat);
    
    for k = 1:Nt-1
        xk = x(:,k);
        vk = v(:,k);
        ak = a(:,k);
        
        F_eff = F(:,k+1) + ...
                M * ((1/(beta*dt^2))*xk + (1/(beta*dt))*vk + (1/(2*beta)-1)*ak) + ...
                C * ((gamma/(beta*dt))*xk + (gamma/beta-1)*vk + dt*(gamma/(2*beta)-1)*ak);
        
        y = L \ (P * F_eff);
        x(:,k+1) = U \ y;
        
        a(:,k+1) = (1/(beta*dt^2))*(x(:,k+1) - xk) - (1/(beta*dt))*vk - (1/(2*beta)-1)*ak;
        v(:,k+1) = vk + dt*((1-gamma)*ak + gamma*a(:,k+1));
    end
    
    x(~isfinite(x)) = 0;
    v(~isfinite(v)) = 0;
    a(~isfinite(a)) = 0;
end

function a0a1 = solve_rayleigh(om1, om2, zeta)
    % Solve for Rayleigh damping coefficients
    A = 0.5 * [1/om1, om1; 1/om2, om2];
    sol = A \ [zeta; zeta];
    a0a1 = sol(:);
end

function drift = compute_interstory_drifts(x)
    % Calculate inter-story drifts
    N = size(x, 1);
    Nt = size(x, 2);
    drift = zeros(N-1, Nt);
    for i = 2:N
        drift(i-1, :) = x(i, :) - x(i-1, :);
    end
end

function DCR = compute_DCR(drift)
    % Compute Drift Concentration Ratio
    peak_per_story = max(abs(drift), [], 2);
    sorted_peaks = sort(peak_per_story);
    n = length(sorted_peaks);
    percentile_75 = sorted_peaks(round(0.75*n));
    max_peak = max(peak_per_story);
    
    if percentile_75 > 0
        DCR = max_peak / percentile_75;
    else
        DCR = Inf;
    end
end

function [roof_d, roof_v, tmd_d, tmd_v] = apply_perturbations(roof_disp, roof_vel, tmd_disp, tmd_vel, params, dt)
    % Apply noise, latency, dropout perturbations
    roof_d = roof_disp;
    roof_v = roof_vel;
    tmd_d = tmd_disp;
    tmd_v = tmd_vel;
    
    % Noise
    if params.noise > 0
        roof_d = roof_d + params.noise * std(roof_d) * randn(size(roof_d));
        roof_v = roof_v + params.noise * std(roof_v) * randn(size(roof_v));
        tmd_d = tmd_d + params.noise * std(tmd_d) * randn(size(tmd_d));
        tmd_v = tmd_v + params.noise * std(tmd_v) * randn(size(tmd_v));
    end
    
    % Latency
    if params.latency > 0
        lag = round(params.latency / dt);
        roof_d = [zeros(lag, 1); roof_d(1:end-lag)];
        roof_v = [zeros(lag, 1); roof_v(1:end-lag)];
        tmd_d = [zeros(lag, 1); tmd_d(1:end-lag)];
        tmd_v = [zeros(lag, 1); tmd_v(1:end-lag)];
    end
    
    % Dropout
    if params.dropout > 0
        n = length(roof_d);
        dropout_mask = rand(n, 1) > params.dropout;
        
        if sum(dropout_mask) > 1
            roof_d(~dropout_mask) = interp1(find(dropout_mask), roof_d(dropout_mask), find(~dropout_mask), 'linear', 'extrap');
            roof_v(~dropout_mask) = interp1(find(dropout_mask), roof_v(dropout_mask), find(~dropout_mask), 'linear', 'extrap');
            tmd_d(~dropout_mask) = interp1(find(dropout_mask), tmd_d(dropout_mask), find(~dropout_mask), 'linear', 'extrap');
            tmd_v(~dropout_mask) = interp1(find(dropout_mask), tmd_v(dropout_mask), find(~dropout_mask), 'linear', 'extrap');
        end
    end
end

function forces = get_forces_from_api(roof_disp, roof_vel, tmd_disp, tmd_vel, API_URL, controller_type)
    % Get control forces from API (batch prediction)
    batch_data = struct(...
        'roof_displacements', roof_disp, ...
        'roof_velocities', roof_vel, ...
        'tmd_displacements', tmd_disp, ...
        'tmd_velocities', tmd_vel ...
    );

    json_data = jsonencode(batch_data);

    options = weboptions(...
        'MediaType', 'application/json', ...
        'ContentType', 'json', ...
        'Timeout', 120, ...
        'RequestMethod', 'post' ...
    );

    % Select endpoint
    switch controller_type
        case 'fuzzy'
            endpoint = [API_URL 'fuzzy/predict-batch'];
        case 'rl'
            endpoint = [API_URL 'rl/predict-batch'];
        case 'rl_cl'
            endpoint = [API_URL 'rl-cl/predict-batch'];
        otherwise
            error('Unknown controller type: %s', controller_type);
    end

    response = webwrite(endpoint, json_data, options);

    % Extract forces from response
    % All endpoints return 'forces' field in kN
    if isfield(response, 'forces')
        forces = response.forces;
    elseif isfield(response, 'forces_kN')
        forces = response.forces_kN;
    else
        error('API response missing forces field for %s', controller_type);
    end

    forces = forces(:);
end

function rl_results = test_rl_simulate(earthquake_data, dt, API_URL)
    % Test RL Baseline controller using full simulation endpoint
    % This endpoint runs the simulation in the RL environment and returns
    % comprehensive metrics including DCR, RMS, max drift, and forces

    % Prepare request data
    sim_data = struct(...
        'earthquake_data', earthquake_data(:)', ...  % Row vector
        'dt', dt ...
    );

    json_data = jsonencode(sim_data);

    options = weboptions(...
        'MediaType', 'application/json', ...
        'ContentType', 'json', ...
        'Timeout', 300, ...  % 5 minutes for full simulation
        'RequestMethod', 'post' ...
    );

    % Call simulate endpoint
    endpoint = [API_URL 'rl/simulate'];

    try
        response = webwrite(endpoint, json_data, options);

        % Extract metrics from response
        % All metrics are already calculated by the API
        rl_results.peak_roof = response.peak_roof_displacement;  % meters
        rl_results.rms_roof = response.rms_roof_displacement;    % meters
        rl_results.max_drift = response.max_drift;               % meters
        rl_results.DCR = response.DCR;                           % dimensionless
        rl_results.peak_force = response.peak_force_kN;          % kN
        rl_results.mean_force = response.mean_force_kN;          % kN

    catch ME
        fprintf('  ⚠️  RL SIMULATE API FAILED: %s\n', ME.message);
        fprintf('     Using fallback values (zeros)\n');

        % Return zero results on failure
        rl_results.peak_roof = 0;
        rl_results.rms_roof = 0;
        rl_results.max_drift = 0;
        rl_results.DCR = 0;
        rl_results.peak_force = 0;
        rl_results.mean_force = 0;
    end
end

function rl_cl_results = test_rl_cl_simulate(earthquake_data, dt, API_URL)
    % Test RL_CL controller using full simulation endpoint
    % This endpoint runs the simulation in the RL environment and returns
    % comprehensive metrics including DCR, RMS, max drift, and forces

    % Prepare request data
    sim_data = struct(...
        'earthquake_data', earthquake_data(:)', ...  % Row vector
        'dt', dt ...
    );

    json_data = jsonencode(sim_data);

    options = weboptions(...
        'MediaType', 'application/json', ...
        'ContentType', 'json', ...
        'Timeout', 300, ...  % 5 minutes for full simulation
        'RequestMethod', 'post' ...
    );

    % Call simulate endpoint
    endpoint = [API_URL 'rl-cl/simulate'];

    try
        response = webwrite(endpoint, json_data, options);

        % Extract metrics from response
        % All metrics are already calculated by the API
        rl_cl_results.peak_roof = response.peak_roof_displacement;  % meters
        rl_cl_results.rms_roof = response.rms_roof_displacement;    % meters
        rl_cl_results.max_drift = response.max_drift;               % meters
        rl_cl_results.DCR = response.DCR;                           % dimensionless
        rl_cl_results.peak_force = response.peak_force_kN;          % kN
        rl_cl_results.mean_force = response.mean_force_kN;          % kN

    catch ME
        fprintf('  ⚠️  RL_CL SIMULATE API FAILED: %s\n', ME.message);
        fprintf('     Using fallback values (zeros)\n');

        % Return zero results on failure
        rl_cl_results.peak_roof = 0;
        rl_cl_results.rms_roof = 0;
        rl_cl_results.max_drift = 0;
        rl_cl_results.DCR = 0;
        rl_cl_results.peak_force = 0;
        rl_cl_results.mean_force = 0;
    end
end

function fuzzy_results = test_fuzzy_simulate(earthquake_data, dt, API_URL)
    % Test Fuzzy Logic controller using full closed-loop simulation endpoint
    %
    % This uses the /fuzzy/simulate endpoint which runs a complete structural
    % simulation with closed-loop feedback (observe → control → update).
    % This is much more accurate than batch prediction.
    %
    % Args:
    %   earthquake_data: Ground acceleration array (m/s²)
    %   dt: Time step (seconds)
    %   API_URL: Base API URL
    %
    % Returns:
    %   fuzzy_results: Struct with all performance metrics

    try
        % Prepare simulation data
        sim_data = struct(...
            'earthquake_data', earthquake_data(:)', ...
            'dt', dt ...
        );

        % Convert to JSON
        json_data = jsonencode(sim_data);

        % Set up HTTP options
        options = weboptions(...
            'MediaType', 'application/json', ...
            'ContentType', 'json', ...
            'Timeout', 300, ...
            'HeaderFields', {'Content-Type', 'application/json'} ...
        );

        % Call API endpoint
        endpoint = [API_URL 'fuzzy/simulate'];
        response = webwrite(endpoint, json_data, options);

        % Extract metrics from response
        fuzzy_results.peak_roof = response.peak_roof_displacement;
        fuzzy_results.rms_roof = response.rms_roof_displacement;
        fuzzy_results.max_drift = response.max_drift;
        fuzzy_results.DCR = response.DCR;
        fuzzy_results.peak_force = response.peak_force_kN;
        fuzzy_results.mean_force = response.mean_force_kN;

    catch ME
        warning('Fuzzy simulate API call failed: %s', ME.message);
        fprintf('Error details: %s\n', ME.getReport());

        % Return zeros on failure
        fuzzy_results.peak_roof = 0;
        fuzzy_results.rms_roof = 0;
        fuzzy_results.max_drift = 0;
        fuzzy_results.DCR = 0;
        fuzzy_results.peak_force = 0;
        fuzzy_results.mean_force = 0;
    end
end

function display_results_table(results, scenarios)
    % Display formatted results table
    fprintf('\n');
    fprintf('════════════════════════════════════════════════════════════════════════════════════════════════════\n');
    fprintf('                           COMPARISON RESULTS (4 CONTROLLERS)\n');
    fprintf('════════════════════════════════════════════════════════════════════════════════════════════════════\n\n');
    
    fprintf('%-10s | %10s | %12s | %12s | %12s | %12s\n', ...
            'Scenario', 'Passive', 'Fuzzy', 'RL Base', 'RL CL', 'Winner');
    fprintf('%-10s | %10s | %12s | %12s | %12s | %12s\n', ...
            '', '(cm)', '(cm / %)', '(cm / %)', '(cm / %)', '');
    fprintf('-----------+------------+--------------+--------------+--------------+--------------\n');

    n_scenarios = size(scenarios, 1);  % Get number of rows (scenarios)
    for i = 1:n_scenarios
        passive = results.Passive.peak_roof(i);
        
        if passive == 0, continue; end
        
        fuzzy = results.Fuzzy.peak_roof(i);
        fuzzy_imp = results.Fuzzy.improvement_roof(i);
        
        rl = results.RL_Base.peak_roof(i);
        rl_imp = results.RL_Base.improvement_roof(i);
        
        rl_cl = results.RL_CL.peak_roof(i);
        rl_cl_imp = results.RL_CL.improvement_roof(i);
        
        [best_imp, best_idx] = max([fuzzy_imp, rl_imp, rl_cl_imp]);
        best_names = {'Fuzzy', 'RL Base', 'RL CL'};
        best = best_names{best_idx};
        
        fprintf('%-10s | %7.2f   | %6.2f/%4.1f%% | %6.2f/%4.1f%% | %6.2f/%4.1f%% | %12s\n', ...
                scenarios{i, 1}, passive*100, ...
                fuzzy*100, fuzzy_imp, rl*100, rl_imp, rl_cl*100, rl_cl_imp, best);
    end
    
    fprintf('-----------+------------+--------------+--------------+--------------+--------------\n');
    
    valid = results.Passive.peak_roof > 0;
    
    fprintf('%-10s | %7.2f   | %6.2f/%4.1f%% | %6.2f/%4.1f%% | %6.2f/%4.1f%% | %12s\n', ...
            'AVERAGE', ...
            mean(results.Passive.peak_roof(valid))*100, ...
            mean(results.Fuzzy.peak_roof(valid))*100, mean(results.Fuzzy.improvement_roof(valid)), ...
            mean(results.RL_Base.peak_roof(valid))*100, mean(results.RL_Base.improvement_roof(valid)), ...
            mean(results.RL_CL.peak_roof(valid))*100, mean(results.RL_CL.improvement_roof(valid)), ...
            '-');
    
    fprintf('\n');
    
    fprintf('FORCE STATISTICS:\n');
    fprintf('%-15s | %12s | %12s | %15s\n', 'Controller', 'Avg (kN)', 'Max (kN)', 'Efficiency');
    fprintf('----------------+--------------+--------------+-----------------\n');
    fprintf('%-15s | %12.1f | %12.1f | %15.3f %%/kN\n', 'Fuzzy', ...
            mean(results.Fuzzy.mean_force(valid)), ...
            mean(results.Fuzzy.peak_force(valid)), ...
            mean(results.Fuzzy.improvement_roof(valid)) / mean(results.Fuzzy.mean_force(valid)));
    fprintf('%-15s | %12.1f | %12.1f | %15.3f %%/kN\n', 'RL Baseline', ...
            mean(results.RL_Base.mean_force(valid)), ...
            mean(results.RL_Base.peak_force(valid)), ...
            mean(results.RL_Base.improvement_roof(valid)) / mean(results.RL_Base.mean_force(valid)));
    fprintf('%-15s | %12.1f | %12.1f | %15.3f %%/kN\n', 'RL CL', ...
            mean(results.RL_CL.mean_force(valid)), ...
            mean(results.RL_CL.peak_force(valid)), ...
            mean(results.RL_CL.improvement_roof(valid)) / mean(results.RL_CL.mean_force(valid)));
    
    fprintf('\n════════════════════════════════════════════════════════════════════════════════════════════════════\n\n');
end

function create_comparison_plots(results, scenarios)
    % Create comprehensive comparison plots
    valid = results.Passive.peak_roof > 0;
    scenario_labels = scenarios(valid, 1);
    n_scenarios = sum(valid);

    fig = figure('Position', [100 100 1600 1000], 'Color', 'w');

    % Plot 1: Peak Roof Displacement
    subplot(2, 3, 1);
    x = 1:n_scenarios;
    width = 0.22;

    bar(x - 1.5*width, results.Passive.peak_roof(valid)*100, width, 'FaceColor', [0.7 0.7 0.7]);
    hold on;
    bar(x - 0.5*width, results.Fuzzy.peak_roof(valid)*100, width, 'FaceColor', [1 0.6 0]);
    bar(x + 0.5*width, results.RL_Base.peak_roof(valid)*100, width, 'FaceColor', [0.3 0.5 0.8]);
    bar(x + 1.5*width, results.RL_CL.peak_roof(valid)*100, width, 'FaceColor', [0.2 0.8 0.2]);
    hold off;

    set(gca, 'XTick', 1:n_scenarios, 'XTickLabel', scenario_labels);
    xtickangle(45);
    ylabel('Peak Roof Displacement (cm)');
    title('Peak Roof Displacement Comparison');
    legend({'Passive', 'Fuzzy', 'RL Base', 'RL CL'}, 'Location', 'best');
    grid on;

    % Plot 2: Improvement vs Passive (with failure annotation)
    subplot(2, 3, 2);
    h_fuzzy = bar(x - width, results.Fuzzy.improvement_roof(valid), width, 'FaceColor', [1 0.6 0]);
    hold on;
    h_rl = bar(x, results.RL_Base.improvement_roof(valid), width, 'FaceColor', [0.3 0.5 0.8]);
    h_rl_cl = bar(x + width, results.RL_CL.improvement_roof(valid), width, 'FaceColor', [0.2 0.8 0.2]);
    hold off;

    set(gca, 'XTick', 1:n_scenarios, 'XTickLabel', scenario_labels);
    xtickangle(45);
    ylabel('Improvement (%)');
    title('Improvement vs Passive TMD');
    legend({'Fuzzy', 'RL Base', 'RL CL'}, 'Location', 'best');
    grid on;
    yline(0, 'k--');

    % Add annotation for catastrophic failure (if any negative values < -100%)
    rl_improvements = results.RL_Base.improvement_roof(valid);
    failure_idx = find(rl_improvements < -100);
    if ~isempty(failure_idx)
        for i = 1:length(failure_idx)
            idx = failure_idx(i);
            y_val = rl_improvements(idx);

            % Highlight the failure with red shaded danger zone
            % fill([idx-0.5, idx+0.5, idx+0.5, idx-0.5], [-250, -250, 0, 0], ...
            %      [1 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);

            % Add catastrophic failure warning (moved to the side with arrow)
            annotation_x = idx - 6;  % Move to the left of the bar
            annotation_y = -150;

            text(annotation_x, annotation_y, '⚠️ CATASTROPHIC FAILURE', ...
                'FontSize', 11, 'FontWeight', 'bold', 'Color', 'r', ...
                'HorizontalAlignment', 'left');
            text(annotation_x, annotation_y - 25, 'RL Baseline unsafe under latency!', ...
                'FontSize', 9, 'FontWeight', 'bold', 'Color', [0.6 0 0], ...
                'HorizontalAlignment', 'left');

            % % Draw arrow pointing to the RL Base bar
            % annotation('arrow', ...
            %     'X', [idx, idx-2.0], ...  % Adjust based on subplot position
            %     'Y', [1.5, 0.25], ...
            %     'Color', 'r', 'LineWidth', 2, 'HeadStyle', 'cback1', 'HeadLength', 8, 'HeadWidth', 8);

            text(idx, y_val - 20, sprintf('%.0f%%', y_val), ...
                'HorizontalAlignment', 'center', 'Color', 'red', 'FontWeight', 'bold', 'FontSize', 9);
        end
    end

    % Plot 3: Force Efficiency (SWAPPED from plot 4)
    subplot(2, 3, 3);
    avg_fuzzy_force = mean(results.Fuzzy.mean_force(valid));
    avg_fuzzy_imp = mean(results.Fuzzy.improvement_roof(valid));
    avg_rl_force = mean(results.RL_Base.mean_force(valid));
    avg_rl_imp = mean(results.RL_Base.improvement_roof(valid));
    avg_rl_cl_force = mean(results.RL_CL.mean_force(valid));
    avg_rl_cl_imp = mean(results.RL_CL.improvement_roof(valid));

    % Draw efficiency reference lines with specific slopes
    max_force = 110;  % Fixed upper limit for consistent visualization

    % Reference lines with exact slopes
    slope_1775 = 1.775;  % Best efficiency
    slope_0812 = 0.812;  % Good efficiency
    slope_0601 = 0.601;  % Baseline efficiency

    plot([0, max_force], [0, max_force * slope_1775], ':', 'Color', [0.8 0.8 0.8], 'LineWidth', 1.5);
    hold on;
    plot([0, max_force], [0, max_force * slope_0812], ':', 'Color', [0.8 0.8 0.8], 'LineWidth', 1.5);
    plot([0, max_force], [0, max_force * slope_0601], ':', 'Color', [0.8 0.8 0.8], 'LineWidth', 1.5);

    % Add efficiency labels with rotation
    text(85, 85 * slope_1775 + 2, '1.78 %/kN', ...
        'Color', [0.5 0.5 0.5], 'FontSize', 8, 'Rotation', atan2d(slope_1775, 1));
    text(85, 85 * slope_0812 - 5, '0.81 %/kN', ...
        'Color', [0.5 0.5 0.5], 'FontSize', 8, 'Rotation', atan2d(slope_0812, 1));

    % Plot points
    h1 = scatter(avg_fuzzy_force, avg_fuzzy_imp, ...
            200, 'o', 'MarkerEdgeColor', [1 0.6 0], 'MarkerFaceColor', [1 0.6 0], 'LineWidth', 2);
    h2 = scatter(avg_rl_force, avg_rl_imp, ...
            200, 's', 'MarkerEdgeColor', [0.3 0.5 0.8], 'MarkerFaceColor', [0.3 0.5 0.8], 'LineWidth', 2);
    h3 = scatter(avg_rl_cl_force, avg_rl_cl_imp, ...
            200, 'd', 'MarkerEdgeColor', [0.2 0.8 0.2], 'MarkerFaceColor', [0.2 0.8 0.2], 'LineWidth', 2);
    hold off;

    % Annotate points with exact coordinates
    text(avg_fuzzy_force - 8, avg_fuzzy_imp + 3, sprintf('(%.1f, %.1f)', avg_fuzzy_force, avg_fuzzy_imp), ...
        'FontSize', 8, 'Color', [1 0.6 0], 'FontWeight', 'bold', 'HorizontalAlignment', 'right');
    text(avg_rl_force + 2, avg_rl_imp - 5, sprintf('(%.1f, %.1f)', avg_rl_force, avg_rl_imp), ...
        'FontSize', 8, 'Color', [0.3 0.5 0.8], 'FontWeight', 'bold');
    text(avg_rl_cl_force + 2, avg_rl_cl_imp + 3, sprintf('(%.1f, %.1f)', avg_rl_cl_force, avg_rl_cl_imp), ...
        'FontSize', 8, 'Color', [0.2 0.8 0.2], 'FontWeight', 'bold');

    xlabel('Average Force (kN)');
    ylabel('Average Improvement (%)');
    title('Force Efficiency (Higher = Better)');
    legend([h1, h2, h3], {'Fuzzy', 'RL Base', 'RL CL', 'Efficiency Lines'}, 'Location', 'best');
    grid on;
    xlim([0 max_force]);
    ylim([min(avg_rl_imp) - 20, max([avg_fuzzy_imp, avg_rl_cl_imp]) + 10]);

    % Plot 4: DCR Comparison (SWAPPED from plot 3)
    subplot(2, 3, 4);
    bar(x - 1.5*width, results.Passive.DCR(valid), width, 'FaceColor', [0.7 0.7 0.7]);
    hold on;
    bar(x - 0.5*width, results.Fuzzy.DCR(valid), width, 'FaceColor', [1 0.6 0]);
    bar(x + 0.5*width, results.RL_Base.DCR(valid), width, 'FaceColor', [0.3 0.5 0.8]);
    bar(x + 1.5*width, results.RL_CL.DCR(valid), width, 'FaceColor', [0.2 0.8 0.2]);
    hold off;

    set(gca, 'XTick', 1:n_scenarios, 'XTickLabel', scenario_labels);
    xtickangle(45);
    ylabel('DCR (Lower = Better)');
    title('Drift Concentration Ratio');
    legend({'Passive', 'Fuzzy', 'RL Base', 'RL CL'}, 'Location', 'best');
    grid on;

    % Plot 5: RMS Displacement (CHANGED from Max Drift)
    subplot(2, 3, 5);
    bar(x - 1.5*width, results.Passive.rms_roof(valid)*100, width, 'FaceColor', [0.7 0.7 0.7]);
    hold on;
    bar(x - 0.5*width, results.Fuzzy.rms_roof(valid)*100, width, 'FaceColor', [1 0.6 0]);
    bar(x + 0.5*width, results.RL_Base.rms_roof(valid)*100, width, 'FaceColor', [0.3 0.5 0.8]);
    bar(x + 1.5*width, results.RL_CL.rms_roof(valid)*100, width, 'FaceColor', [0.2 0.8 0.2]);
    hold off;

    set(gca, 'XTick', 1:n_scenarios, 'XTickLabel', scenario_labels);
    xtickangle(45);
    ylabel('RMS Displacement (cm)');
    title('Root Mean Square Displacement');
    legend({'Passive', 'Fuzzy', 'RL Base', 'RL CL'}, 'Location', 'best');
    grid on;

    % Plot 6: Robustness (Perturbation scenarios) - ENHANCED
    subplot(2, 3, 6);
    % Find perturbation scenarios (Mod_10Noise, Mod_60Latency, etc.)
    stress_idx = find(valid & startsWith(scenario_labels, 'Mod_'));

    if ~isempty(stress_idx)
        % Find baseline (PEER_Moderate) for comparison
        baseline_idx_valid = find(valid & strcmp(scenario_labels, 'PEER_Moderate'));

        if ~isempty(baseline_idx_valid)
            % Find global index for baseline
            baseline_global = find(strcmp(scenarios(:, 1), 'PEER_Moderate'));

            % Calculate performance degradation vs baseline
            fuzzy_baseline = results.Fuzzy.peak_roof(baseline_global);
            rl_baseline = results.RL_Base.peak_roof(baseline_global);
            rl_cl_baseline = results.RL_CL.peak_roof(baseline_global);

            fuzzy_deg = (results.Fuzzy.peak_roof(valid) - fuzzy_baseline) / fuzzy_baseline * 100;
            rl_deg = (results.RL_Base.peak_roof(valid) - rl_baseline) / rl_baseline * 100;
            rl_cl_deg = (results.RL_CL.peak_roof(valid) - rl_cl_baseline) / rl_cl_baseline * 100;

            stress_labels_subset = scenario_labels(stress_idx);
            x_stress = 1:length(stress_idx);

            % Highlight danger zone for latency failures (x=2, around -60%)
            fill([1.5, 2.5, 2.5, 1.5], [-100, -100, -10, -10], ...
                 [1 0.7 0.7], 'EdgeColor', 'r', 'LineWidth', 1, 'FaceAlpha', 0.3);
            hold on;
            text(2, -75, 'UNSAFE', 'Color', 'r', 'FontSize', 10, 'FontWeight', 'bold', ...
                'HorizontalAlignment', 'center');

            % Highlight acceptable zone (degradation < 10%)
            y_limits = [-80, 50];
            fill([0.5, length(stress_idx)+0.5, length(stress_idx)+0.5, 0.5], ...
                 [10, 10, 50, 50], ...
                 [1 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.2);

            % Plot with thicker lines (LineWidth 3)
            h_fuzzy = plot(x_stress, fuzzy_deg(stress_idx), 'o-', 'Color', [1 0.6 0], 'LineWidth', 3, 'MarkerSize', 10, 'MarkerFaceColor', [1 0.6 0]);
            h_rl = plot(x_stress, rl_deg(stress_idx), 's-', 'Color', [0.3 0.5 0.8], 'LineWidth', 3, 'MarkerSize', 10, 'MarkerFaceColor', [0.3 0.5 0.8]);
            h_rl_cl = plot(x_stress, rl_cl_deg(stress_idx), 'd-', 'Color', [0.2 0.8 0.2], 'LineWidth', 3, 'MarkerSize', 10, 'MarkerFaceColor', [0.2 0.8 0.2]);

            % Add baseline
            yline(0, 'k--', 'LineWidth', 2);
            hold off;

            % Fix x-axis labels to readable names
            readable_labels = {'10% Noise', '40ms Latency', '8% Dropout', 'Combined'};
            set(gca, 'XTick', 1:length(stress_idx), 'XTickLabel', readable_labels);
            xtickangle(25);
            ylabel('Performance Degradation (%)');
            title('Robustness Under Perturbations');
            legend([h_fuzzy, h_rl, h_rl_cl], {'Fuzzy', 'RL Base', 'RL CL'}, 'Location', 'northeast');
            grid on;
            ylim(y_limits);

            % Annotate critical points with exact values
            % Point 1: RL Base latency failure at x=2
            text(2, -62, sprintf('%.0f%%', rl_deg(stress_idx(2))), ...
                'Color', [0.3 0.5 0.8], 'FontWeight', 'bold', 'FontSize', 9, ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');

            % Point 2: RL_CL combined stress at x=4
            text(4, 40, sprintf('+%.0f%%', rl_cl_deg(stress_idx(4))), ...
                'Color', [0.2 0.8 0.2], 'FontWeight', 'bold', 'FontSize', 9, ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
        end
    end

    sgtitle('4-Way TMD Controller Comparison', 'FontSize', 14, 'FontWeight', 'bold');

    saveas(fig, ['4way_comparison_plots_' datestr(now, 'yyyy-mm-dd_HHMMSS') '.png']);

end

function create_analysis_plots(results, scenarios)
    % Create additional analysis plots for earthquake and structural response characterization
    valid = results.Passive.peak_roof > 0;
    scenario_labels = scenarios(valid, 1);
    n_scenarios = sum(valid);

    % Find the 4 base scenarios (not perturbation variants)
    base_idx = find(valid & ~startsWith(scenario_labels, 'Mod_'));
    base_labels = scenario_labels(base_idx);

    fig = figure('Position', [100 100 1800 1000], 'Color', 'w');

    % Plot 1: Main Datasets' Distribution of Acceleration (time series for 4 base scenarios)
    subplot(2, 3, 1);
    colors = {[0.2 0.4 0.8], [1 0.6 0], [0.8 0.2 0.2], [0.6 0.2 0.6]};
    hold on;
    for i = 1:length(base_idx)
        global_idx = find(strcmp(scenarios(:,1), base_labels{i}));
        t = results.earthquake.time{global_idx};
        ag = results.earthquake.accel{global_idx};
        plot(t, ag, 'Color', colors{i}, 'LineWidth', 1.5, 'DisplayName', base_labels{i});
    end
    hold off;
    xlabel('Time (s)');
    ylabel('Acceleration (m/s²)');
    title('Earthquake Ground Motion Time Series');
    legend('Location', 'best');
    grid on;

    % Plot 2: PGA of All Scenarios
    subplot(2, 3, 2);
    x = 1:n_scenarios;
    pga_values = results.earthquake.PGA(valid);
    bar(x, pga_values, 'FaceColor', [0.4 0.6 0.8]);
    set(gca, 'XTick', 1:n_scenarios, 'XTickLabel', scenario_labels);
    xtickangle(45);
    ylabel('PGA (m/s²)');
    title('Peak Ground Acceleration - All Scenarios');
    grid on;

    % Plot 3: Peak Displacement By Story (for 4 base scenarios)
    subplot(2, 3, 3);
    N = size(results.Passive.peak_disp_by_floor, 1);  % Number of floors
    floors = 1:N;
    hold on;
    for i = 1:min(4, length(base_idx))
        global_idx = find(strcmp(scenarios(:,1), base_labels{i}));
        % Average peak displacement across all 4 controllers
        avg_peak = (results.Passive.peak_disp_by_floor(:, global_idx) + ...
                    results.Fuzzy.peak_disp_by_floor(:, global_idx) + ...
                    results.RL_Base.peak_disp_by_floor(:, global_idx) + ...
                    results.RL_CL.peak_disp_by_floor(:, global_idx)) / 4;
        plot(avg_peak * 100, floors, 'o-', 'Color', colors{i}, 'LineWidth', 2, ...
            'MarkerSize', 6, 'MarkerFaceColor', colors{i}, 'DisplayName', base_labels{i});
    end
    hold off;
    ylabel('Floor Number');
    xlabel('Peak Displacement (cm)');
    title('Peak Displacement Profile (Avg of 4 Controllers)');
    legend('Location', 'best');
    grid on;

    % Plot 4: Maximum RMS Displacement Reduction
    subplot(2, 3, 4);
    width = 0.25;
    x = 1:n_scenarios;

    % Calculate RMS reduction vs passive
    fuzzy_rms_reduction = (results.Passive.rms_roof(valid) - results.Fuzzy.rms_roof(valid)) ./ results.Passive.rms_roof(valid) * 100;
    rl_rms_reduction = (results.Passive.rms_roof(valid) - results.RL_Base.rms_roof(valid)) ./ results.Passive.rms_roof(valid) * 100;
    rl_cl_rms_reduction = (results.Passive.rms_roof(valid) - results.RL_CL.rms_roof(valid)) ./ results.Passive.rms_roof(valid) * 100;

    bar(x - width, fuzzy_rms_reduction, width, 'FaceColor', [1 0.6 0]);
    hold on;
    bar(x, rl_rms_reduction, width, 'FaceColor', [0.3 0.5 0.8]);
    bar(x + width, rl_cl_rms_reduction, width, 'FaceColor', [0.2 0.8 0.2]);
    hold off;

    set(gca, 'XTick', 1:n_scenarios, 'XTickLabel', scenario_labels);
    xtickangle(45);
    ylabel('RMS Displacement Reduction (%)');
    title('RMS Displacement Reduction vs Passive');
    legend({'Fuzzy', 'RL Base', 'RL CL'}, 'Location', 'best');
    grid on;
    yline(0, 'k--');

    % Plot 5: Max RMS Roof Acceleration (for all scenarios)
    subplot(2, 3, 5);
    bar(x - 1.5*width, results.Passive.rms_roof_accel(valid), width, 'FaceColor', [0.7 0.7 0.7]);
    hold on;
    bar(x - 0.5*width, results.Fuzzy.rms_roof_accel(valid), width, 'FaceColor', [1 0.6 0]);
    bar(x + 0.5*width, results.RL_Base.rms_roof_accel(valid), width, 'FaceColor', [0.3 0.5 0.8]);
    bar(x + 1.5*width, results.RL_CL.rms_roof_accel(valid), width, 'FaceColor', [0.2 0.8 0.2]);
    hold off;

    set(gca, 'XTick', 1:n_scenarios, 'XTickLabel', scenario_labels);
    xtickangle(45);
    ylabel('RMS Roof Acceleration (m/s²)');
    title('RMS Roof Acceleration - All Controllers');
    legend({'Passive', 'Fuzzy', 'RL Base', 'RL CL'}, 'Location', 'best');
    grid on;

    % Plot 6: Improvement Summary Table (text visualization)
    subplot(2, 3, 6);
    axis off;

    % Calculate average improvements
    avg_peak_imp = [mean(results.Fuzzy.improvement_roof(valid)), ...
                    mean(results.RL_Base.improvement_roof(valid)), ...
                    mean(results.RL_CL.improvement_roof(valid))];
    avg_rms_imp = [mean(fuzzy_rms_reduction), mean(rl_rms_reduction), mean(rl_cl_rms_reduction)];
    avg_dcr_imp = [mean(results.Fuzzy.improvement_DCR(valid)), ...
                   mean(results.RL_Base.improvement_DCR(valid)), ...
                   mean(results.RL_CL.improvement_DCR(valid))];

    % Create text table
    text(0.1, 0.9, 'Average Performance Summary', 'FontSize', 12, 'FontWeight', 'bold');
    text(0.1, 0.8, sprintf('Fuzzy: Peak %.1f%%, RMS %.1f%%, DCR %.1f%%', ...
        avg_peak_imp(1), avg_rms_imp(1), avg_dcr_imp(1)), 'FontSize', 10, 'Color', [1 0.6 0]);
    text(0.1, 0.7, sprintf('RL Base: Peak %.1f%%, RMS %.1f%%, DCR %.1f%%', ...
        avg_peak_imp(2), avg_rms_imp(2), avg_dcr_imp(2)), 'FontSize', 10, 'Color', [0.3 0.5 0.8]);
    text(0.1, 0.6, sprintf('RL CL: Peak %.1f%%, RMS %.1f%%, DCR %.1f%%', ...
        avg_peak_imp(3), avg_rms_imp(3), avg_dcr_imp(3)), 'FontSize', 10, 'Color', [0.2 0.8 0.2]);

    text(0.1, 0.4, 'Key Findings:', 'FontSize', 11, 'FontWeight', 'bold');
    [~, best_idx] = max(avg_peak_imp);
    best_names = {'Fuzzy', 'RL Base', 'RL CL'};
    text(0.1, 0.3, sprintf('• Best Peak Reduction: %s', best_names{best_idx}), 'FontSize', 9);
    text(0.1, 0.2, sprintf('• Scenarios Tested: %d', n_scenarios), 'FontSize', 9);
    text(0.1, 0.1, sprintf('• Building: %d floors with soft story', N), 'FontSize', 9);

    sgtitle('4-Way TMD Controller - Additional Analysis', 'FontSize', 14, 'FontWeight', 'bold');

    saveas(fig, '4way_analysis_plots.png');
end

function save_results(results, scenarios, API_URL, test_name)
    % Save results to MAT and JSON
    save(sprintf('4way_comparison_%s.mat', test_name), 'results', 'scenarios', 'API_URL');
    fprintf('💾 Saved: 4way_comparison_%s.mat\n', test_name);
    
    try
        json_data = struct();
        json_data.test_name = test_name;
        json_data.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
        json_data.api_url = API_URL;
        json_data.results = results;
        json_data.scenarios = scenarios(:,1:2);
        
        json_str = jsonencode(json_data);
        fid = fopen(sprintf('4way_comparison_%s.json', test_name), 'w');
        fprintf(fid, '%s', json_str);
        fclose(fid);
        fprintf('💾 Saved: 4way_comparison_%s.json\n', test_name);
    catch ME
        fprintf('⚠️  JSON save failed: %s\n', ME.message);
    end
end

function exists = check_datasets_exist(folder)
    % Check if required PEER synthetic dataset files exist
    required = {
        'PEER_small_M4.5_PGA0.25g.csv'
        'PEER_moderate_M5.7_PGA0.35g.csv'
        'PEER_high_M7.4_PGA0.75g.csv'
        'PEER_insane_M8.4_PGA0.9g.csv'
        'PEER_moderate_10pct_noise.csv'
        'PEER_moderate_60ms_latency.csv'
        'PEER_moderate_8pct_dropout.csv'
        'PEER_moderate_combined_stress.csv'
    };

    exists = true;

    % Check individual files
    for i = 1:length(required)
        filepath = fullfile(folder, required{i});
        if ~isfile(filepath)
            fprintf('❌ Missing: %s\n', filepath);
            fprintf('   Run: python matlab/data/earthquakes/generate_peer_earthquakes.py\n');
            exists = false;
        end
    end

    if exists
        fprintf('✅ All PEER synthetic earthquake datasets found\n');
    end
end
