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
    API_URL = 'https://perfect-rl-api-887344515766.us-east4.run.app/';
    % ===================================================

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
        'PEER_Small', 'Small (M4.5, 0.25g)', fullfile(folder, 'peer_synthetic', 'PEER_small_M4.5_PGA0.25g.csv'), struct('noise', 0, 'latency', 0, 'dropout', 0);
    };

    results = run_comparison(API_URL, scenarios, N, m0, k0, zeta_target, dt, soft_story_idx, soft_story_factor, tmd_mass);

    % Display and save
    display_results_table(results, scenarios);
    save_results(results, scenarios, API_URL, 'quick_demo');

    fprintf('\n═══ QUICK DEMO COMPLETE ═══\n');
end

%% ============================================================
%% RUN ALL 4 PEER SCENARIOS (SMALL → INSANE)
%% ============================================================
function run_all_scenarios(API_URL, folder, N, m0, k0, zeta_target, dt, soft_story_idx, soft_story_factor, tmd_mass)
    fprintf('\n═══ COMPREHENSIVE 4-WAY COMPARISON (4 PEER SCENARIOS) ═══\n\n');

    % All scenarios include realistic perturbations:
    % - 10% noise (sensor noise + site effects)
    % - 60ms delay (communication latency)
    % - 8% dropout (packet loss with hold-last-value)
    scenarios = {
        'PEER_Small',    'Small (M4.5, 0.25g)',    fullfile(folder, 'peer_synthetic', 'PEER_small_M4.5_PGA0.25g.csv'),       struct('noise', 0, 'latency', 0, 'dropout', 0);
        'PEER_Moderate', 'Moderate (M5.7, 0.35g)', fullfile(folder, 'peer_synthetic', 'PEER_moderate_M5.7_PGA0.35g.csv'),   struct('noise', 0, 'latency', 0, 'dropout', 0);
        'PEER_High',     'High (M7.4, 0.75g)',     fullfile(folder, 'peer_synthetic', 'PEER_high_M7.4_PGA0.75g.csv'),       struct('noise', 0, 'latency', 0, 'dropout', 0);
        'PEER_Insane',   'Insane (M8.4, 0.9g)',    fullfile(folder, 'peer_synthetic', 'PEER_insane_M8.4_PGA0.9g.csv'),      struct('noise', 0, 'latency', 0, 'dropout', 0);
    };

    results = run_comparison(API_URL, scenarios, N, m0, k0, zeta_target, dt, soft_story_idx, soft_story_factor, tmd_mass);

    % Display and save
    display_results_table(results, scenarios);
    create_comparison_plots(results, scenarios);
    fprintf('  ✓ Saved: 4way_comparison_plots.png\n\n');
    save_results(results, scenarios, API_URL, 'comprehensive');

    fprintf('\n═══ COMPREHENSIVE TEST COMPLETE ═══\n');
end

%% ============================================================
%% RUN SPECIFIC SCENARIO
%% ============================================================
function run_specific_scenario(API_URL, folder, N, m0, k0, zeta_target, dt, soft_story_idx, soft_story_factor, tmd_mass)
    fprintf('\n═══ SELECT SCENARIO ═══\n\n');
    fprintf('  1. Small: M4.5, PGA 0.25g (20s)\n');
    fprintf('  2. Moderate: M5.7, PGA 0.35g (40s)\n');
    fprintf('  3. High: M7.4, PGA 0.75g (80s)\n');
    fprintf('  4. Insane: M8.4, PGA 0.9g (120s)\n\n');
    fprintf('  All scenarios include: 10%% noise, 60ms delay, 8%% dropout\n\n');

    test_num = input('Enter test number (1-4): ');

    all_scenarios = {
        'PEER_Small',    'Small (M4.5, 0.25g)',    fullfile(folder, 'peer_synthetic', 'PEER_small_M4.5_PGA0.25g.csv'),       struct('noise', 0, 'latency', 0, 'dropout', 0);
        'PEER_Moderate', 'Moderate (M5.7, 0.35g)', fullfile(folder, 'peer_synthetic', 'PEER_moderate_M5.7_PGA0.35g.csv'),   struct('noise', 0, 'latency', 0, 'dropout', 0);
        'PEER_High',     'High (M7.4, 0.75g)',     fullfile(folder, 'peer_synthetic', 'PEER_high_M7.4_PGA0.75g.csv'),       struct('noise', 0, 'latency', 0, 'dropout', 0);
        'PEER_Insane',   'Insane (M8.4, 0.9g)',    fullfile(folder, 'peer_synthetic', 'PEER_insane_M8.4_PGA0.9g.csv'),      struct('noise', 0, 'latency', 0, 'dropout', 0);
    };

    if test_num < 1 || test_num > 4
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
    end

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

        % ========== TEST 1: PASSIVE TMD ==========
        fprintf('  [1/4] Testing Passive TMD... ');
        tic;
        [passive_results, x_passive, v_passive, M_passive, K_passive, C_passive] = test_passive_tmd(...
            M, K, C, Fg, t, om_sorted, tmd_mass, N, Nt);
        results.Passive.peak_roof(scenario_idx) = passive_results.peak_roof;
        results.Passive.max_drift(scenario_idx) = passive_results.max_drift;
        results.Passive.DCR(scenario_idx) = passive_results.DCR;
        results.Passive.rms_roof(scenario_idx) = passive_results.rms_roof;
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
    
    subplot(2, 3, 2);
    bar(x - width, results.Fuzzy.improvement_roof(valid), width, 'FaceColor', [1 0.6 0]);
    hold on;
    bar(x, results.RL_Base.improvement_roof(valid), width, 'FaceColor', [0.3 0.5 0.8]);
    bar(x + width, results.RL_CL.improvement_roof(valid), width, 'FaceColor', [0.2 0.8 0.2]);
    hold off;
    
    set(gca, 'XTick', 1:n_scenarios, 'XTickLabel', scenario_labels);
    xtickangle(45);
    ylabel('Improvement (%)');
    title('Improvement vs Passive TMD');
    legend({'Fuzzy', 'RL Base', 'RL CL'}, 'Location', 'best');
    grid on;
    yline(0, 'k--');
    
    sgtitle('4-Way TMD Controller Comparison', 'FontSize', 14, 'FontWeight', 'bold');
    
    saveas(fig, '4way_comparison_plots.png');
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
    peer_folder = fullfile(folder, 'peer_synthetic');
    required = {
        'PEER_small_M4.5_PGA0.25g.csv'
        'PEER_moderate_M5.7_PGA0.35g.csv'
        'PEER_high_M7.4_PGA0.75g.csv'
        'PEER_insane_M8.4_PGA0.9g.csv'
    };

    exists = true;

    % Check if peer_synthetic directory exists
    if ~isfolder(peer_folder)
        fprintf('❌ Missing directory: %s\n', peer_folder);
        fprintf('   Run: python matlab/data/earthquakes/generate_peer_earthquakes.py\n');
        exists = false;
        return;
    end

    % Check individual files
    for i = 1:length(required)
        filepath = fullfile(peer_folder, required{i});
        if ~isfile(filepath)
            fprintf('❌ Missing: %s\n', filepath);
            exists = false;
        end
    end

    if exists
        fprintf('✅ All PEER synthetic earthquake datasets found\n');
    end
end
