% ============================================================================
% COMPREHENSIVE 3-WAY TMD CONTROLLER COMPARISON
% ============================================================================
% 
% Compares THREE controllers across 9 test scenarios:
%   1. Passive TMD (mechanical, no API)
%   2. Fuzzy Logic Control (REST API)
%   3. Perfect RL with Curriculum Learning (REST API)
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
%   - Earthquake CSV files in 'datasets/' folder
%   - REST API deployed with Fuzzy and RL endpoints
%   - API URL configured below
%
% Author: Siddharth
% Date: December 2025
% ============================================================================

clear all; close all; clc;

%% ================================================================
%% CONFIGURATION
%% ================================================================

fprintf('\n════════════════════════════════════════════════════════\n');
fprintf('  3-WAY TMD CONTROLLER COMPARISON TEST\n');
fprintf('════════════════════════════════════════════════════════\n\n');

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

% TMD parameters (will be optimized for passive)
mu_tmd = 0.02;               % 2% mass ratio
tmd_mass = mu_tmd * m0;      % TMD mass

% Test scenarios
folder = 'datasets';
scenarios = {
    'TEST3', 'Small Earthquake (M4.5)', fullfile(folder, 'TEST3_small_earthquake_M4.5.csv'), struct('noise', 0, 'latency', 0, 'dropout', 0);
    'TEST4', 'Large Earthquake (M6.9)', fullfile(folder, 'TEST4_large_earthquake_M6.9.csv'), struct('noise', 0, 'latency', 0, 'dropout', 0);
    'TEST5', 'Moderate Earthquake (M6.7)', fullfile(folder, 'TEST5_earthquake_M6.7.csv'), struct('noise', 0, 'latency', 0, 'dropout', 0);
    'TEST6a', 'Baseline Clean', fullfile(folder, 'TEST6a_baseline_clean.csv'), struct('noise', 0, 'latency', 0, 'dropout', 0);
    'TEST6b', '10% Noise', fullfile(folder, 'TEST6b_with_10pct_noise.csv'), struct('noise', 0.10, 'latency', 0, 'dropout', 0);
    'TEST6c', '50ms Latency', fullfile(folder, 'TEST6c_with_50ms_latency.csv'), struct('noise', 0, 'latency', 0.050, 'dropout', 0);
    'TEST6d', '5% Dropout', fullfile(folder, 'TEST6d_with_5pct_dropout.csv'), struct('noise', 0, 'latency', 0, 'dropout', 0.05);
    'TEST6e', 'Combined Stress', fullfile(folder, 'TEST6e_combined_stress.csv'), struct('noise', 0.10, 'latency', 0.050, 'dropout', 0.05);
};

n_scenarios = size(scenarios, 1);

%% ================================================================
%% CHECK PREREQUISITES
%% ================================================================

fprintf('🔍 Checking prerequisites...\n\n');

% Check API connection
fprintf('  [1/2] Checking API connection... ');
try
    options = weboptions('Timeout', 10);
    health = webread([API_URL 'health'], options);
    fprintf('✅\n');
    fprintf('        Status: %s\n', health.status);
    if isfield(health, 'model')
        fprintf('        Model: %s\n', health.model);
    end
catch ME
    fprintf('❌ FAILED\n');
    fprintf('        Error: %s\n', ME.message);
    fprintf('\n⚠️  UPDATE API_URL in this script with your Cloud Run URL!\n');
    fprintf('   Current: %s\n\n', API_URL);
    error('Cannot connect to API. Please fix and retry.');
end

% Check earthquake files
fprintf('  [2/2] Checking earthquake files... ');
missing_files = {};
for i = 1:n_scenarios
    fprintf('%s: %s\n', scenarios{i,2},  scenarios{i,3});
    if ~exist(scenarios{i,3}, 'file')
        missing_files{end+1} = scenarios{i,3};
    end
end

if ~isempty(missing_files)
    fprintf('❌ MISSING\n');
    fprintf('        Missing files:\n');
    for i = 1:length(missing_files)
        fprintf('          - %s\n', missing_files{i});
    end
    error('Earthquake files missing. Please generate datasets first.');
else
    fprintf('✅\n');
end

fprintf('\n✓ All prerequisites satisfied!\n\n');

%% ================================================================
%% BUILD STRUCTURAL SYSTEM
%% ================================================================

fprintf('🏗️  Building structural system...\n');

% Mass matrix
m = m0 * ones(N, 1);
M = diag(m);

% Stiffness matrix (with soft 8th floor)
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

fprintf('  Building: %d floors\n', N);
fprintf('  Soft story: Floor %d (%.0f%% stiffness)\n', soft_story_idx, soft_story_factor*100);
fprintf('  First mode period: %.3f s\n', 2*pi/om_sorted(1));
fprintf('✓ System built\n\n');

%% ================================================================
%% INITIALIZE RESULTS STORAGE
%% ================================================================

results = struct();
% FIX: Use actual controller names that match what you store in the loop!
controllers = {'Passive', 'Fuzzy', 'RL_Base', 'RL_CL'};  % ← FIXED!

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

%% ================================================================
%% MAIN TEST LOOP
%% ================================================================

fprintf('🚀 Starting comprehensive comparison...\n\n');

for scenario_idx = 1:1 %n_scenarios
    
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
    
    % ============================================================
    % CONTROLLER 1: PASSIVE TMD
    % ============================================================
    fprintf('  [1/4] Testing Passive TMD... \n');
    tic;
    
    % Find optimal passive TMD parameters
    [passive_floor, passive_freq, passive_damping] = optimize_passive_tmd(M, K, C, Fg, t, om_sorted(1), tmd_mass);
    
    % Build passive TMD system
    k_tmd_passive = tmd_mass * passive_freq^2;
    c_tmd_passive = 2 * passive_damping * sqrt(k_tmd_passive * tmd_mass);
    
    [M_passive, K_passive, C_passive, F_passive] = augment_with_TMD(M, K, C, Fg, N, tmd_mass, k_tmd_passive, c_tmd_passive, Nt, passive_floor);
    
    % Simulate
    [x_passive, v_passive, a_passive] = newmark_simulate(M_passive, C_passive, K_passive, F_passive, t);
    
    % for print_idx = length(x_passive) -10 :length(x_passive)
    %     fprintf('Simulate with newmark, x: %f, v: %f, a: %f \n',x_passive, v_passive, a_passive)
    % end
    % Extract results
    roof_passive = x_passive(N, :);
    drift_passive = compute_interstory_drifts(x_passive(1:N, :));
    
    results.Passive.peak_roof(scenario_idx) = max(abs(roof_passive));
    results.Passive.max_drift(scenario_idx) = max(abs(drift_passive(:)));
    results.Passive.DCR(scenario_idx) = compute_DCR(drift_passive);
    results.Passive.rms_roof(scenario_idx) = rms(roof_passive);
    results.Passive.peak_force(scenario_idx) = 0;
    results.Passive.mean_force(scenario_idx) = 0;
    results.Passive.time(scenario_idx) = toc;
    
    % ✅ CHANGED: Display in cm
    fprintf('Peak: %.2f cm (%.2f s)\n', results.Passive.peak_roof(scenario_idx)*100, results.Passive.time(scenario_idx));
    
    % ============================================================
    % CONTROLLER 2: FUZZY LOGIC (API)
    % ============================================================
    fprintf('  [2/4] Testing Fuzzy Logic... \n');
    tic;
    
    % Get state history from passive simulation
    roof_disp = x_passive(N, :)';
    roof_vel = v_passive(N, :)';
    tmd_disp = x_passive(N+1, :)';  % TMD is last DOF
    tmd_vel = v_passive(N+1, :)';
    
    % for print_idx = length(roof_disp) -10 :length(roof_disp)
    %     fprintf('roof_disp: %f,roof_vel:%f,tmd_disp:%f,tmd_vel:%f \n',roof_disp,roof_vel,tmd_disp,tmd_vel )
    % end

    % Apply perturbations
    [roof_disp_pert, roof_vel_pert, tmd_disp_pert, tmd_vel_pert] = ...
        apply_perturbations(roof_disp, roof_vel, tmd_disp, tmd_vel, perturbations, dt);
    
    %fprintf('roof_disp_pert: %f,roof_vel_pert:%f,tmd_disp_pert:%f,tmd_vel_pert:%f \n',roof_disp_pert, roof_vel_pert, tmd_disp_pert, tmd_vel_pert)

    % Get forces from Fuzzy API
    try
        forces_fuzzy_kN  = get_forces_from_api(roof_disp_pert, roof_vel_pert, tmd_disp_pert, tmd_vel_pert, API_URL, 'fuzzy');
        forces_fuzzy = forces_fuzzy_kN * 1000;
    catch ME
        fprintf('❌ API FAILED\n');
        fprintf('        Error: %s\n', ME.message);
        fprintf('        Skipping Fuzzy for this scenario\n');
        forces_fuzzy = zeros(Nt, 1);
    end
    
    % Re-simulate with fuzzy control
    % ✅ FIX: Correct sign convention!
    F_fuzzy_active = [Fg; zeros(1, Nt)];
    F_fuzzy_active(N, :) = F_fuzzy_active(N, :) + forces_fuzzy';      % Roof (CHANGED: + not -)
    F_fuzzy_active(N+1, :) = F_fuzzy_active(N+1, :) - forces_fuzzy';  % TMD (CHANGED: - not +)
    
    [x_fuzzy, v_fuzzy, a_fuzzy] = newmark_simulate(M_passive, C_passive, K_passive, F_fuzzy_active, t);
    
    % Extract results
    roof_fuzzy = x_fuzzy(N, :);
    drift_fuzzy = compute_interstory_drifts(x_fuzzy(1:N, :));
    
    results.Fuzzy.peak_roof(scenario_idx) = max(abs(roof_fuzzy));
    results.Fuzzy.max_drift(scenario_idx) = max(abs(drift_fuzzy(:)));
    results.Fuzzy.DCR(scenario_idx) = compute_DCR(drift_fuzzy);
    results.Fuzzy.rms_roof(scenario_idx) = rms(roof_fuzzy);
    results.Fuzzy.peak_force(scenario_idx) = max(abs(forces_fuzzy_kN));
    results.Fuzzy.mean_force(scenario_idx) = mean(abs(forces_fuzzy_kN));
    results.Fuzzy.time(scenario_idx) = toc;
    
    % ✅ CHANGED: Display in cm
    fprintf('Peak: %.2f cm, Force: %.1f kN (%.2f s)\n', ...
    results.Fuzzy.peak_roof(scenario_idx)*100, results.Fuzzy.mean_force(scenario_idx), results.Fuzzy.time(scenario_idx));
    
    % ============================================================
    % CONTROLLER 3: RL (API)
    % ============================================================
    fprintf('  [3/4] Testing RL... ');
    tic;
    
    % Get forces from RL API
    try
        forces_rl_kN  = get_forces_from_api(roof_disp_pert, roof_vel_pert, tmd_disp_pert, tmd_vel_pert, API_URL, 'rl');
        forces_rl = forces_rl_kN * 1000;  % ← FIX: Convert kN to N!
    catch ME
        fprintf('❌ API FAILED\n');
        fprintf('        Error: %s\n', ME.message);
        fprintf('        Skipping RL for this scenario\n');
        forces_rl = zeros(Nt, 1);
    end
    
    % Re-simulate with RL control
    F_rl_active = [Fg; zeros(1, Nt)];
    F_rl_active(N, :) = F_rl_active(N, :) + forces_rl';      % Roof (CHANGED)
    F_rl_active(N+1, :) = F_rl_active(N+1, :) - forces_rl';  % TMD (CHANGED)

    
    [x_rl, v_rl, a_rl] = newmark_simulate(M_passive, C_passive, K_passive, F_rl_active, t);
    
    % Extract results
    roof_rl = x_rl(N, :);
    drift_rl = compute_interstory_drifts(x_rl(1:N, :));
    
    results.RL_Base.peak_roof(scenario_idx) = max(abs(roof_rl));
    results.RL_Base.max_drift(scenario_idx) = max(abs(drift_rl(:)));
    results.RL_Base.DCR(scenario_idx) = compute_DCR(drift_rl);
    results.RL_Base.rms_roof(scenario_idx) = rms(roof_rl);
    results.RL_Base.peak_force(scenario_idx) = max(abs(forces_rl_kN));
    results.RL_Base.mean_force(scenario_idx) = mean(abs(forces_rl_kN));
    results.RL_Base.time(scenario_idx) = toc;
    
    % ✅ CHANGED: Display in cm
    fprintf('Peak: %.2f cm, Force: %.1f kN (%.2f s)\n', ...
    results.RL_Base.peak_roof(scenario_idx)*100, results.RL_Base.mean_force(scenario_idx), results.RL_Base.time(scenario_idx));
    
    fprintf('\n');

    % ============================================================
    % CONTROLLER 4: RL W/CURRICULUM LEARNING (API)
    % ============================================================
    fprintf('  [4/4] Testing RL_CL... ');
    tic;
    
    % Get forces from Perfect RL API
    try
        forces_rl_cl_kN  = get_forces_from_api(roof_disp_pert, roof_vel_pert, tmd_disp_pert, tmd_vel_pert, API_URL, 'rl_cl');
        forces_rl_cl = forces_rl_cl_kN * 1000;
    catch ME
        fprintf('❌ API FAILED\n');
        fprintf('        Error: %s\n', ME.message);
        fprintf('        Skipping RL_CL for this scenario\n');
        forces_rl_cl = zeros(Nt, 1);
    end
    
    % Re-simulate with RL control
    F_rl_cl_active = [Fg; zeros(1, Nt)];
    F_rl_cl_active(N, :) = F_rl_cl_active(N, :) + forces_rl_cl';      % Roof (CHANGED)
    F_rl_cl_active(N+1, :) = F_rl_cl_active(N+1, :) - forces_rl_cl';  % TMD (CHANGED)
    
    [x_rl_cl, v_rl_cl, a_rl_cl] = newmark_simulate(M_passive, C_passive, K_passive, F_rl_cl_active, t);
    
    % Extract results
    roof_rl_cl = x_rl_cl(N, :);
    drift_rl_cl = compute_interstory_drifts(x_rl(1:N, :));
    
    results.RL_CL.peak_roof(scenario_idx) = max(abs(roof_rl_cl));
    results.RL_CL.max_drift(scenario_idx) = max(abs(drift_rl_cl(:)));
    results.RL_CL.DCR(scenario_idx) = compute_DCR(drift_rl_cl);
    results.RL_CL.rms_roof(scenario_idx) = rms(roof_rl_cl);
    results.RL_CL.peak_force(scenario_idx) = max(abs(forces_rl_cl_kN));
    results.RL_CL.mean_force(scenario_idx) = mean(abs(forces_rl_cl_kN));
    results.RL_CL.time(scenario_idx) = toc;

    % ✅ CHANGED: Display in cm
    fprintf('Peak: %.2f cm, Force: %.1f kN (%.2f s)\n', ...
    results.RL_CL.peak_roof(scenario_idx)*100, results.RL_CL.mean_force(scenario_idx), results.RL_CL.time(scenario_idx));
    
    fprintf('\n');
end








%% ================================================================
%% CALCULATE IMPROVEMENTS
%% ================================================================

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

%% ================================================================
%% DISPLAY RESULTS TABLE
%% ================================================================

display_results_table(results, scenarios);

%% ================================================================
%% CREATE COMPARISON PLOTS
%% ================================================================

fprintf('📊 Creating comparison plots...\n');
create_comparison_plots(results, scenarios);
fprintf('  ✓ Saved: comparison_passive_fuzzy_rl.png\n\n');

%% ================================================================
%% SAVE RESULTS
%% ================================================================

fprintf('💾 Saving results...\n');

% Save MATLAB data
save('comparison_passive_fuzzy_rl_results.mat', 'results', 'scenarios', 'API_URL');
fprintf('  ✓ Saved: comparison_passive_fuzzy_rl_results.mat\n');

% Save JSON
try
    json_data = struct();
    json_data.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    json_data.api_url = API_URL;
    json_data.n_scenarios = n_scenarios;
    json_data.scenarios = scenarios(:,1:2);
    
    % Summary statistics
    json_data.summary = struct();
    json_data.summary.passive_avg_roof = mean(results.Passive.peak_roof(results.Passive.peak_roof > 0));
    json_data.summary.fuzzy_avg_roof = mean(results.Fuzzy.peak_roof(results.Fuzzy.peak_roof > 0));
    json_data.summary.rl_avg_roof = mean(results.Perfect_RL.peak_roof(results.Perfect_RL.peak_roof > 0));
    json_data.summary.fuzzy_avg_improvement = mean(results.Fuzzy.improvement_roof(results.Fuzzy.improvement_roof > 0));
    json_data.summary.rl_avg_improvement = mean(results.Perfect_RL.improvement_roof(results.Perfect_RL.improvement_roof > 0));
    
    json_str = jsonencode(json_data);
    fid = fopen('comparison_passive_fuzzy_rl_summary.json', 'w');
    fprintf(fid, '%s', json_str);
    fclose(fid);
    fprintf('  ✓ Saved: comparison_passive_fuzzy_rl_summary.json\n');
catch ME
    fprintf('  ⚠️  JSON save failed: %s\n', ME.message);
end

fprintf('\n════════════════════════════════════════════════════════\n');
fprintf('  ✅ COMPREHENSIVE TEST COMPLETE!\n');
fprintf('════════════════════════════════════════════════════════\n\n');

fprintf('📋 Results Summary:\n');
fprintf('  Scenarios tested: %d\n', n_scenarios);

% ✅ CHANGED: Display in cm
fprintf('  Avg Passive roof: %.2f cm\n', mean(results.Passive.peak_roof(results.Passive.peak_roof > 0))*100);
fprintf('  Avg Fuzzy roof:   %.2f cm (%.1f%% improvement)\n', ...
    mean(results.Fuzzy.peak_roof(results.Fuzzy.peak_roof > 0))*100, ...
    mean(results.Fuzzy.improvement_roof(results.Fuzzy.improvement_roof > 0)));
fprintf('  Avg RL roof:      %.2f cm (%.1f%% improvement)\n', ...
    mean(results.RL.peak_roof(results.RL.peak_roof > 0))*100, ...
    mean(results.RL.improvement_roof(results.RL.improvement_roof > 0)));
fprintf('  Avg RL_CL roof:   %.2f cm (%.1f%% improvement)\n', ...
    mean(results.RL_CL.peak_roof(results.RL_CL.peak_roof > 0))*100, ...
    mean(results.RL_CL.improvement_roof(results.RL_CL.improvement_roof > 0)));
fprintf('\n');

%% ================================================================
%% HELPER FUNCTIONS
%% ================================================================

function [floor, freq, damping] = optimize_passive_tmd(M, K, C, F, t, om1, tmd_mass)
    % Simple passive TMD optimization
    % Uses Den Hartog formulas
    
    % For simplicity, place at top floor
    N = size(M, 1);
    floor = N;
    
    % Den Hartog optimal tuning
    mu = tmd_mass / M(floor, floor);
    freq = om1 / (1 + mu);  % Optimal frequency ratio
    damping = sqrt(3*mu / (8*(1+mu)));  % Optimal damping ratio
end

function [M2, K2, C2, F2] = augment_with_TMD(M, K, C, F, N, m_t, k_t, c_t, Nt, floor)
    % Add TMD to structure at specified floor
    
    M2 = blkdiag(M, m_t);
    K2 = blkdiag(K, 0);
    C2 = blkdiag(C, 0);
    
    % Couple TMD to floor
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
    
    % Initial acceleration
    a(:,1) = M \ (F(:,1) - C*v(:,1) - K*x(:,1));
    
    % Effective stiffness
    Khat = K + gamma/(beta*dt)*C + M/(beta*dt^2);
    [L, U, P] = lu(Khat);
    
    for k = 1:Nt-1
        xk = x(:,k);
        vk = v(:,k);
        ak = a(:,k);
        
        % Effective force
        F_eff = F(:,k+1) + ...
                M * ((1/(beta*dt^2))*xk + (1/(beta*dt))*vk + (1/(2*beta)-1)*ak) + ...
                C * ((gamma/(beta*dt))*xk + (gamma/beta-1)*vk + dt*(gamma/(2*beta)-1)*ak);
        
        % Solve
        y = L \ (P * F_eff);
        x(:,k+1) = U \ y;
        
        % Update velocity and acceleration
        a(:,k+1) = (1/(beta*dt^2))*(x(:,k+1) - xk) - (1/(beta*dt))*vk - (1/(2*beta)-1)*ak;
        v(:,k+1) = vk + dt*((1-gamma)*ak + gamma*a(:,k+1));
    end
    
    % Clean up numerical errors
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
    % Compute Demand-to-Capacity Ratio
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
        
        % Interpolate missing values
        if sum(dropout_mask) > 1  % Need at least 2 points to interpolate
            roof_d(~dropout_mask) = interp1(find(dropout_mask), roof_d(dropout_mask), find(~dropout_mask), 'linear', 'extrap');
            roof_v(~dropout_mask) = interp1(find(dropout_mask), roof_v(dropout_mask), find(~dropout_mask), 'linear', 'extrap');
            tmd_d(~dropout_mask) = interp1(find(dropout_mask), tmd_d(dropout_mask), find(~dropout_mask), 'linear', 'extrap');
            tmd_v(~dropout_mask) = interp1(find(dropout_mask), tmd_v(dropout_mask), find(~dropout_mask), 'linear', 'extrap');
        end
    end
end

function forces = get_forces_from_api(roof_disp, roof_vel, tmd_disp, tmd_vel, API_URL, controller_type)
    % Get control forces from API (batch prediction)
    
    % Prepare batch request
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
    
    % Select endpoint, endpoint = [API_URL '/fuzzylogic-batch'];
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
    
    % Call API response = webwrite(endpoint, json_data, options);
    response = webwrite(endpoint, json_data, options);
     switch controller_type
        case 'fuzzy'
            forces = response.forces;
        case 'rl'
            forces = response.forces;
        case 'rl_cl'
            forces = response.forces_kN;
        otherwise
            error('Unknown controller type: %s', controller_type);
    end
    
    forces = forces(:);  % Ensure column vector
end

function display_results_table(results, scenarios)
    % Display formatted results table
    
    fprintf('\n');
    fprintf('════════════════════════════════════════════════════════════════════════════════════════════════════\n');
    fprintf('                           COMPARISON RESULTS (4 CONTROLLERS)\n');
    fprintf('════════════════════════════════════════════════════════════════════════════════════════════════════\n\n');
    
    % ✅ CHANGED: Header shows (cm) instead of (m)
    fprintf('%-10s | %10s | %12s | %12s | %12s | %12s\n', ...
            'Scenario', 'Passive', 'Fuzzy', 'RL Base', 'RL CL', 'Winner');
    fprintf('%-10s | %10s | %12s | %12s | %12s | %12s\n', ...
            '', '(cm)', '(cm / %)', '(cm / %)', '(cm / %)', '');
    fprintf('-----------+------------+--------------+--------------+--------------+--------------\n');
    
    for i = 1:length(scenarios)
        passive = results.Passive.peak_roof(i);
        
        if passive == 0, continue; end
        
        fuzzy = results.Fuzzy.peak_roof(i);
        fuzzy_imp = results.Fuzzy.improvement_roof(i);
        
        rl = results.RL_Base.peak_roof(i);
        rl_imp = results.RL_Base.improvement_roof(i);
        
        rl_cl = results.RL_CL.peak_roof(i);
        rl_cl_imp = results.RL_CL.improvement_roof(i);
        
        % Find best
        [best_imp, best_idx] = max([fuzzy_imp, rl_imp, rl_cl_imp]);
        best_names = {'Fuzzy', 'RL Base', 'RL CL'};
        best = best_names{best_idx};
        
        % ✅ CHANGED: Display in cm (multiply by 100)
        fprintf('%-10s | %7.2f   | %6.2f/%4.1f%% | %6.2f/%4.1f%% | %6.2f/%4.1f%% | %12s\n', ...
                scenarios{i, 1}, passive*100, ...
                fuzzy*100, fuzzy_imp, rl*100, rl_imp, rl_cl*100, rl_cl_imp, best);
    end
    
    fprintf('-----------+------------+--------------+--------------+--------------+--------------\n');
    
    % Averages
    valid = results.Passive.peak_roof > 0;
    
    % ✅ CHANGED: Display averages in cm
    fprintf('%-10s | %7.2f   | %6.2f/%4.1f%% | %6.2f/%4.1f%% | %6.2f/%4.1f%% | %12s\n', ...
            'AVERAGE', ...
            mean(results.Passive.peak_roof(valid))*100, ...
            mean(results.Fuzzy.peak_roof(valid))*100, mean(results.Fuzzy.improvement_roof(valid)), ...
            mean(results.RL.peak_roof(valid))*100, mean(results.RL.improvement_roof(valid)), ...
            mean(results.RL_CL.peak_roof(valid))*100, mean(results.RL_CL.improvement_roof(valid)), ...
            '-');
    
    fprintf('\n');
    
    % Force statistics (already in kN, no change needed)
    fprintf('FORCE STATISTICS:\n');
    fprintf('%-15s | %12s | %12s | %15s\n', 'Controller', 'Avg (kN)', 'Max (kN)', 'Efficiency');
    fprintf('----------------+--------------+--------------+-----------------\n');
    fprintf('%-15s | %12.1f | %12.1f | %15.3f %%/kN\n', 'Fuzzy', ...
            mean(results.Fuzzy.mean_force(valid)), ...
            mean(results.Fuzzy.peak_force(valid)), ...
            mean(results.Fuzzy.improvement_roof(valid)) / mean(results.Fuzzy.mean_force(valid)));
    fprintf('%-15s | %12.1f | %12.1f | %15.3f %%/kN\n', 'RL Baseline', ...
            mean(results.RL.mean_force(valid)), ...
            mean(results.RL.peak_force(valid)), ...
            mean(results.RL.improvement_roof(valid)) / mean(results.RL.mean_force(valid)));
    fprintf('%-15s | %12.1f | %12.1f | %15.3f %%/kN\n', 'RL CL', ...
            mean(results.RL_CL.mean_force(valid)), ...
            mean(results.RL_CL.peak_force(valid)), ...
            mean(results.RL_CL.improvement_roof(valid)) / mean(results.RL_CL.mean_force(valid)));
    
    fprintf('\n════════════════════════════════════════════════════════════════════════════════════════════════════\n\n');
end
% ```
% 
% ---
% 
%  📋 **SUMMARY OF ALL FIXES:**
% 
% 1. **✅ Fix unit conversion:** Multiply API forces by 1000 (kN → N)
% 2. **✅ Fix controller names:** Change `'Perfect_RL'` to `'RL', 'RL_CL'`
% 3. **✅ Fix variable names:** Use `results.RL` and `results.RL_CL` consistently
% 4. **✅ Fix display function:** Show all 4 controllers in table
% 5. **✅ Fix improvements calculation:** Include RL and RL_CL
% 
% ---
% 
% ## 🎯 **EXPECTED RESULTS AFTER FIX:**
% ```
% Scenario   | Passive  | Fuzzy       | RL Base     | RL CL       | Winner
% -----------|----------|-------------|-------------|-------------|----------
% TEST3      | 0.0491   | 0.0407/17%  | 0.0385/22%  | 0.0370/25%  | RL CL
% TEST4      | 0.4228   | 0.3182/25%  | 0.2870/32%  | 0.2800/34%  | RL CL
% ...



function display_results_table_OLD(results, scenarios)
    % Display formatted results table
    
    fprintf('\n');
    fprintf('════════════════════════════════════════════════════════════════════════════════════════\n');
    fprintf('                           COMPARISON RESULTS\n');
    fprintf('════════════════════════════════════════════════════════════════════════════════════════\n\n');
    
    fprintf('%-10s | %12s | %12s | %12s | %12s | %10s\n', ...
            'Scenario', 'Passive', 'Fuzzy', 'Perfect RL', 'Best', 'Winner');
    fprintf('%-10s | %12s | %12s | %12s | %12s | %10s\n', ...
            '', '(m)', '(m / %)', '(m / %)', 'Improv', '');
    fprintf('-----------+--------------+--------------+--------------+--------------+------------\n');
    
    for i = 1:length(scenarios)
        passive = results.Passive.peak_roof(i);
        
        if passive == 0, continue; end
        
        fuzzy = results.Fuzzy.peak_roof(i);
        fuzzy_imp = results.Fuzzy.improvement_roof(i);
        
        rl = results.Perfect_RL.peak_roof(i);
        rl_imp = results.Perfect_RL.improvement_roof(i);
        
        % Find best
        [best_imp, best_idx] = max([fuzzy_imp, rl_imp]);
        best_names = {'Fuzzy', 'Perfect RL'};
        best = best_names{best_idx};
        
        fprintf('%-10s | %7.4f     | %6.4f/%4.1f%% | %6.4f/%4.1f%% | %10.1f%% | %10s\n', ...
                scenarios{i, 1}, passive, ...
                fuzzy, fuzzy_imp, rl, rl_imp, best_imp, best);
    end
    
    fprintf('-----------+--------------+--------------+--------------+--------------+------------\n');
    
    % Averages
    valid = results.Passive.peak_roof > 0;
    fprintf('%-10s | %7.4f     | %6.4f/%4.1f%% | %6.4f/%4.1f%% | %10s | %10s\n', ...
            'AVERAGE', ...
            mean(results.Passive.peak_roof(valid)), ...
            mean(results.Fuzzy.peak_roof(valid)), mean(results.Fuzzy.improvement_roof(valid)), ...
            mean(results.Perfect_RL.peak_roof(valid)), mean(results.Perfect_RL.improvement_roof(valid)), ...
            '-', '-');
    
    fprintf('\n');
    
    % Force statistics
    fprintf('FORCE STATISTICS:\n');
    fprintf('%-15s | %12s | %12s | %15s\n', 'Controller', 'Avg (kN)', 'Max (kN)', 'Efficiency');
    fprintf('----------------+--------------+--------------+-----------------\n');
    fprintf('%-15s | %12.1f | %12.1f | %15.3f %%/kN\n', 'Fuzzy', ...
            mean(results.Fuzzy.mean_force(valid)), ...
            mean(results.Fuzzy.peak_force(valid)), ...
            mean(results.Fuzzy.improvement_roof(valid)) / mean(results.Fuzzy.mean_force(valid)));
    fprintf('%-15s | %12.1f | %12.1f | %15.3f %%/kN\n', 'Perfect RL', ...
            mean(results.Perfect_RL.mean_force(valid)), ...
            mean(results.Perfect_RL.peak_force(valid)), ...
            mean(results.Perfect_RL.improvement_roof(valid)) / mean(results.Perfect_RL.mean_force(valid)));
    
    fprintf('\n════════════════════════════════════════════════════════════════════════════════════════\n\n');
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
    
    % ✅ CHANGED: Convert to cm for plotting
    bar(x - 1.5*width, results.Passive.peak_roof(valid)*100, width, 'FaceColor', [0.7 0.7 0.7]);
    hold on;
    bar(x - 0.5*width, results.Fuzzy.peak_roof(valid)*100, width, 'FaceColor', [1 0.6 0]);
    bar(x + 0.5*width, results.RL.peak_roof(valid)*100, width, 'FaceColor', [0.3 0.5 0.8]);
    bar(x + 1.5*width, results.RL_CL.peak_roof(valid)*100, width, 'FaceColor', [0.2 0.8 0.2]);
    hold off;
    
    set(gca, 'XTick', 1:n_scenarios, 'XTickLabel', scenario_labels);
    xtickangle(45);
    ylabel('Peak Roof Displacement (cm)');  % ✅ CHANGED: (cm)
    title('Peak Roof Displacement Comparison');
    legend({'Passive', 'Fuzzy', 'RL Base', 'RL CL'}, 'Location', 'best');
    grid on;
    
    % Plot 2: Improvement vs Passive
    subplot(2, 3, 2);
    bar(x - width, results.Fuzzy.improvement_roof(valid), width, 'FaceColor', [1 0.6 0]);
    hold on;
    bar(x, results.RL.improvement_roof(valid), width, 'FaceColor', [0.3 0.5 0.8]);
    bar(x + width, results.RL_CL.improvement_roof(valid), width, 'FaceColor', [0.2 0.8 0.2]);
    hold off;
    
    set(gca, 'XTick', 1:n_scenarios, 'XTickLabel', scenario_labels);
    xtickangle(45);
    ylabel('Improvement (%)');
    title('Improvement vs Passive TMD');
    legend({'Fuzzy', 'RL Base', 'RL CL'}, 'Location', 'best');
    grid on;
    yline(0, 'k--');
    
    % Plot 3: DCR Comparison
    subplot(2, 3, 3);
    bar(x - 1.5*width, results.Passive.DCR(valid), width, 'FaceColor', [0.7 0.7 0.7]);
    hold on;
    bar(x - 0.5*width, results.Fuzzy.DCR(valid), width, 'FaceColor', [1 0.6 0]);
    bar(x + 0.5*width, results.RL.DCR(valid), width, 'FaceColor', [0.3 0.5 0.8]);
    bar(x + 1.5*width, results.RL_CL.DCR(valid), width, 'FaceColor', [0.2 0.8 0.2]);
    hold off;
    
    set(gca, 'XTick', 1:n_scenarios, 'XTickLabel', scenario_labels);
    xtickangle(45);
    ylabel('DCR');
    title('Demand-to-Capacity Ratio');
    legend({'Passive', 'Fuzzy', 'RL Base', 'RL CL'}, 'Location', 'best');
    grid on;
    
    % Plot 4: Force Efficiency
    subplot(2, 3, 4);
    scatter(mean(results.Fuzzy.mean_force(valid)), mean(results.Fuzzy.improvement_roof(valid)), ...
            200, 'o', 'MarkerEdgeColor', [1 0.6 0], 'MarkerFaceColor', [1 0.6 0], 'LineWidth', 2);
    hold on;
    scatter(mean(results.RL.mean_force(valid)), mean(results.RL.improvement_roof(valid)), ...
            200, 's', 'MarkerEdgeColor', [0.3 0.5 0.8], 'MarkerFaceColor', [0.3 0.5 0.8], 'LineWidth', 2);
    scatter(mean(results.RL_CL.mean_force(valid)), mean(results.RL_CL.improvement_roof(valid)), ...
            200, 'd', 'MarkerEdgeColor', [0.2 0.8 0.2], 'MarkerFaceColor', [0.2 0.8 0.2], 'LineWidth', 2);
    hold off;
    
    xlabel('Average Force (kN)');
    ylabel('Average Improvement (%)');
    title('Force Efficiency');
    legend({'Fuzzy', 'RL Base', 'RL CL'}, 'Location', 'best');
    grid on;
    
    % Plot 5: Max Drift Comparison
    subplot(2, 3, 5);
    
    % ✅ CHANGED: Already multiply by 100 for cm (keep as is)
    bar(x - 1.5*width, results.Passive.max_drift(valid)*100, width, 'FaceColor', [0.7 0.7 0.7]);
    hold on;
    bar(x - 0.5*width, results.Fuzzy.max_drift(valid)*100, width, 'FaceColor', [1 0.6 0]);
    bar(x + 0.5*width, results.RL.max_drift(valid)*100, width, 'FaceColor', [0.3 0.5 0.8]);
    bar(x + 1.5*width, results.RL_CL.max_drift(valid)*100, width, 'FaceColor', [0.2 0.8 0.2]);
    hold off;
    
    set(gca, 'XTick', 1:n_scenarios, 'XTickLabel', scenario_labels);
    xtickangle(45);
    ylabel('Max Drift (cm)');
    title('Maximum Inter-Story Drift');
    legend({'Passive', 'Fuzzy', 'RL Base', 'RL CL'}, 'Location', 'best');
    grid on;
    
    % Plot 6: Robustness (TEST6 scenarios)
    subplot(2, 3, 6);
    test6_idx = find(valid & startsWith(scenarios(:, 1), 'TEST6'));
    
    if ~isempty(test6_idx)
        test3_idx_valid = find(valid & strcmp(scenarios(:, 1), 'TEST3'));
        
        if ~isempty(test3_idx_valid)
            test3_global = find(strcmp(scenarios(:, 1), 'TEST3'));
            
            fuzzy_baseline = results.Fuzzy.peak_roof(test3_global);
            rl_baseline = results.RL.peak_roof(test3_global);
            rl_cl_baseline = results.RL_CL.peak_roof(test3_global);
            
            fuzzy_deg = (results.Fuzzy.peak_roof(test6_idx) - fuzzy_baseline) / fuzzy_baseline * 100;
            rl_deg = (results.RL.peak_roof(test6_idx) - rl_baseline) / rl_baseline * 100;
            rl_cl_deg = (results.RL_CL.peak_roof(test6_idx) - rl_cl_baseline) / rl_cl_baseline * 100;
            
            stress_labels = scenarios(test6_idx, 1);
            x_stress = 1:length(test6_idx);
            
            plot(x_stress, fuzzy_deg, 'o-', 'Color', [1 0.6 0], 'LineWidth', 2, 'MarkerSize', 8);
            hold on;
            plot(x_stress, rl_deg, 's-', 'Color', [0.3 0.5 0.8], 'LineWidth', 2, 'MarkerSize', 8);
            plot(x_stress, rl_cl_deg, 'd-', 'Color', [0.2 0.8 0.2], 'LineWidth', 2, 'MarkerSize', 8);
            hold off;
            
            set(gca, 'XTick', 1:length(test6_idx), 'XTickLabel', stress_labels);
            xtickangle(45);
            ylabel('Performance Degradation (%)');
            title('Robustness Under Perturbations');
            legend({'Fuzzy', 'RL Base', 'RL CL'}, 'Location', 'best');
            grid on;
            yline(0, 'k--');
        end
    end
    
    sgtitle('4-Way TMD Controller Comparison (Passive, Fuzzy, RL Baseline, RL CL)', 'FontSize', 14, 'FontWeight', 'bold');
    
    % Save figure
    saveas(fig, 'comparison_passive_fuzzy_rl.png');
end
