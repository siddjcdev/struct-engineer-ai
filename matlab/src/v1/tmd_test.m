
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

% Check earthquake files
fprintf('  [2/2] Checking earthquake files... \n');
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
controllers = {'Passive', 'Fuzzy', 'Perfect_RL'};

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

    fprintf('Find optimal passive TMD parameters, passive_floor: %f ,passive_floor: %f, passive_damping: %f \n',passive_floor,passive_freq,passive_damping)
    
    % Build passive TMD system
    k_tmd_passive = tmd_mass * passive_freq^2;
    c_tmd_passive = 2 * passive_damping * sqrt(k_tmd_passive * tmd_mass);
    
    [M_passive, K_passive, C_passive, F_passive] = augment_with_TMD(M, K, C, Fg, N, tmd_mass, k_tmd_passive, c_tmd_passive, Nt, passive_floor);
    
    % Simulate
    [x_passive, v_passive, a_passive] = newmark_simulate(M_passive, C_passive, K_passive, F_passive, t);
    
    % Extract results
    roof_passive = x_passive(N, :);
    drift_passive = compute_interstory_drifts(x_passive(1:N, :));
    
    results.Passive.peak_roof(scenario_idx) = max(abs(roof_passive));
    results.Passive.max_drift(scenario_idx) = max(abs(drift_passive(:)));
    results.Passive.DCR(scenario_idx) = compute_DCR(drift_passive);
    results.Passive.rms_roof(scenario_idx) = rms(roof_passive);
    results.Passive.peak_force(scenario_idx) = 0;  % Passive has no active force
    results.Passive.mean_force(scenario_idx) = 0;
    results.Passive.time(scenario_idx) = toc;
    
    fprintf('Peak: %.4f m (%.2f s)\n', results.Passive.peak_roof(scenario_idx), results.Passive.time(scenario_idx));
    
   
    fprintf('\n');

end

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




