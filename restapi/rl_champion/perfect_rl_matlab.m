% PERFECT RL MATLAB INTEGRATION
% ==============================
% 
% Functions to call Perfect RL API from MATLAB
%
% Author: Siddharth
% Date: December 2025

% ================================================================
% SINGLE PREDICTION
% ================================================================

function force_N = perfect_rl_predict(API_URL, roof_disp, roof_vel, tmd_disp, tmd_vel)
    % Get control force from Perfect RL API (single prediction)
    %
    % Inputs:
    %   API_URL: Base URL (e.g., 'http://localhost:8000')
    %   roof_disp: Roof displacement (m)
    %   roof_vel: Roof velocity (m/s)
    %   tmd_disp: TMD displacement (m)
    %   tmd_vel: TMD velocity (m/s)
    %
    % Output:
    %   force_N: Control force (N)
    
    % Prepare request
    data = struct(...
        'roof_displacement', roof_disp, ...
        'roof_velocity', roof_vel, ...
        'tmd_displacement', tmd_disp, ...
        'tmd_velocity', tmd_vel ...
    );
    
    json_data = jsonencode(data);
    
    options = weboptions(...
        'MediaType', 'application/json', ...
        'ContentType', 'json', ...
        'Timeout', 30 ...
    );
    
    % Call API
    try
        response = webwrite([API_URL '/predict'], json_data, options);
        force_N = response.force_N;
        
    catch ME
        error('Perfect RL API error: %s', ME.message);
    end
end


% ================================================================
% BATCH PREDICTION
% ================================================================

function forces_N = perfect_rl_predict_batch(API_URL, roof_disp, roof_vel, tmd_disp, tmd_vel)
    % Get control forces from Perfect RL API (batch prediction)
    %
    % Inputs:
    %   API_URL: Base URL (e.g., 'http://localhost:8000')
    %   roof_disp: Roof displacement array (m)
    %   roof_vel: Roof velocity array (m/s)
    %   tmd_disp: TMD displacement array (m)
    %   tmd_vel: TMD velocity array (m/s)
    %
    % Output:
    %   forces_N: Control forces array (N)
    
    % Ensure column vectors
    roof_disp = roof_disp(:);
    roof_vel = roof_vel(:);
    tmd_disp = tmd_disp(:);
    tmd_vel = tmd_vel(:);
    
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
        'Timeout', 120 ...
    );
    
    % Call API
    try
        response = webwrite([API_URL '/predict-batch'], json_data, options);
        forces_N = response.forces_N;
        
        % Ensure column vector
        forces_N = forces_N(:);
        
    catch ME
        error('Perfect RL API batch error: %s', ME.message);
    end
end


% ================================================================
% HEALTH CHECK
% ================================================================

function status = perfect_rl_health(API_URL)
    % Check if Perfect RL API is healthy
    %
    % Input:
    %   API_URL: Base URL (e.g., 'http://localhost:8000')
    %
    % Output:
    %   status: Struct with health info
    
    options = weboptions('Timeout', 10);
    
    try
        status = webread([API_URL '/health'], options);
        fprintf('✅ Perfect RL API: %s\n', status.status);
        fprintf('   Model: %s\n', status.model);
        fprintf('   Performance: %s\n', status.performance);
        
    catch ME
        error('Perfect RL API health check failed: %s', ME.message);
    end
end


% ================================================================
% MODEL INFO
% ================================================================

function info = perfect_rl_info(API_URL)
    % Get Perfect RL model information
    %
    % Input:
    %   API_URL: Base URL (e.g., 'http://localhost:8000')
    %
    % Output:
    %   info: Struct with model info
    
    options = weboptions('Timeout', 10);
    
    try
        info = webread([API_URL '/info'], options);
        
        fprintf('\n📊 Perfect RL Model Info:\n');
        fprintf('   Name: %s\n', info.name);
        fprintf('   Type: %s\n', info.type);
        fprintf('   Training: %s\n\n', info.training);
        
        fprintf('   Performance:\n');
        perf_fields = fieldnames(info.performance);
        for i = 1:length(perf_fields)
            field = perf_fields{i};
            fprintf('      %s: %s\n', field, info.performance.(field));
        end
        
        fprintf('\n   Comparison:\n');
        comp_fields = fieldnames(info.comparison);
        for i = 1:length(comp_fields)
            field = comp_fields{i};
            fprintf('      %s: %s\n', field, info.comparison.(field));
        end
        fprintf('\n');
        
    catch ME
        error('Perfect RL API info request failed: %s', ME.message);
    end
end


% ================================================================
% EXAMPLE USAGE
% ================================================================

function example_usage()
    % Example of how to use Perfect RL API in MATLAB
    
    fprintf('\n=================================================================\n');
    fprintf('  PERFECT RL API - MATLAB EXAMPLE\n');
    fprintf('=================================================================\n\n');
    
    % API URL
    API_URL = 'http://localhost:8000';
    
    % 1. Health check
    fprintf('1. Checking API health...\n');
    try
        status = perfect_rl_health(API_URL);
    catch
        fprintf('❌ API not running! Start it with: python perfect_rl_api.py\n');
        return;
    end
    
    % 2. Get model info
    fprintf('\n2. Getting model info...\n');
    info = perfect_rl_info(API_URL);
    
    % 3. Single prediction
    fprintf('\n3. Testing single prediction...\n');
    roof_disp = 0.15;
    roof_vel = 0.8;
    tmd_disp = 0.16;
    tmd_vel = 0.9;
    
    force = perfect_rl_predict(API_URL, roof_disp, roof_vel, tmd_disp, tmd_vel);
    fprintf('   Input: roof_disp=%.2f, roof_vel=%.2f\n', roof_disp, roof_vel);
    fprintf('   Output force: %.2f N (%.2f kN)\n', force, force/1000);
    
    % 4. Batch prediction
    fprintf('\n4. Testing batch prediction...\n');
    n = 100;
    roof_disp_batch = 0.2 * randn(n, 1);
    roof_vel_batch = 1.0 * randn(n, 1);
    tmd_disp_batch = roof_disp_batch + 0.05 * randn(n, 1);
    tmd_vel_batch = roof_vel_batch + 0.1 * randn(n, 1);
    
    tic;
    forces = perfect_rl_predict_batch(API_URL, roof_disp_batch, roof_vel_batch, ...
                                      tmd_disp_batch, tmd_vel_batch);
    elapsed = toc;
    
    fprintf('   Batch size: %d\n', n);
    fprintf('   Time: %.3f seconds\n', elapsed);
    fprintf('   Time per prediction: %.2f ms\n', elapsed/n * 1000);
    fprintf('   Force range: [%.2f, %.2f] kN\n', min(forces)/1000, max(forces)/1000);
    fprintf('   Mean force: %.2f kN\n', mean(forces)/1000);
    
    fprintf('\n✅ All tests passed!\n\n');
    fprintf('=================================================================\n\n');
end


% ================================================================
% SIMULATION INTEGRATION EXAMPLE
% ================================================================

function example_simulation_integration()
    % Example: Integrate Perfect RL into building simulation
    
    fprintf('\n=================================================================\n');
    fprintf('  PERFECT RL - SIMULATION INTEGRATION EXAMPLE\n');
    fprintf('=================================================================\n\n');
    
    API_URL = 'http://localhost:8000';
    
    % Simulation parameters
    dt = 0.02;  % Time step (s)
    T = 20;     % Total time (s)
    n_steps = T / dt;
    
    fprintf('Setting up simulation...\n');
    fprintf('   Time step: %.3f s\n', dt);
    fprintf('   Duration: %.1f s\n', T);
    fprintf('   Steps: %d\n\n', n_steps);
    
    % Example earthquake (simple sine wave)
    t = (0:n_steps-1) * dt;
    earthquake = 3.0 * sin(2*pi*1.5*t) .* exp(-0.1*t);
    
    % Initialize state
    roof_disp = zeros(n_steps, 1);
    roof_vel = zeros(n_steps, 1);
    tmd_disp = zeros(n_steps, 1);
    tmd_vel = zeros(n_steps, 1);
    forces = zeros(n_steps, 1);
    
    fprintf('Running simulation with Perfect RL control...\n');
    
    % Simulation loop (simplified)
    for i = 2:n_steps
        % Get control force from Perfect RL
        forces(i) = perfect_rl_predict(API_URL, ...
                                       roof_disp(i-1), roof_vel(i-1), ...
                                       tmd_disp(i-1), tmd_vel(i-1));
        
        % Update dynamics (simplified - replace with your actual dynamics)
        % This is just a placeholder
        roof_vel(i) = roof_vel(i-1) - 0.5*roof_disp(i-1) + earthquake(i);
        roof_disp(i) = roof_disp(i-1) + roof_vel(i)*dt;
        
        tmd_vel(i) = tmd_vel(i-1) + forces(i)/6000;  % mass = 6000 kg
        tmd_disp(i) = tmd_disp(i-1) + tmd_vel(i)*dt;
        
        % Progress
        if mod(i, 100) == 0
            fprintf('   Step %d/%d (%.1f%%)\n', i, n_steps, i/n_steps*100);
        end
    end
    
    % Results
    peak_disp = max(abs(roof_disp)) * 100;  % cm
    mean_force = mean(abs(forces)) / 1000;  % kN
    
    fprintf('\n📊 Results:\n');
    fprintf('   Peak displacement: %.2f cm\n', peak_disp);
    fprintf('   Mean force: %.2f kN\n', mean_force);
    
    % Plot
    figure('Position', [100 100 1200 600]);
    
    subplot(2,1,1);
    plot(t, roof_disp*100, 'LineWidth', 2);
    xlabel('Time (s)');
    ylabel('Displacement (cm)');
    title('Roof Displacement with Perfect RL Control');
    grid on;
    
    subplot(2,1,2);
    plot(t, forces/1000, 'LineWidth', 2);
    xlabel('Time (s)');
    ylabel('Force (kN)');
    title('Perfect RL Control Force');
    grid on;
    
    fprintf('\n✅ Simulation complete!\n\n');
    fprintf('=================================================================\n\n');
end


% ================================================================
% RUN EXAMPLES
% ================================================================

% Uncomment to run examples:
% example_usage();
% example_simulation_integration();
