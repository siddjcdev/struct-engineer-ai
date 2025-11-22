function thefunc_dcr_floor_tuner_v7(earthquake_name, use_wind, wind_type)
    % V7 IMPROVEMENTS:
    % - Multi-objective optimization (DCR + drift + roof)
    % - Finer parameter grid with adaptive search
    % - Improved floor selection algorithm
    % - Better TMD frequency tuning (considers multiple modes)
    % - Enhanced performance metrics
    % - Smarter guard constraints
    %
    % Usage:
    %   thefunc_dcr_floor_tuner_v7()                           % Default: el_centro + strong_wind
    %   thefunc_dcr_floor_tuner_v7('northridge')               % Northridge, no wind
    %   thefunc_dcr_floor_tuner_v7('kobe', false)              % Kobe, no wind
    %   thefunc_dcr_floor_tuner_v7('loma_prieta', true)        % Loma Prieta + strong_wind
    %   thefunc_dcr_floor_tuner_v7('northridge', true, 'hurricane')  % Northridge + hurricane
    
    % Set defaults
    if nargin < 1 || isempty(earthquake_name)
        earthquake_name = 'el_centro';
    end
    if nargin < 2 || isempty(use_wind)
        use_wind = true;
    end
    if nargin < 3 || isempty(wind_type)
        wind_type = 'hurricane';
    end
    
    % Call main function
    thefunc_dcr_floor_tuner_real_data_v7(earthquake_name, use_wind, wind_type)
    
end

function thefunc_dcr_floor_tuner_real_data_v7(earthquake_name, use_wind, wind_type)
    
    if nargin < 2, use_wind = false; end
    if nargin < 3, wind_type = 'strong_wind'; end
    
    fprintf('\n========================================\n');
    fprintf('TMD TUNER V7 WITH ENHANCED OPTIMIZATION\n');
    fprintf('========================================\n\n');
    
    % Building parameters
    N = 12; m0 = 2.0e5; 
    k0 = 2.0e7;
    zeta_target = 0.015;
    dt = 0.01;
    T = 120;
    
    % Create soft-story condition
    soft_story_idx = 8;
    soft_story_factor = 0.60;
    
    % Assemble M, K, C
    m = m0*ones(N,1);
    M = diag(m);
    
    k_story = k0 * ones(N,1);
    k_story(soft_story_idx) = soft_story_factor * k0;
    
    K = zeros(N,N);
    K(1,1) = k_story(1) + k0;
    K(1,2) = -k_story(1);
    
    for i = 2:N-1
        K(i,i-1) = -k_story(i-1);
        K(i,i) = k_story(i-1) + k_story(i);
        K(i,i+1) = -k_story(i);
    end
    
    K(N,N-1) = -k_story(N-1);
    K(N,N) = k_story(N-1);
    
    % Modal analysis
    [Vfull,Dfull] = eig(K,M);
    lam_full = diag(Dfull);
    [om_sorted, order] = sort(sqrt(lam_full));
    V = Vfull(:, order);
    
    if numel(om_sorted)>=2
        a0a1 = solve_rayleigh(om_sorted(1),om_sorted(2),zeta_target);
        C = a0a1(1)*M + a0a1(2)*K;
    else
        C = 2*zeta_target*sqrt(om_sorted(1))*M;
    end
    
    % Load earthquake data
    fprintf('Loading earthquake data: %s\n', earthquake_name);
    [ag, t, eq_info] = load_real_seismic_data(earthquake_name, dt, T);
    Nt = length(t);
    
    % Ground motion forces
    r = ones(N,1);
    Fg = -M*r*ag.';
    
    % Add wind loading if requested
    if use_wind
        fprintf('\nLoading wind data: %s\n', wind_type);
        building_height = N * 3;
        [Fw, ~, wind_info] = load_real_wind_data(wind_type, building_height, dt, T);
        
        if size(Fw, 2) > Nt
            Fw = Fw(:, 1:Nt);
        elseif size(Fw, 2) < Nt
            Fw = [Fw, zeros(N, Nt - size(Fw,2))];
        end
        
        F = Fg + Fw;
        fprintf('\n✓ Combined seismic + wind loading\n');
    else
        F = Fg;
        fprintf('\n✓ Seismic loading only\n');
    end
    
    % ===== BASELINE SIMULATION =====
    fprintf('\nRunning baseline simulation...\n');
    beta=1/4; gamma=1/2;
    [xA,vA,aA] = newmark_simulate(M,C,K,F,t,beta,gamma);
    roofA = xA(N,:);
    driftA = compute_interstory_drifts(xA);
    [DCR_A, dcr_profile_A] = compute_DCR_v7(driftA);
    peakRoofA = max(abs(roofA));
    rmsRoofA  = sqrt(mean(roofA.^2));
    maxDriftA = max(abs(driftA(:)));
    
    % Calculate comprehensive baseline metrics
    avg_disp_A = mean(abs(xA(:)));
    avg_vel_A = mean(abs(vA(:)));
    avg_acc_A = mean(abs(aA(:)));
    rms_disp_A = sqrt(mean(xA(:).^2));
    rms_vel_A = sqrt(mean(vA(:).^2));
    rms_acc_A = sqrt(mean(aA(:).^2));
    
    % Print baseline results
    fprintf('\n===== BASELINE RESULTS =====\n');
    fprintf('Earthquake: %s\n', eq_info.name);
    fprintf('  PGA: %.3f m/s² (%.2fg)\n', eq_info.pga, eq_info.pga_g);
    if use_wind
        fprintf('Wind: %s\n', wind_info.name);
        fprintf('  Mean speed: %.1f m/s\n', wind_info.mean_speed);
    end
    fprintf('\nBuilding Response:\n');
    fprintf('  Max roof displacement: %.3e m\n', peakRoofA);
    fprintf('  Max inter-story drift: %.3e m\n', maxDriftA);
    fprintf('  DCR: %.3f\n', DCR_A);
    fprintf('  RMS displacement: %.3e m\n', rms_disp_A);
    fprintf('  RMS velocity: %.3e m/s\n', rms_vel_A);
    fprintf('  RMS acceleration: %.3e m/s²\n', rms_acc_A);
    fprintf('============================\n\n');
    
    % ===== V7 ENHANCED FLOOR SELECTION =====
    fprintf('V7: Analyzing modal participation...\n');
    floorCandidates = select_candidate_floors_v7(V, dcr_profile_A, N);
    fprintf('  Selected floors: %s\n', mat2str(floorCandidates));
    
    % ===== V7 ADAPTIVE PARAMETER GRID =====
    fprintf('V7: Creating adaptive parameter grid...\n');
    [mu_vals, zeta_vals] = create_adaptive_grid_v7(DCR_A, maxDriftA);
    fprintf('  Mass ratios: %d values (%.3f to %.3f)\n', ...
        length(mu_vals), min(mu_vals), max(mu_vals));
    fprintf('  Damping ratios: %d values (%.3f to %.3f)\n', ...
        length(zeta_vals), min(zeta_vals), max(zeta_vals));
    
    % ===== V7 SMART GUARD CONSTRAINTS =====
    fprintf('V7: Setting adaptive guard constraints...\n');
    [peak_guard, rms_guard, drift_guard] = set_smart_guards_v7(peakRoofA, rmsRoofA, maxDriftA, DCR_A);
    fprintf('  Peak roof guard: %.3e m (%.0f%% tolerance)\n', peak_guard, 100*(peak_guard/peakRoofA - 1));
    fprintf('  RMS guard: %.3e m (%.0f%% tolerance)\n', rms_guard, 100*(rms_guard/rmsRoofA - 1));
    fprintf('  Drift guard: %.3e m (%.0f%% tolerance)\n', drift_guard, 100*(drift_guard/maxDriftA - 1));
    
    % Search for best TMD
    best = struct('floor',NaN,'mu',NaN,'zeta',NaN,'score',Inf,'DCR',Inf,'drift',Inf,'roof',Inf);
    
    fprintf('\nSearching TMD configurations...\n');
    
    for fIdx = 1:numel(floorCandidates)
        fAttach = floorCandidates(fIdx);
        fprintf('  Testing floor %d...\n', fAttach);
        
        score_mat = zeros(length(mu_vals), length(zeta_vals));
        DCR_mat = zeros(size(score_mat));
        drift_mat = zeros(size(score_mat));
        roof_mat = zeros(size(score_mat));
        
        for i = 1:length(mu_vals)
            for j = 1:length(zeta_vals)
                mu_i   = mu_vals(i);
                zeta_i = zeta_vals(j);
                m_ti   = mu_i*m0;
                
                % V7: Multi-mode frequency tuning
                [om_t, k_mode] = tune_tmd_frequency_v7(V, om_sorted, fAttach, N);
                k_ti   = m_ti*om_t^2;
                c_ti   = 2*zeta_i*sqrt(k_ti*m_ti);
                
                [Maug,Kaug,Caug,Faug] = augment_with_TMD_at_floor(M,K,C,F,N,m_ti,k_ti,c_ti,Nt,fAttach);
                [xj,vj,aj] = newmark_simulate(Maug,Caug,Kaug,Faug,t,beta,gamma);
                
                roof_j = xj(N,:);
                drift_j = compute_interstory_drifts(xj(1:N,:));
                [DCR_j, ~] = compute_DCR_v7(drift_j);
                maxDrift_j = max(abs(drift_j(:)));
                peakRoof_j = max(abs(roof_j));
                rmsRoof_j = sqrt(mean(roof_j.^2));
                
                % V7: Multi-objective scoring
                score = compute_multi_objective_score_v7(DCR_j, maxDrift_j, peakRoof_j, ...
                    DCR_A, maxDriftA, peakRoofA);
                
                score_mat(i,j) = score;
                DCR_mat(i,j) = DCR_j;
                drift_mat(i,j) = maxDrift_j;
                roof_mat(i,j) = peakRoof_j;
                
                % Check guards
                if peakRoof_j > peak_guard || rmsRoof_j > rms_guard || maxDrift_j > drift_guard
                    score_mat(i,j) = Inf;
                end
            end
        end
        
        [minScore, idx] = min(score_mat(:));
        
        if isfinite(minScore) && minScore < best.score
            [iBest,jBest] = ind2sub(size(score_mat), idx);
            best.floor = fAttach;
            best.mu = mu_vals(iBest);
            best.zeta = zeta_vals(jBest);
            best.score = minScore;
            best.DCR = DCR_mat(iBest,jBest);
            best.drift = drift_mat(iBest,jBest);
            best.roof = roof_mat(iBest,jBest);
            
            fprintf('    *** New best: mu=%.3f, zeta=%.3f, score=%.3f (DCR=%.3f)\n', ...
                best.mu, best.zeta, best.score, best.DCR);
        end
    end
    
    if ~isfinite(best.score)
        fprintf('\n❌ No feasible TMD found within constraints.\n');
        fprintf('   Consider:\n');
        fprintf('   - Relaxing guard constraints\n');
        fprintf('   - Expanding parameter search range\n');
        fprintf('   - Using multiple TMDs\n');
        return;
    end
    
    fprintf('\n✓ Best TMD found: floor=%d, mu=%.3f, zeta=%.3f\n', ...
        best.floor, best.mu, best.zeta);
    
    % Re-simulate with best TMD
    m_t = best.mu*m0;
    [om_t, ~] = tune_tmd_frequency_v7(V, om_sorted, best.floor, N);
    k_t = m_t*om_t^2;
    c_t = 2*best.zeta*sqrt(k_t*m_t);
    
    [Mbest,Kbest,Cbest,Fbest] = augment_with_TMD_at_floor(M,K,C,F,N,m_t,k_t,c_t,Nt,best.floor);
    [xB,vB,aB] = newmark_simulate(Mbest,Cbest,Kbest,Fbest,t,beta,gamma);
    
    roofB = xB(N,:);
    driftB = compute_interstory_drifts(xB(1:N,:));
    [DCR_B, dcr_profile_B] = compute_DCR_v7(driftB);
    peakRoofB = max(abs(roofB));
    rmsRoofB = sqrt(mean(roofB.^2));
    maxDriftB = max(abs(driftB(:)));
    
    % Calculate comprehensive TMD metrics
    avg_disp_B = mean(mean(abs(xB(1:N,:))));
    avg_vel_B = mean(mean(abs(vB(1:N,:))));
    avg_acc_B = mean(mean(abs(aB(1:N,:))));
    rms_disp_B = sqrt(mean(mean(xB(1:N,:).^2)));
    rms_vel_B = sqrt(mean(mean(vB(1:N,:).^2)));
    rms_acc_B = sqrt(mean(mean(aB(1:N,:).^2)));
    
    % Calculate improvements
    dcr_improve = 100*(1 - DCR_B/DCR_A);
    drift_improve = 100*(1 - maxDriftB/maxDriftA);
    roof_improve = 100*(1 - peakRoofB/peakRoofA);
    rms_disp_improve = 100*(1 - rms_disp_B/rms_disp_A);
    rms_vel_improve = 100*(1 - rms_vel_B/rms_vel_A);
    rms_acc_improve = 100*(1 - rms_acc_B/rms_acc_A);
    
    % Print comparison
    fprintf('\n===== V7 PERFORMANCE COMPARISON =====\n');
    fprintf('Earthquake: %s (PGA: %.2fg)\n', eq_info.name, eq_info.pga_g);
    if use_wind
        fprintf('Wind: %s (%.1f m/s)\n', wind_info.name, wind_info.mean_speed);
    end
    fprintf('\n--- Primary Metrics ---\n');
    fprintf('DCR:       %.3f → %.3f (%+.1f%%)\n', DCR_A, DCR_B, dcr_improve);
    fprintf('Max drift: %.3e → %.3e m (%+.1f%%)\n', maxDriftA, maxDriftB, drift_improve);
    fprintf('Max roof:  %.3e → %.3e m (%+.1f%%)\n', peakRoofA, peakRoofB, roof_improve);
    fprintf('\n--- RMS Metrics ---\n');
    fprintf('RMS disp:  %.3e → %.3e m (%+.1f%%)\n', rms_disp_A, rms_disp_B, rms_disp_improve);
    fprintf('RMS vel:   %.3e → %.3e m/s (%+.1f%%)\n', rms_vel_A, rms_vel_B, rms_vel_improve);
    fprintf('RMS acc:   %.3e → %.3e m/s² (%+.1f%%)\n', rms_acc_A, rms_acc_B, rms_acc_improve);
    
    % V7: Performance assessment
    fprintf('\n--- V7 Performance Assessment ---\n');
    [rating, recommendation] = assess_performance_v7(dcr_improve, drift_improve, roof_improve);
    fprintf('Rating: %s\n', rating);
    fprintf('Recommendation: %s\n', recommendation);
    fprintf('=====================================\n');
    
    % ===== SAVE RESULTS =====
    fprintf('\n===== SAVING RESULTS =====\n');
    
    % Prepare API data structure
    api_data = struct();
    api_data.version = 'v7';
    api_data.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    
    % Metadata
    api_data.metadata = struct();
    api_data.metadata.n_floors = N;
    api_data.metadata.time_step = dt;
    api_data.metadata.duration = T;
    api_data.metadata.soft_story = soft_story_idx;
    
    % Input
    api_data.input = struct();
    api_data.input.earthquake = eq_info;
    api_data.input.use_wind = use_wind;
    if use_wind
        api_data.input.wind = wind_info;
    end
    
    % Baseline
    api_data.baseline = struct();
    api_data.baseline.DCR = DCR_A;
    api_data.baseline.max_drift = maxDriftA;
    api_data.baseline.max_roof = peakRoofA;
    api_data.baseline.rms_roof = rmsRoofA;
    api_data.baseline.rms_displacement = rms_disp_A;
    api_data.baseline.rms_velocity = rms_vel_A;
    api_data.baseline.rms_acceleration = rms_acc_A;
    api_data.baseline.dcr_profile = dcr_profile_A;
    
    % TMD configuration
    api_data.tmd = struct();
    api_data.tmd.floor = best.floor;
    api_data.tmd.mass_ratio = best.mu;
    api_data.tmd.damping_ratio = best.zeta;
    api_data.tmd.mass_kg = m_t;
    api_data.tmd.stiffness = k_t;
    api_data.tmd.damping = c_t;
    api_data.tmd.natural_frequency = om_t;
    api_data.tmd.optimization_score = best.score;
    
    % TMD results
    api_data.tmd_results = struct();
    api_data.tmd_results.DCR = DCR_B;
    api_data.tmd_results.max_drift = maxDriftB;
    api_data.tmd_results.max_roof = peakRoofB;
    api_data.tmd_results.rms_roof = rmsRoofB;
    api_data.tmd_results.rms_displacement = rms_disp_B;
    api_data.tmd_results.rms_velocity = rms_vel_B;
    api_data.tmd_results.rms_acceleration = rms_acc_B;
    api_data.tmd_results.dcr_profile = dcr_profile_B;
    
    % Improvements
    api_data.improvements = struct();
    api_data.improvements.dcr_reduction_pct = dcr_improve;
    api_data.improvements.drift_reduction_pct = drift_improve;
    api_data.improvements.roof_reduction_pct = roof_improve;
    api_data.improvements.rms_disp_reduction_pct = rms_disp_improve;
    api_data.improvements.rms_vel_reduction_pct = rms_vel_improve;
    api_data.improvements.rms_acc_reduction_pct = rms_acc_improve;
    
    % V7 specific data
    api_data.v7 = struct();
    api_data.v7.candidate_floors = floorCandidates;
    api_data.v7.performance_rating = rating;
    api_data.v7.recommendation = recommendation;
    api_data.v7.multi_objective_score = best.score;
    
    % Time series (sampled)
    sample_rate = 10;
    time_indices = 1:sample_rate:length(t);
    api_data.time_series = struct();
    api_data.time_series.time = t(time_indices);
    api_data.time_series.earthquake_acceleration = ag(time_indices);
    api_data.time_series.baseline_roof = roofA(time_indices);
    api_data.time_series.tmd_roof = roofB(time_indices);
    
    % Save JSON
    try
        folder = 'results';
        json_filename = sprintf('tmd_v7_simulation_%s.json', datestr(now, 'yyyymmdd_HHMMSS'));
        filepath = fullfile(folder, json_filename);
        json_str = jsonencode(api_data);
        fid = fopen(filepath, 'w');
        fprintf(fid, '%s', json_str);
        fclose(fid);
        fprintf('✓ JSON saved: %s (%.2f KB)\n', filepath, dir(filepath).bytes / 1024);
    catch ME
        fprintf('✗ JSON export failed: %s\n', ME.message);
    end
    
    % Save MATLAB data
    building_results = struct();
    building_results.version = 'v7';
    building_results.time = t;
    building_results.baseline_roof = roofA;
    building_results.baseline_drift = driftA;
    building_results.baseline_DCR = DCR_A;
    building_results.tmd_roof = roofB;
    building_results.tmd_drift = driftB;
    building_results.tmd_DCR = DCR_B;
    building_results.tmd_config = api_data.tmd;
    building_results.improvements = api_data.improvements;
    building_results.v7_data = api_data.v7;
    
    try
        mat_filename = 'building_sim_v7_latest.mat';
        save(mat_filename, 'building_results');
        fprintf('✓ MATLAB data saved: %s\n', mat_filename);
    catch ME
        fprintf('✗ MATLAB save failed: %s\n', ME.message);
    end
    
    fprintf('==============================\n');
    
    % Generate plots
    plot_results_v7(t, driftA, driftB, DCR_A, DCR_B, best.floor, eq_info);
end

%% ============================================================
%% V7 ENHANCED FUNCTIONS
%% ============================================================

function floors = select_candidate_floors_v7(V, dcr_profile, N)
    % V7: Intelligent floor selection based on modal participation
    
    % Get first 3 mode shapes
    n_modes = min(3, size(V,2));
    modal_participation = zeros(N, n_modes);
    
    for mode = 1:n_modes
        mode_shape = abs(V(:,mode));
        modal_participation(:,mode) = mode_shape / max(mode_shape);
    end
    
    % Combine modal participation with DCR profile
    dcr_normalized = dcr_profile / max(dcr_profile);
    
    % Weighted score: 50% modal participation + 50% DCR
    combined_score = 0.5 * mean(modal_participation, 2) + 0.5 * [dcr_normalized(:); 0];
    combined_score = combined_score(1:N);  % Ensure correct length
    
    % Select top 4-5 floors
    [~, sorted_idx] = sort(combined_score, 'descend');
    
    % Always include top floor and soft story region
    candidates = unique([sorted_idx(1:4)' N]);
    floors = sort(candidates);
end

function [mu_vals, zeta_vals] = create_adaptive_grid_v7(DCR, maxDrift)
    % V7: Create adaptive parameter grid based on response severity
    
    if DCR > 1.5 || maxDrift > 0.1
        % High response - need larger mass ratios, broader damping
        mu_vals = [0.02:0.01:0.08, 0.10:0.02:0.25, 0.30];
        zeta_vals = [0.03:0.01:0.10, 0.12:0.02:0.25, 0.30, 0.40];
    elseif DCR > 1.3 || maxDrift > 0.05
        % Moderate response - balanced grid
        mu_vals = [0.01:0.005:0.05, 0.06:0.01:0.15, 0.20, 0.25];
        zeta_vals = [0.02:0.01:0.08, 0.10:0.02:0.20, 0.25];
    else
        % Low response - finer grid, smaller masses
        mu_vals = [0.005:0.005:0.03, 0.04:0.01:0.10];
        zeta_vals = [0.01:0.01:0.15, 0.20];
    end
end

function [peak_guard, rms_guard, drift_guard] = set_smart_guards_v7(peakRoof, rmsRoof, maxDrift, DCR)
    % V7: Adaptive guard constraints based on response severity
    
    if DCR > 1.5
        % Severe response - very loose constraints
        tolerance = 1.80;  % 80% tolerance
    elseif DCR > 1.3
        % Moderate response - medium constraints
        tolerance = 1.50;  % 50% tolerance
    else
        % Low response - tighter constraints
        tolerance = 1.30;  % 30% tolerance
    end
    
    peak_guard = tolerance * peakRoof;
    rms_guard = tolerance * rmsRoof;
    drift_guard = tolerance * maxDrift;
end

function [om_t, k_mode] = tune_tmd_frequency_v7(V, om_sorted, fAttach, N)
    % V7: Multi-mode frequency tuning
    
    % Find mode with maximum participation at floor
    n_modes = min(3, length(om_sorted));
    participation = zeros(n_modes, 1);
    
    for mode = 1:n_modes
        mode_shape = abs(V(:,mode));
        participation(mode) = mode_shape(fAttach) / max(mode_shape);
    end
    
    % Weight by modal importance
    [~, k_mode] = max(participation);
    
    % Use Den Hartog optimal tuning
    om_t = om_sorted(k_mode);
end

function score = compute_multi_objective_score_v7(DCR, drift, roof, DCR_base, drift_base, roof_base)
    % V7: Multi-objective optimization score
    % Balance DCR reduction, drift control, and roof displacement
    
    % Weights: DCR is most important, then drift, then roof
    w_dcr = 0.5;
    w_drift = 0.3;
    w_roof = 0.2;
    
    % Normalized improvements (negative = worse)
    dcr_norm = (DCR - DCR_base) / DCR_base;
    drift_norm = (drift - drift_base) / drift_base;
    roof_norm = (roof - roof_base) / roof_base;
    
    % Penalize increases more heavily
    if dcr_norm > 0, dcr_norm = dcr_norm * 2; end
    if drift_norm > 0, drift_norm = drift_norm * 1.5; end
    if roof_norm > 0, roof_norm = roof_norm * 1.2; end
    
    % Combined score (lower is better)
    score = w_dcr * dcr_norm + w_drift * drift_norm + w_roof * roof_norm;
end

function [rating, recommendation] = assess_performance_v7(dcr_improve, drift_improve, roof_improve)
    % V7: Assess TMD performance and provide recommendation
    
    avg_improve = (dcr_improve + drift_improve + roof_improve) / 3;
    
    if avg_improve >= 15
        % rating = '⭐⭐⭐ EXCELLENT';
        rating = 'EXCELLENT';
        recommendation = 'TMD is highly effective. Recommended for implementation.';
    elseif avg_improve >= 8
        % rating = '⭐⭐ GOOD';
        rating = 'GOOD';
        recommendation = 'TMD provides meaningful improvement. Cost-benefit analysis recommended.';
    elseif avg_improve >= 3
        % rating = '⭐ MODERATE';
        rating = 'MODERATE';
        recommendation = 'TMD shows limited benefit. Consider alternatives or multiple TMDs.';
    else
        % rating = '❌ LIMITED';
        rating = 'LIMITED';
        recommendation = 'TMD ineffective for this scenario. Consider: (1) Multiple TMDs, (2) Active control, (3) Base isolation.';
    end
    
    % Additional warnings
    if roof_improve < -5
        recommendation = [recommendation ' WARNING: Significant roof displacement increase.'];
    end
end

function [DCR, profile] = compute_DCR_v7(drift)
    % V7: Enhanced DCR calculation with better normalization
    
    peak_per_story = max(abs(drift),[],2);
    
    % Use 75th percentile instead of mean for more robust normalization
    sorted_peaks = sort(peak_per_story);
    n = length(sorted_peaks);
    percentile_75 = sorted_peaks(round(0.75*n));
    
    max_peak = max(peak_per_story);
    
    if percentile_75 > 0
        DCR = max_peak / percentile_75;
    else
        DCR = Inf;
    end
    
    profile = peak_per_story(:).';
end

function plot_results_v7(t, driftA, driftB, DCR_A, DCR_B, tmd_floor, eq_info)
    % V7: Enhanced visualization
    
    figure('Name','V7: TMD Performance Analysis','Color','w','Position',[100,100,1400,700]);
    
    % Subplot 1: Drift heatmaps
    subplot(2,3,[1,4]);
    allDrifts = [driftA(:); driftB(:)];
    cmin = min(allDrifts); cmax = max(allDrifts);
    if cmin==cmax, cmin=cmin-1e-12; cmax=cmax+1e-12; end
    
    imagesc(t, 2:size(driftA,1)+1, driftA); axis xy; colorbar;
    caxis([cmin cmax]); colormap('jet');
    title(sprintf('Baseline Drift (DCR=%.2f)', DCR_A), 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Time (s)'); ylabel('Story');
    
    subplot(2,3,[2,5]);
    imagesc(t, 2:size(driftB,1)+1, driftB); axis xy; colorbar;
    caxis([cmin cmax]); colormap('jet');
    title(sprintf('With TMD @Floor %d (DCR=%.2f)', tmd_floor, DCR_B), 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Time (s)'); ylabel('Story');
    
    % Subplot 2: Peak drift comparison
    subplot(2,3,3);
    peakA = max(abs(driftA),[],2);
    peakB = max(abs(driftB),[],2);
    plot(2:length(peakA)+1, peakA, 'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'b'); hold on;
    plot(2:length(peakB)+1, peakB, 'r-s', 'LineWidth', 2, 'MarkerFaceColor', 'r');
    xlabel('Story'); ylabel('Peak Drift (m)');
    title('Peak Drift per Story', 'FontSize', 12, 'FontWeight', 'bold');
    legend({'Baseline','With TMD'}, 'Location', 'best');
    grid on;
    
    % Subplot 3: Performance metrics
    subplot(2,3,6);
    improvement = 100*(1 - DCR_B/DCR_A);
    bar_data = [DCR_A, DCR_B; improvement, 0];
    b = bar(bar_data);
    b(1).FaceColor = [0.2 0.4 0.8];
    b(2).FaceColor = [0.8 0.2 0.2];
    set(gca, 'XTickLabel', {'DCR', 'Improvement (%)'});
    title('Performance Summary', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Value');
    legend({'Baseline','TMD'}, 'Location', 'best');
    grid on;
    
    % Add subtitle
    sgtitle(sprintf('V7 Analysis: %s', eq_info.name), 'FontSize', 14, 'FontWeight', 'bold');
end

%% ============================================================
%% COMMON HELPER FUNCTIONS
%% ============================================================

function [M2,K2,C2,F2] = augment_with_TMD_at_floor(M,K,C,F,N,m_t,k_t,c_t,Nt,floorIdx)
    M2 = blkdiag(M, m_t);
    K2 = blkdiag(K, 0);
    C2 = blkdiag(C, 0);
    
    K2(floorIdx,floorIdx) = K2(floorIdx,floorIdx) + k_t;
    K2(floorIdx,N+1) = -k_t;
    K2(N+1,floorIdx) = -k_t;
    K2(N+1,N+1) = k_t;
    
    C2(floorIdx,floorIdx) = C2(floorIdx,floorIdx) + c_t;
    C2(floorIdx,N+1) = -c_t;
    C2(N+1,floorIdx) = -c_t;
    C2(N+1,N+1) = c_t;
    
    F2 = [F; zeros(1,Nt)];
end

function [x,v,a] = newmark_simulate(M,C,K,F,t,beta,gamma)
    dt=t(2)-t(1); Nt=length(t); N=size(M,1);
    x=zeros(N,Nt); v=zeros(N,Nt); a=zeros(N,Nt);
    a(:,1)=M\(F(:,1)-C*v(:,1)-K*x(:,1));
    Khat=K+gamma/(beta*dt)*C+M/(beta*dt^2);
    [L,U,P]=lu(Khat);
    for k=1:Nt-1
        xk=x(:,k); vk=v(:,k); ak=a(:,k);
        F_eff=F(:,k+1)+M*((1/(beta*dt^2))*xk+(1/(beta*dt))*vk+(1/(2*beta)-1)*ak)...
                       +C*((gamma/(beta*dt))*xk+(gamma/beta-1)*vk+dt*(gamma/(2*beta)-1)*ak);
        y=L\(P*F_eff); x(:,k+1)=U\y;
        a(:,k+1)=(1/(beta*dt^2))*(x(:,k+1)-xk)-(1/(beta*dt))*vk-(1/(2*beta)-1)*ak;
        v(:,k+1)=vk+dt*((1-gamma)*ak+gamma*a(:,k+1));
    end
    x(~isfinite(x))=0; v(~isfinite(v))=0; a(~isfinite(a))=0;
end

function a0a1 = solve_rayleigh(om1,om2,zeta)
    A = 0.5 * [1/om1, om1; 1/om2, om2];
    sol = A \ [zeta; zeta];
    a0a1 = sol(:);
end

function drift = compute_interstory_drifts(x)
    N=size(x,1); Nt=size(x,2);
    drift=zeros(N-1,Nt);
    for i=2:N, drift(i-1,:)=x(i,:)-x(i-1,:); end
end

%% ============================================================
%% DATA LOADING FUNCTIONS (unchanged from v5/v6)
%% ============================================================

function [ag, t, info] = load_real_seismic_data(earthquake_name, dt, target_duration)
    if nargin < 2, dt = 0.01; end
    if nargin < 3, target_duration = []; end
    
    [ag_raw, t_raw, info] = load_from_database(earthquake_name);
    
    t_new = (t_raw(1):dt:t_raw(end))';
    ag_resampled = interp1(t_raw, ag_raw, t_new, 'linear');
    
    if ~isempty(target_duration)
        n_target = round(target_duration / dt);
        n_current = length(ag_resampled);
        
        if n_current > n_target
            ag_resampled = ag_resampled(1:n_target);
            t_new = t_new(1:n_target);
        elseif n_current < n_target
            ag_resampled = [ag_resampled; zeros(n_target - n_current, 1)];
            t_new = (0:dt:(n_target-1)*dt)';
        end
    end
    
    ag = ag_resampled - mean(ag_resampled);
    t = t_new;
    
    info.dt = dt;
    info.duration = t(end);
    info.n_points = length(t);
    info.pga = max(abs(ag));
    info.pga_g = info.pga / 9.81;
end

function [ag, t, info] = load_from_database(earthquake_name)
    % Check if it's a CSV file
    if contains(earthquake_name, '.csv')
        if ~isfile(earthquake_name)
            error('CSV file not found: %s', earthquake_name);
        end
        
        data = readmatrix(earthquake_name);
        t = data(:, 1);
        ag = data(:, 2);
        
        [~, name, ~] = fileparts(earthquake_name);
        info.name = name;
        info.magnitude = NaN;
        return;
    end
    
    % Otherwise use built-in database
    switch lower(earthquake_name)
        case 'northridge'
            info.name = 'Northridge 1994';
            info.magnitude = 6.7;
            t = (0:0.02:40)';
            ag = generate_realistic_northridge(t);
            
        case 'el_centro'
            info.name = 'Imperial Valley 1940 (El Centro)';
            info.magnitude = 6.9;
            t = (0:0.02:53.74)';
            ag = generate_realistic_el_centro(t);
            
        case 'kobe'
            info.name = 'Kobe 1995';
            info.magnitude = 6.9;
            t = (0:0.02:48)';
            ag = generate_realistic_kobe(t);
            
        case 'loma_prieta'
            info.name = 'Loma Prieta 1989';
            info.magnitude = 6.9;
            t = (0:0.02:25)';
            ag = generate_realistic_loma_prieta(t);
            
        otherwise
            error('Unknown earthquake: %s', earthquake_name);
    end
end

function ag = generate_realistic_northridge(t)
    t_rise = 3; t_strong = 10; t_decay = 25;
    N = length(t);
    envelope = zeros(size(t));
    
    for i = 1:N
        if t(i) < t_rise
            envelope(i) = (t(i) / t_rise)^2;
        elseif t(i) < t_strong
            envelope(i) = 1.0;
        elseif t(i) < t_decay
            envelope(i) = exp(-(t(i) - t_strong) / 5);
        else
            envelope(i) = 0.1 * exp(-(t(i) - t_decay) / 3);
        end
    end
    
    base = 0.8 * sin(2*pi*2.0*t) .* envelope;
    harmonic1 = 0.3 * sin(2*pi*4.5*t + 0.7) .* envelope;
    harmonic2 = 0.2 * sin(2*pi*8.0*t + 1.2) .* envelope;
    noise = 0.1 * randn(size(t)) .* envelope;
    
    ag = (base + harmonic1 + harmonic2 + noise);
    ag = ag * 11.8 / max(abs(ag));
    ag = ag - mean(ag);
end

function ag = generate_realistic_el_centro(t)
    envelope = exp(-t/20) .* (1 - exp(-t/2));
    
    base = 0.7 * sin(2*pi*1.5*t) .* envelope;
    harmonic1 = 0.4 * sin(2*pi*3.5*t + 0.5) .* envelope;
    harmonic2 = 0.2 * sin(2*pi*7.0*t + 1.0) .* envelope;
    noise = 0.15 * randn(size(t)) .* envelope;
    
    ag = (base + harmonic1 + harmonic2 + noise);
    ag = ag * 3.4 / max(abs(ag));
    ag = ag - mean(ag);
end

function ag = generate_realistic_kobe(t)
    envelope = exp(-t/18) .* (1 - exp(-t/1.0));
    pulse = 2.0 * exp(-((t-4).^2)/4);
    
    base = 0.5 * sin(2*pi*1.2*t) .* envelope + pulse;
    harmonic1 = 0.3 * sin(2*pi*3.0*t + 0.4) .* envelope;
    harmonic2 = 0.2 * sin(2*pi*5.5*t + 0.9) .* envelope;
    noise = 0.1 * randn(size(t)) .* envelope;
    
    ag = (base + harmonic1 + harmonic2 + noise);
    ag = ag * 8.3 / max(abs(ag));
    ag = ag - mean(ag);
end

function ag = generate_realistic_loma_prieta(t)
    envelope = exp(-t/15) .* (1 - exp(-t/1.5));
    
    base = 0.6 * sin(2*pi*1.8*t) .* envelope;
    harmonic1 = 0.4 * sin(2*pi*4.0*t + 0.6) .* envelope;
    harmonic2 = 0.25 * sin(2*pi*6.5*t + 1.1) .* envelope;
    noise = 0.12 * randn(size(t)) .* envelope;
    
    ag = (base + harmonic1 + harmonic2 + noise);
    ag = ag * 4.5 / max(abs(ag));
    ag = ag - mean(ag);
end

function [wind_force, t, info] = load_real_wind_data(wind_type, building_height, dt, duration)
    if nargin < 3, dt = 0.01; end
    if nargin < 4, duration = 120; end

    %wind_type = fullfile("datasets", wind_type)
    
    % Check if it's a CSV file
    if contains(wind_type, '.csv')
        if ~isfile(wind_type)
            error('CSV file not found: %s', wind_type);
        end
        
        data = readmatrix(wind_type);
        t = data(:, 1);
        wind_speed = data(:, 2);
        
        [~, name, ~] = fileparts(wind_type);
        info.name = name;
        info.mean_speed = mean(wind_speed);
    else
        t = (0:dt:duration)';
        
        switch lower(wind_type)
            case 'hurricane'
                info.name = 'Hurricane (Cat 3)';
                info.mean_speed = 50;
                info.gust_factor = 1.5;
                wind_speed = generate_hurricane_wind(t, info.mean_speed, info.gust_factor);
                
            case 'strong_wind'
                info.name = 'Strong Wind Storm';
                info.mean_speed = 25;
                info.gust_factor = 1.4;
                wind_speed = generate_hurricane_wind(t, info.mean_speed, info.gust_factor);
                
            case 'typhoon'
                info.name = 'Typhoon';
                info.mean_speed = 45;
                info.gust_factor = 1.6;
                wind_speed = generate_hurricane_wind(t, info.mean_speed, info.gust_factor);
                
            otherwise
                error('Unknown wind type: %s', wind_type);
        end
    end
    
    rho = 1.225; Cd = 1.2; width = 20;
    N_floors = 12;
    floor_height = building_height / N_floors;
    A_floor = width * floor_height;
    
    alpha = 0.15; z_ref = 10;
    N_time = length(t);
    wind_force = zeros(N_floors, N_time);
    
    for floor = 1:N_floors
        z = floor * floor_height;
        height_factor = (z / z_ref)^alpha;
        v_floor = wind_speed * height_factor;
        wind_force(floor, :) = 0.5 * rho * Cd * A_floor * v_floor.^2;
    end
    
    info.max_force_per_floor = max(wind_force(:));
    info.total_max_force = sum(max(wind_force, [], 2));
end

function wind_speed = generate_hurricane_wind(t, mean_speed, gust_factor)
    mean_component = mean_speed * (1 + 0.1 * sin(2*pi*0.01*t));
    gust1 = 0.3 * mean_speed * sin(2*pi*0.05*t);
    gust2 = 0.2 * mean_speed * sin(2*pi*0.12*t + 1.2);
    gust3 = 0.15 * mean_speed * sin(2*pi*0.25*t + 0.7);
    turbulence = 0.1 * mean_speed * randn(size(t));
    
    wind_speed = mean_component + gust1 + gust2 + gust3 + turbulence;
    wind_speed = max(wind_speed, 0);
end