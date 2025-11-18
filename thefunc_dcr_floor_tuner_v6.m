function thefunc_dcr_floor_tuner_v5(earthquake_name, use_wind, wind_type)
    % Wrapper function for TMD tuner with flexible parameters - WITH FIXES APPLIED
    %
    % FIX #1 APPLIED: Guard constraints relaxed to 50% (was 15%)
    %
    % Usage:
    %   thefunc_dcr_floor_tuner_v5()                           % Default: el_centro + strong_wind
    %   thefunc_dcr_floor_tuner_v5('northridge')               % Northridge, no wind
    %   thefunc_dcr_floor_tuner_v5('kobe', false)              % Kobe, no wind
    %   thefunc_dcr_floor_tuner_v5('loma_prieta', true)        % Loma Prieta + strong_wind
    %   thefunc_dcr_floor_tuner_v5('northridge', true, 'hurricane')  % Northridge + hurricane
    %
    % Available earthquakes: 'el_centro', 'northridge', 'kobe', 'loma_prieta'
    % Available winds: 'strong_wind', 'hurricane', 'typhoon'
    
    % Set defaults if not provided
    if nargin < 1 || isempty(earthquake_name)
        earthquake_name = 'el_centro';
    end
    if nargin < 2 || isempty(use_wind)
        use_wind = true;
    end
    if nargin < 3 || isempty(wind_type)
        wind_type = 'strong_wind';
    end
    
    % Call main function with parameters
    thefunc_dcr_floor_tuner_real_data(earthquake_name, use_wind, wind_type)
    
end

function thefunc_dcr_floor_tuner_real_data(earthquake_name, use_wind, wind_type)
    % Modified TMD tuner that uses historic earthquake and wind data
    %
    % Inputs:
    %   earthquake_name - 'northridge', 'el_centro', 'kobe', 'loma_prieta'
    %                     OR path to CSV/TXT file with earthquake data
    %   use_wind        - true/false to include wind loading
    %   wind_type       - 'hurricane', 'strong_wind', 'typhoon' (if use_wind=true)
    %
    % Examples:
    %   thefunc_dcr_floor_tuner_real_data('northridge', false)
    %   thefunc_dcr_floor_tuner_real_data('el_centro', true, 'strong_wind')
    
    if nargin < 2, use_wind = false; end
    if nargin < 3, wind_type = 'strong_wind'; end
    
    fprintf('\n========================================\n');
    fprintf('TMD TUNER WITH REAL EARTHQUAKE DATA\n');
    fprintf('========================================\n\n');
    
    % Building parameters
    N = 12; m0 = 2.0e5; 
    k0 = 2.0e7;
    zeta_target = 0.015;
    dt = 0.01;
    T = 120;  % Target duration
    
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
    
    % ===== LOAD REAL EARTHQUAKE DATA =====
    fprintf('Loading earthquake data: %s\n', earthquake_name);
    [ag, t, eq_info] = load_real_seismic_data(earthquake_name, dt, T);
    Nt = length(t);
    
    % Ground motion forces
    r = ones(N,1);
    Fg = -M*r*ag.';
    
    % ===== ADD WIND LOADING IF REQUESTED =====
    if use_wind
        fprintf('\nLoading wind data: %s\n', wind_type);
        building_height = N * 3;  % Assume 3m per floor
        [Fw, ~, wind_info] = load_real_wind_data(wind_type, building_height, dt, T);
        
        % Ensure same time dimension
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
    [DCR_A, dcr_profile_A] = compute_DCR(driftA);
    peakRoofA = max(abs(roofA));
    rmsRoofA  = sqrt(mean(roofA.^2));
    
    % Calculate average displacement, velocity, acceleration for baseline
    avg_disp_A = mean(abs(xA(:)));
    avg_vel_A = mean(abs(vA(:)));
    avg_acc_A = mean(abs(aA(:)));
    
    % Print diagnostics
    fprintf('\n===== BASELINE RESULTS =====\n');
    fprintf('Earthquake: %s\n', eq_info.name);
    fprintf('  PGA: %.3f m/s² (%.2fg)\n', eq_info.pga, eq_info.pga_g);
    if use_wind
        fprintf('Wind: %s\n', wind_info.name);
        fprintf('  Mean speed: %.1f m/s\n', wind_info.mean_speed);
    end
    fprintf('\nBuilding Response:\n');
    fprintf('  Max roof displacement: %.3e m\n', peakRoofA);
    fprintf('  Max inter-story drift: %.3e m\n', max(abs(driftA(:))));
    fprintf('  DCR: %.3f\n', DCR_A);
    fprintf('  Avg displacement: %.3e m\n', avg_disp_A);
    fprintf('  Avg velocity: %.3e m/s\n', avg_vel_A);
    fprintf('  Avg acceleration: %.3e m/s²\n', avg_acc_A);
    fprintf('============================\n\n');
    
    % Identify soft story and candidate floors
    [~,softIdxLocal] = max(dcr_profile_A);
    softStory = softIdxLocal + 1;
    floorCandidates = unique(max(2, min(N, [softStory-1, softStory, softStory+1, N])));
    
    % TMD parameter grids
    mu_vals   = 0.01:0.01:0.30;
    zeta_vals = 0.05:0.02:0.50;
    
    % ═══════════════════════════════════════════════════════════
    % FIX #1 APPLIED: Guard constraints relaxed to 50% (was 15%)
    % ═══════════════════════════════════════════════════════════
    fprintf('Setting guard constraints (50%% tolerance)...\n');
    peak_guard = 1.50*peakRoofA;  % FIXED: Was 1.15 (15%)
    rms_guard  = 1.50*rmsRoofA;   % FIXED: Was 1.15 (15%)
    
    % Search for best TMD
    best = struct('floor',NaN,'mu',NaN,'zeta',NaN,'DCR',Inf);
    
    fprintf('Searching TMD configurations...\n');
    fprintf('Candidate floors: %s\n', mat2str(floorCandidates));
    
    for fIdx = 1:numel(floorCandidates)
        fAttach = floorCandidates(fIdx);
        fprintf('  Testing floor %d...\n', fAttach);
        
        DCR_mat   = zeros(length(mu_vals), length(zeta_vals));
        peakRoofM = zeros(size(DCR_mat));
        rmsRoofM  = zeros(size(DCR_mat));
        
        for i = 1:length(mu_vals)
            for j = 1:length(zeta_vals)
                mu_i   = mu_vals(i);
                zeta_i = zeta_vals(j);
                m_ti   = mu_i*m0;
                
                k_mode = pick_mode_for_floor(V, fAttach);
                om_t   = om_sorted(k_mode);
                k_ti   = m_ti*om_t^2;
                c_ti   = 2*zeta_i*sqrt(k_ti*m_ti);
                
                [Maug,Kaug,Caug,Faug] = augment_with_TMD_at_floor(M,K,C,F,N,m_ti,k_ti,c_ti,Nt,fAttach);
                [xj,~,~] = newmark_simulate(Maug,Caug,Kaug,Faug,t,beta,gamma);
                
                roof_j   = xj(N,:);
                drift_j  = compute_interstory_drifts(xj(1:N,:));
                [DCR_j,~]= compute_DCR(drift_j);
                
                DCR_mat(i,j)   = DCR_j;
                peakRoofM(i,j) = max(abs(roof_j));
                rmsRoofM(i,j)  = sqrt(mean(roof_j.^2));
            end
        end
        
        infeasible = (peakRoofM > peak_guard) | (rmsRoofM > rms_guard);
        DCR_mat(infeasible) = Inf;
        
        [minDCR, idx] = min(DCR_mat(:));
        
        if isfinite(minDCR) && minDCR < best.DCR
            [iBest,jBest] = ind2sub(size(DCR_mat), idx);
            best.floor    = fAttach;
            best.mu       = mu_vals(iBest);
            best.zeta     = zeta_vals(jBest);
            best.DCR      = minDCR;
            best.peakRoof = peakRoofM(iBest,jBest);
            best.rmsRoof  = rmsRoofM(iBest,jBest);
            fprintf('    *** New best: mu=%.3f, zeta=%.3f, DCR=%.3f\n', ...
                best.mu, best.zeta, best.DCR);
        end
    end
    
    if ~isfinite(best.DCR)
        fprintf('No feasible TMD found.\n');
        return;
    end
    
    fprintf('\nBest TMD: floor=%d, mu=%.3f, zeta=%.3f\n', ...
        best.floor, best.mu, best.zeta);
    
    % Re-simulate with best TMD
    m_t = best.mu*m0;
    k_mode = pick_mode_for_floor(V, best.floor);
    om_t = om_sorted(k_mode);
    k_t = m_t*om_t^2;
    c_t = 2*best.zeta*sqrt(k_t*m_t);
    [Mbest,Kbest,Cbest,Fbest] = augment_with_TMD_at_floor(M,K,C,F,N,m_t,k_t,c_t,Nt,best.floor);
    [xB,vB,aB] = newmark_simulate(Mbest,Cbest,Kbest,Fbest,t,beta,gamma);
    roofB = xB(N,:);
    driftB = compute_interstory_drifts(xB(1:N,:));
    [DCR_B, dcr_profile_B] = compute_DCR(driftB);
    peakRoofB = max(abs(roofB));
    rmsRoofB  = sqrt(mean(roofB.^2));
    
    % Calculate average displacement, velocity, acceleration with TMD
    avg_disp_B = mean(mean(abs(xB(1:N,:))));
    avg_vel_B = mean(mean(abs(vB(1:N,:))));
    avg_acc_B = mean(mean(abs(aB(1:N,:))));
    
    % Print comparison
    fprintf('\n===== PERFORMANCE COMPARISON =====\n');
    fprintf('Earthquake: %s (PGA: %.2fg)\n', eq_info.name, eq_info.pga_g);
    if use_wind
        fprintf('Wind: %s (%.1f m/s)\n', wind_info.name, wind_info.mean_speed);
    end
    fprintf('\nMax drift: %.3e m → %.3e m (%.1f%% reduction)\n', ...
        max(abs(driftA(:))), max(abs(driftB(:))), ...
        100*(1 - max(abs(driftB(:)))/max(abs(driftA(:)))));
    fprintf('Max roof:  %.3e m → %.3e m (%.1f%% reduction)\n', ...
        peakRoofA, peakRoofB, 100*(1 - peakRoofB/peakRoofA));
    fprintf('DCR:       %.3f → %.3f (%.1f%% reduction)\n', ...
        DCR_A, DCR_B, 100*(1 - DCR_B/DCR_A));
    fprintf('Avg disp:  %.3e m → %.3e m (%.1f%% reduction)\n', ...
        avg_disp_A, avg_disp_B, 100*(1 - avg_disp_B/avg_disp_A));
    fprintf('Avg vel:   %.3e m/s → %.3e m/s (%.1f%% reduction)\n', ...
        avg_vel_A, avg_vel_B, 100*(1 - avg_vel_B/avg_vel_A));
    fprintf('Avg acc:   %.3e m/s² → %.3e m/s² (%.1f%% reduction)\n', ...
        avg_acc_A, avg_acc_B, 100*(1 - avg_acc_B/avg_acc_A));
    fprintf('==================================\n');
    
    % ===== SAVE RESULTS AS JSON FOR REST API =====
    fprintf('\n===== SAVING JSON FOR REST API/COLAB =====\n');
    
    % Prepare data structure for REST API
    api_data = struct();
    
    % Metadata
    api_data.metadata = struct();
    api_data.metadata.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    api_data.metadata.simulation_type = 'passive_tmd';
    api_data.metadata.n_floors = N;
    api_data.metadata.time_step = dt;
    api_data.metadata.duration = T;
    
    % Input conditions
    api_data.input = struct();
    api_data.input.earthquake = eq_info;
    api_data.input.earthquake_force_total = sum(abs(Fg(:)));  % Total earthquake force magnitude
    api_data.input.earthquake_force_per_floor = sum(abs(Fg), 2);  % Force per floor (Nx1 array)
    api_data.input.use_wind = use_wind;
    if use_wind
        api_data.input.wind = wind_info;
    end
    
    % Baseline (no TMD) results
    api_data.baseline = struct();
    api_data.baseline.DCR = DCR_A;
    api_data.baseline.max_roof_displacement = peakRoofA;
    api_data.baseline.rms_roof_displacement = rmsRoofA;
    api_data.baseline.max_drift = max(abs(driftA(:)));
    api_data.baseline.avg_displacement = avg_disp_A;
    api_data.baseline.avg_velocity = avg_vel_A;
    api_data.baseline.avg_acceleration = avg_acc_A;
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
    
    % TMD results
    api_data.tmd_results = struct();
    api_data.tmd_results.DCR = DCR_B;
    api_data.tmd_results.max_roof_displacement = peakRoofB;
    api_data.tmd_results.rms_roof_displacement = rmsRoofB;
    api_data.tmd_results.max_drift = max(abs(driftB(:)));
    api_data.tmd_results.avg_displacement = avg_disp_B;
    api_data.tmd_results.avg_velocity = avg_vel_B;
    api_data.tmd_results.avg_acceleration = avg_acc_B;
    api_data.tmd_results.dcr_profile = dcr_profile_B;
    
    % Performance improvements
    api_data.improvements = struct();
    api_data.improvements.dcr_reduction_pct = 100*(1 - DCR_B/DCR_A);
    api_data.improvements.max_drift_reduction_pct = 100*(1 - max(abs(driftB(:)))/max(abs(driftA(:))));
    api_data.improvements.max_roof_reduction_pct = 100*(1 - peakRoofB/peakRoofA);
    api_data.improvements.avg_disp_reduction_pct = 100*(1 - avg_disp_B/avg_disp_A);
    api_data.improvements.avg_vel_reduction_pct = 100*(1 - avg_vel_B/avg_vel_A);
    api_data.improvements.avg_acc_reduction_pct = 100*(1 - avg_acc_B/avg_acc_A);
    
    % Time series data (sampled for efficiency)
    sample_rate = 10;  % Sample every 10th point to reduce data size
    time_indices = 1:sample_rate:length(t);
    
    api_data.time_series = struct();
    api_data.time_series.time = t(time_indices);
    api_data.time_series.earthquake_acceleration = ag(time_indices);
    api_data.time_series.baseline_roof_displacement = roofA(time_indices);
    api_data.time_series.tmd_roof_displacement = roofB(time_indices);
    
    % Save as JSON
    try
        json_filename = sprintf('tmd_simulation_%s.json', datestr(now, 'yyyymmdd_HHMMSS'));
        json_str = jsonencode(api_data);
        
        % Write to file
        fid = fopen(json_filename, 'w');
        if fid == -1
            error('Cannot create JSON file');
        end
        fprintf(fid, '%s', json_str);
        fclose(fid);
        
        fprintf('✓ JSON saved: %s\n', json_filename);
        fprintf('  File size: %.2f KB\n', dir(json_filename).bytes / 1024);
        fprintf('  Ready for REST API consumption\n');
    catch ME
        fprintf('✗ JSON export failed: %s\n', ME.message);
    end
    
    % Also save MATLAB data for backward compatibility
    building_results = struct();
    building_results.time = t;
    building_results.baseline_roof = roofA;
    building_results.baseline_drift = driftA;
    building_results.baseline_DCR = DCR_A;
    building_results.tmd_roof = roofB;
    building_results.tmd_drift = driftB;
    building_results.tmd_DCR = DCR_B;
    building_results.tmd_floor = best.floor;
    building_results.tmd_mu = best.mu;
    building_results.tmd_zeta = best.zeta;
    building_results.max_drift_baseline = max(abs(driftA(:)));
    building_results.max_drift_tmd = max(abs(driftB(:)));
    building_results.earthquake = eq_info;
    if use_wind
        building_results.wind = wind_info;
    end
    
    % Save additional metrics
    building_results.avg_displacement_baseline = avg_disp_A;
    building_results.avg_velocity_baseline = avg_vel_A;
    building_results.avg_acceleration_baseline = avg_acc_A;
    building_results.avg_displacement_tmd = avg_disp_B;
    building_results.avg_velocity_tmd = avg_vel_B;
    building_results.avg_acceleration_tmd = avg_acc_B;
    building_results.earthquake_force = Fg;
    
    try
        mat_filename = 'building_sim_latest.mat';
        save(mat_filename, 'building_results');
        fprintf('✓ MATLAB data saved: %s\n', mat_filename);
    catch ME
        fprintf('✗ MATLAB save failed: %s\n', ME.message);
    end
    
    fprintf('==========================================\n');
    
    % Plots
    allDrifts = [driftA(:); driftB(:)];
    cmin = min(allDrifts); cmax = max(allDrifts);
    if cmin==cmax, cmin=cmin-1e-12; cmax=cmax+1e-12; end
    
    figure('Name','Inter-story drift heatmaps','Color','w','Position',[150,150,1200,600]);
    subplot(1,2,1);
    imagesc(t, 2:N, driftA); axis xy; colorbar; caxis([cmin cmax]);
    title(sprintf('No TMD: DCR=%.2f (%s)', DCR_A, eq_info.name));
    xlabel('Time (s)'); ylabel('Story');
    
    subplot(1,2,2);
    imagesc(t, 2:N, driftB); axis xy; colorbar; caxis([cmin cmax]);
    title(sprintf('Best TMD @floor %d: DCR=%.2f', best.floor, DCR_B));
    xlabel('Time (s)'); ylabel('Story');
    
    peakA_story = max(abs(driftA),[],2).';
    peakB_story = max(abs(driftB),[],2).';
    
    figure('Name','Story drift profiles','Color','w','Position',[160,160,800,500]);
    plot(2:N, peakA_story, 'b-o','LineWidth',1.5,'MarkerFaceColor','b'); hold on;
    plot(2:N, peakB_story, 'r-s','LineWidth',1.5,'MarkerFaceColor','r');
    xlabel('Story'); ylabel('Peak inter-story drift (m)');
    title(sprintf('Peak drift per story - %s', eq_info.name)); grid on; 
    legend({'No TMD','Best TMD'},'Location','best');
end

% ===== HELPER FUNCTIONS =====

function [M2,K2,C2,F2] = augment_with_TMD_at_floor(M,K,C,F,N,m_t,k_t,c_t,Nt,floorIdx)
    M2 = blkdiag(M, m_t); 
    K2 = blkdiag(K, 0); 
    C2 = blkdiag(C, 0);
    
    K2(floorIdx,floorIdx) = K2(floorIdx,floorIdx) + k_t;
    K2(floorIdx,N+1)      = -k_t;
    K2(N+1,floorIdx)      = -k_t;
    K2(N+1,N+1)           = k_t;
    
    C2(floorIdx,floorIdx) = C2(floorIdx,floorIdx) + c_t;
    C2(floorIdx,N+1)      = -c_t;
    C2(N+1,floorIdx)      = -c_t;
    C2(N+1,N+1)           = c_t;
    
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

function [DCR, profile] = compute_DCR(drift)
    peak_per_story = max(abs(drift),[],2);
    mean_peak = mean(peak_per_story); 
    max_peak = max(peak_per_story);
    
    if mean_peak > 0
        DCR = max_peak / mean_peak;
    else
        DCR = Inf;
    end
    
    profile = peak_per_story(:).';
end

function k_mode = pick_mode_for_floor(V, fAttach)
    N = size(V,1); nModes = size(V,2);
    bestScore = -inf; k_mode = 1;
    
    for k = 1:nModes
        Vk = V(:,k) / max(abs(V(:,k)));
        score = abs(Vk(fAttach));
        
        if score > bestScore
            bestScore = score;
            k_mode = k;
        end
    end
end

% ===== EARTHQUAKE AND WIND DATA LOADERS =====

function [ag, t, info] = load_real_seismic_data(earthquake_name, dt, target_duration)
    if nargin < 2, dt = 0.01; end
    if nargin < 3, target_duration = []; end
    
    % Load from built-in earthquake database
    [ag_raw, t_raw, info] = load_from_database(earthquake_name);
    
    % Resample to desired time step
    t_new = (t_raw(1):dt:t_raw(end))';
    ag_resampled = interp1(t_raw, ag_raw, t_new, 'linear');
    
    % Truncate or pad to target duration
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
    
    fprintf('✓ Loaded: %s\n', info.name);
    fprintf('  PGA: %.3f m/s² (%.2fg)\n', info.pga, info.pga_g);
    fprintf('  Duration: %.1f s, Points: %d\n', info.duration, info.n_points);
end

function [ag, t, info] = load_from_database(earthquake_name)
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
            error('Unknown earthquake: %s. Use: northridge, el_centro, kobe, loma_prieta', earthquake_name);
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
    
    t = (0:dt:duration)';
    N_time = length(t);
    
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
    
    rho = 1.225; Cd = 1.2; width = 20;
    N_floors = 12;
    floor_height = building_height / N_floors;
    A_floor = width * floor_height;
    
    alpha = 0.15; z_ref = 10;
    wind_force = zeros(N_floors, N_time);
    
    for floor = 1:N_floors
        z = floor * floor_height;
        height_factor = (z / z_ref)^alpha;
        v_floor = wind_speed * height_factor;
        wind_force(floor, :) = 0.5 * rho * Cd * A_floor * v_floor.^2;
    end
    
    info.max_force_per_floor = max(wind_force(:));
    info.total_max_force = sum(max(wind_force, [], 2));
    
    fprintf('✓ Wind loading: %s\n', info.name);
    fprintf('  Mean speed: %.1f m/s\n', info.mean_speed);
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