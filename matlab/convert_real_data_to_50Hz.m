% ============================================================
% CONVERT REAL DOWNLOADED DATA TO 0.02s INTERVALS
% ============================================================
% This script converts real PEER .AT2 and NREL CSV files
% to the proper format at dt = 0.02 seconds (50 Hz)
%
% Use this if you downloaded real data from:
%   - PEER NGA-West2 (.AT2 files)
%   - NREL Wind Toolkit (CSV files)
%   - Any other source
% ============================================================

function convert_real_data_to_50Hz()
    
    fprintf('\n╔════════════════════════════════════════════════════════╗\n');
    fprintf('║  REAL DATA CONVERTER TO 50 Hz (dt=0.02s)              ║\n');
    fprintf('╚════════════════════════════════════════════════════════╝\n\n');
    
    fprintf('What type of data do you want to convert?\n');
    fprintf('  1. PEER Earthquake (.AT2 file)\n');
    fprintf('  2. NREL Wind (CSV file)\n');
    fprintf('  3. Generic CSV (time, value)\n');
    fprintf('  4. Batch convert multiple files\n\n');
    
    choice = input('Enter choice (1-4): ');
    
    switch choice
        case 1
            convert_peer_earthquake();
        case 2
            convert_nrel_wind();
        case 3
            convert_generic_csv();
        case 4
            batch_convert();
        otherwise
            fprintf('Invalid choice.\n');
    end
end

%% ============================================================
%% CONVERT PEER EARTHQUAKE .AT2 FILE
%% ============================================================
function convert_peer_earthquake()
    fprintf('\n=== Convert PEER .AT2 Earthquake ===\n');
    
    % Get input file
    [file, path] = uigetfile('*.AT2;*.at2;*.txt', 'Select PEER .AT2 file');
    if isequal(file,0)
        fprintf('Cancelled.\n');
        return;
    end
    
    input_file = fullfile(path, file);
    fprintf('Input file: %s\n', input_file);
    
    % Parse .AT2 file
    fprintf('Parsing .AT2 file...\n');
    [ag_raw, t_raw, npts, dt_orig, ~] = parse_at2_file(input_file);
    
    fprintf('Original data:\n');
    fprintf('  Points: %d\n', npts);
    fprintf('  dt: %.4f s\n', dt_orig);
    fprintf('  Duration: %.2f s\n', t_raw(end));
    fprintf('  PGA: %.3f m/s² (%.2fg)\n', max(abs(ag_raw)), max(abs(ag_raw))/9.81);
    
    % Target parameters
    dt_target = 0.02;  % 50 Hz
    duration_target = 120;  % 120 seconds
    
    % Resample
    fprintf('\nResampling to dt=%.3f s (50 Hz)...\n', dt_target);
    [ag_new, t_new] = resample_signal(ag_raw, t_raw, dt_target, duration_target);
    
    fprintf('Resampled data:\n');
    fprintf('  Points: %d\n', length(t_new));
    fprintf('  dt: %.4f s\n', dt_target);
    fprintf('  Duration: %.2f s\n', t_new(end));
    fprintf('  PGA: %.3f m/s² (%.2fg)\n', max(abs(ag_new)), max(abs(ag_new))/9.81);
    
    % Save
    [~, name, ~] = fileparts(file);
    output_file = [name '_50Hz.csv'];
    writematrix([t_new, ag_new], output_file);
    
    fprintf('\n✓ Saved: %s\n', output_file);
    fprintf('  Ready to use in simulations!\n\n');
end

%% ============================================================
%% CONVERT NREL WIND CSV
%% ============================================================
function convert_nrel_wind()
    fprintf('\n=== Convert NREL Wind CSV ===\n');
    
    % Get input file
    [file, path] = uigetfile('*.csv;*.CSV', 'Select NREL wind CSV file');
    if isequal(file,0)
        fprintf('Cancelled.\n');
        return;
    end
    
    input_file = fullfile(path, file);
    fprintf('Input file: %s\n', input_file);
    
    % Read CSV
    fprintf('Reading CSV file...\n');
    try
        data = readmatrix(input_file);
    catch
        % Try with readtable if readmatrix fails
        T = readtable(input_file);
        data = table2array(T);
    end
    
    if size(data, 2) < 2
        error('CSV must have at least 2 columns: time, wind_speed');
    end
    
    t_raw = data(:, 1);
    wind_raw = data(:, 2);
    
    fprintf('Original data:\n');
    fprintf('  Points: %d\n', length(t_raw));
    fprintf('  Duration: %.2f s\n', t_raw(end));
    fprintf('  Mean wind: %.2f m/s\n', mean(wind_raw));
    fprintf('  Max wind: %.2f m/s\n', max(wind_raw));
    
    % Target parameters
    dt_target = 0.02;
    duration_target = 120;
    
    % Resample
    fprintf('\nResampling to dt=%.3f s (50 Hz)...\n', dt_target);
    [wind_new, t_new] = resample_signal(wind_raw, t_raw, dt_target, duration_target);
    
    % Ensure non-negative
    wind_new = max(wind_new, 0);
    
    fprintf('Resampled data:\n');
    fprintf('  Points: %d\n', length(t_new));
    fprintf('  dt: %.4f s\n', dt_target);
    fprintf('  Duration: %.2f s\n', t_new(end));
    fprintf('  Mean wind: %.2f m/s\n', mean(wind_new));
    fprintf('  Max wind: %.2f m/s\n', max(wind_new));
    
    % Save
    [~, name, ~] = fileparts(file);
    output_file = [name '_50Hz.csv'];
    writematrix([t_new, wind_new], output_file);
    
    fprintf('\n✓ Saved: %s\n', output_file);
    fprintf('  Ready to use in simulations!\n\n');
end

%% ============================================================
%% CONVERT GENERIC CSV
%% ============================================================
function convert_generic_csv()
    fprintf('\n=== Convert Generic CSV ===\n');
    
    % Get input file
    [file, path] = uigetfile('*.csv;*.CSV;*.txt', 'Select CSV file (time, value)');
    if isequal(file,0)
        fprintf('Cancelled.\n');
        return;
    end
    
    input_file = fullfile(path, file);
    fprintf('Input file: %s\n', input_file);
    
    % Read data
    fprintf('Reading file...\n');
    try
        data = readmatrix(input_file);
    catch
        T = readtable(input_file);
        data = table2array(T);
    end
    
    if size(data, 2) < 2
        error('File must have at least 2 columns: time, value');
    end
    
    t_raw = data(:, 1);
    value_raw = data(:, 2);
    
    fprintf('Original data: %d points, duration %.2f s\n', length(t_raw), t_raw(end));
    
    % Target parameters
    dt_target = 0.02;
    duration_target = 120;
    
    % Resample
    [value_new, t_new] = resample_signal(value_raw, t_raw, dt_target, duration_target);
    
    fprintf('Resampled data: %d points, dt=%.4f s\n', length(t_new), dt_target);
    
    % Save
    [~, name, ~] = fileparts(file);
    output_file = [name '_50Hz.csv'];
    writematrix([t_new, value_new], output_file);
    
    fprintf('✓ Saved: %s\n\n', output_file);
end

%% ============================================================
%% BATCH CONVERT MULTIPLE FILES
%% ============================================================
function batch_convert()
    fprintf('\n=== Batch Convert Multiple Files ===\n');
    fprintf('Select all files to convert...\n');
    
    [files, path] = uigetfile({'*.AT2;*.at2;*.csv;*.CSV;*.txt', 'All Data Files'}, ...
        'Select files to convert', 'MultiSelect', 'on');
    
    if isequal(files,0)
        fprintf('Cancelled.\n');
        return;
    end
    
    % Make sure files is a cell array
    if ~iscell(files)
        files = {files};
    end
    
    fprintf('Converting %d files...\n\n', length(files));
    
    for i = 1:length(files)
        input_file = fullfile(path, files{i});
        fprintf('[%d/%d] Converting: %s\n', i, length(files), files{i});
        
        try
            [~, ~, ext] = fileparts(files{i});
            
            if strcmpi(ext, '.at2') || strcmpi(ext, '.txt')
                % PEER earthquake
                [ag_raw, t_raw, ~, ~, ~] = parse_at2_file(input_file);
                [ag_new, t_new] = resample_signal(ag_raw, t_raw, 0.02, 120);
                
                [~, name, ~] = fileparts(files{i});
                output_file = fullfile(path, [name '_50Hz.csv']);
                writematrix([t_new, ag_new], output_file);
                
            elseif strcmpi(ext, '.csv')
                % CSV file
                data = readmatrix(input_file);
                t_raw = data(:, 1);
                value_raw = data(:, 2);
                [value_new, t_new] = resample_signal(value_raw, t_raw, 0.02, 120);
                
                [~, name, ~] = fileparts(files{i});
                output_file = fullfile(path, [name '_50Hz.csv']);
                writematrix([t_new, value_new], output_file);
            end
            
            fprintf('  ✓ Saved: %s\n', output_file);
            
        catch ME
            fprintf('  ✗ Error: %s\n', ME.message);
        end
        
        fprintf('\n');
    end
    
    fprintf('Batch conversion complete!\n\n');
end

%% ============================================================
%% HELPER FUNCTIONS
%% ============================================================

function [ag, t, npts, dt, info] = parse_at2_file(filename)
    % Parse PEER NGA .AT2 file
    
    fid = fopen(filename, 'r');
    if fid == -1
        error('Cannot open file: %s', filename);
    end
    
    % Read header
    line1 = fgetl(fid);
    line2 = fgetl(fid);
    line3 = fgetl(fid);
    line4 = fgetl(fid);
    
    info.name = strtrim(line2);
    
    % Parse NPTS and DT
    tokens = regexp(line4, 'NPTS[=\s]+(\d+)[,\s]+DT[=\s]+([\d.]+)', 'tokens', 'ignorecase');
    
    if isempty(tokens)
        fclose(fid);
        error('Cannot parse NPTS and DT from: %s', line4);
    end
    
    npts = str2double(tokens{1}{1});
    dt = str2double(tokens{1}{2});
    
    % Read acceleration data
    ag_raw = fscanf(fid, '%f');
    fclose(fid);
    
    ag = ag_raw(1:min(npts, length(ag_raw)));
    
    % Check units and convert if needed
    if max(abs(ag)) < 20
        ag = ag * 9.81;  % Convert g to m/s²
    end
    
    % Create time vector
    t = (0:dt:(length(ag)-1)*dt)';
    
    % Remove mean
    ag = ag - mean(ag);
end

function [signal_new, t_new] = resample_signal(signal_raw, t_raw, dt_target, duration_target)
    % Resample signal to target dt and duration
    
    % Create new time vector
    t_new = (t_raw(1):dt_target:t_raw(end))';
    
    % Interpolate
    signal_resampled = interp1(t_raw, signal_raw, t_new, 'linear', 'extrap');
    
    % Pad or truncate to target duration
    n_target = round(duration_target / dt_target);
    n_current = length(signal_resampled);
    
    if n_current > n_target
        signal_new = signal_resampled(1:n_target);
        t_new = t_new(1:n_target);
    elseif n_current < n_target
        signal_new = [signal_resampled; zeros(n_target - n_current, 1)];
        t_new = (0:dt_target:(n_target-1)*dt_target)';
    else
        signal_new = signal_resampled;
    end
    
    % Remove mean (for earthquakes)
    signal_new = signal_new - mean(signal_new);
end

%% ============================================================
%% END OF CONVERTER
%% ============================================================