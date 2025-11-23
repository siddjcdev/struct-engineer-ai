function response = send_to_api(data, api_url, data_id)
    % Send data from MATLAB to the REST API
    
    if nargin < 3
        data_id = ['matlab_' datestr(now, 'yyyymmdd_HHMMSS')];
    end
    
    % Prepare payload
    payload = struct();
    payload.id = data_id;
    payload.payload = data;
    payload.metadata = struct('source', 'MATLAB', 'timestamp', posixtime(datetime('now')));
    
    % Convert to JSON
    json_data = jsonencode(payload);
    
    % Set up HTTP options
    options = weboptions(...
        'MediaType', 'application/json', ...
        'RequestMethod', 'post', ...
        'Timeout', 30);
    
    try
        % Send POST request
        url = [api_url '/upload'];
        response = webwrite(url, json_data, options);
        
        fprintf('✓ Data uploaded successfully!\n');
        fprintf('  ID: %s\n', response.id);
        fprintf('  Status: %s\n', response.status);
        
    catch ME
        fprintf('✗ Error uploading data:\n');
        fprintf('  %s\n', ME.message);
        response = struct('status', 'error', 'message', ME.message);
    end
end