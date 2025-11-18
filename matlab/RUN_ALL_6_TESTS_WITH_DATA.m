% ============================================================
% RUN ALL 6 TEST CASES WITH PREPARED DATASETS
% ============================================================
% This script runs all 6 test cases using the prepared CSV files
% at dt = 0.02 seconds (50 Hz sampling)
%
% Prerequisites: Run create_all_6_test_datasets.m first!
% ============================================================

function RUN_ALL_6_TESTS_WITH_DATA()
    
    fprintf('\n╔════════════════════════════════════════════════════════╗\n');
    fprintf('║  RUNNING ALL 6 TEST CASES WITH PREPARED DATA          ║\n');
    fprintf('╚════════════════════════════════════════════════════════╝\n\n');
    
    % Check if datasets exist
    if ~check_datasets_exist()
        fprintf('❌ Dataset files not found!\n');
        fprintf('Run create_all_6_test_datasets.m first to generate data.\n\n');
        response = input('Generate datasets now? (y/n): ', 's');
        if strcmpi(response, 'y')
            create_all_6_test_datasets();
        else
            return;
        end
    end
    
    fprintf('✓ All dataset files found!\n\n');
    
    % Ask user what to run
    fprintf('Options:\n');
    fprintf('  1. Run all 6 test cases (comprehensive, ~15 minutes)\n');
    fprintf('  2. Run specific test case\n');
    fprintf('  3. Quick demo (one example from each, ~6 minutes)\n\n');
    
    choice = input('Enter choice (1, 2, or 3): ');
    
    switch choice
        case 1
            run_all_comprehensive();
        case 2
            run_specific_test();
        case 3
            run_quick_demo();
        otherwise
            fprintf('Invalid choice. Running quick demo...\n');
            run_quick_demo();
    end
    
    fprintf('\n✓ All simulations complete!\n');
    fprintf('Check JSON files in current directory.\n\n');
end

%% ============================================================
%% QUICK DEMO
%% ============================================================
function run_quick_demo()
    fprintf('\n═══ QUICK DEMO: One Test from Each Case ═══\n\n');
    
    % Test 1: Stationary Wind
    fprintf('[1/6] Test Case 1: Stationary Wind\n');
    fprintf('      File: TEST1_stationary_wind_12ms.csv\n');
    thefunc_dcr_floor_tuner_real_data('el_centro', true, 'TEST1_stationary_wind_12ms.csv');
    fprintf('✓ Test 1 complete\n\n');
    pause(1);
    
    % Test 2: Turbulent Wind
    fprintf('[2/6] Test Case 2: Turbulent Wind\n');
    fprintf('      File: TEST2_turbulent_wind_25ms.csv\n');
    thefunc_dcr_floor_tuner_real_data('el_centro', true, 'TEST2_turbulent_wind_25ms.csv');
    fprintf('✓ Test 2 complete\n\n');
    pause(1);
    
    % Test 3: Small Earthquake
    fprintf('[3/6] Test Case 3: Small Earthquake (M 4.5)\n');
    fprintf('      File: TEST3_small_earthquake_M4.5.csv\n');
    thefunc_dcr_floor_tuner_real_data('TEST3_small_earthquake_M4.5.csv', false);
    fprintf('✓ Test 3 complete\n\n');
    pause(1);
    
    % Test 4: Large Earthquake
    fprintf('[4/6] Test Case 4: Large Earthquake (M 6.9)\n');
    fprintf('      File: TEST4_large_earthquake_M6.9.csv\n');
    thefunc_dcr_floor_tuner_real_data('TEST4_large_earthquake_M6.9.csv', false);
    fprintf('✓ Test 4 complete\n\n');
    pause(1);
    
    % Test 5: Mixed Input
    fprintf('[5/6] Test Case 5: Mixed Seismic-Wind Input\n');
    fprintf('      Files: TEST5_earthquake_M6.7.csv + TEST5_hurricane_wind_50ms.csv\n');
    thefunc_dcr_floor_tuner_real_data('TEST5_earthquake_M6.7.csv', true, 'TEST5_hurricane_wind_50ms.csv');
    fprintf('✓ Test 5 complete\n\n');
    pause(1);
    
    % Test 6: Stress Test
    fprintf('[6/6] Test Case 6: Stress/Noise Test\n');
    fprintf('      File: TEST6b_with_10pct_noise.csv (10%% white noise)\n');
    thefunc_dcr_floor_tuner_real_data('TEST6b_with_10pct_noise.csv', false);
    fprintf('✓ Test 6 complete\n\n');
    
    fprintf('\n═══ QUICK DEMO COMPLETE ═══\n');
    fprintf('6 simulations run, 6 JSON files created\n\n');
end

%% ============================================================
%% COMPREHENSIVE - ALL TESTS
%% ============================================================
function run_all_comprehensive()
    fprintf('\n═══ COMPREHENSIVE: All Test Cases ═══\n\n');
    
    test_count = 0;
    
    %% TEST CASE 1: STATIONARY WIND
    fprintf('╔═══════════════════════════════════════════════════════╗\n');
    fprintf('║  TEST CASE 1: STATIONARY WIND                         ║\n');
    fprintf('╚═══════════════════════════════════════════════════════╝\n');
    
    fprintf('[%d] Stationary wind + El Centro\n', test_count+1);
    thefunc_dcr_floor_tuner_real_data('el_centro', true, 'TEST1_stationary_wind_12ms.csv');
    test_count = test_count + 1;
    fprintf('✓ Complete\n\n');
    
    %% TEST CASE 2: TURBULENT WIND
    fprintf('╔═══════════════════════════════════════════════════════╗\n');
    fprintf('║  TEST CASE 2: TURBULENT/UNSTATIONARY WIND             ║\n');
    fprintf('╚═══════════════════════════════════════════════════════╝\n');
    
    fprintf('[%d] Turbulent wind + El Centro\n', test_count+1);
    thefunc_dcr_floor_tuner_real_data('el_centro', true, 'TEST2_turbulent_wind_25ms.csv');
    test_count = test_count + 1;
    fprintf('✓ Complete\n\n');
    
    %% TEST CASE 3: SMALL EARTHQUAKE
    fprintf('╔═══════════════════════════════════════════════════════╗\n');
    fprintf('║  TEST CASE 3: SMALL EARTHQUAKE (M < 5.0)              ║\n');
    fprintf('╚═══════════════════════════════════════════════════════╝\n');
    
    fprintf('[%d] Small earthquake M 4.5 (no wind)\n', test_count+1);
    thefunc_dcr_floor_tuner_real_data('TEST3_small_earthquake_M4.5.csv', false);
    test_count = test_count + 1;
    fprintf('✓ Complete\n\n');
    
    %% TEST CASE 4: LARGE EARTHQUAKE
    fprintf('╔═══════════════════════════════════════════════════════╗\n');
    fprintf('║  TEST CASE 4: LARGE EARTHQUAKE (M > 6.5)              ║\n');
    fprintf('╚═══════════════════════════════════════════════════════╝\n');
    
    fprintf('[%d] Large earthquake M 6.9 (no wind)\n', test_count+1);
    thefunc_dcr_floor_tuner_real_data('TEST4_large_earthquake_M6.9.csv', false);
    test_count = test_count + 1;
    fprintf('✓ Complete\n\n');
    
    %% TEST CASE 5: MIXED SEISMIC-WIND
    fprintf('╔═══════════════════════════════════════════════════════╗\n');
    fprintf('║  TEST CASE 5: MIXED SEISMIC-WIND INPUT                ║\n');
    fprintf('╚═══════════════════════════════════════════════════════╝\n');
    
    fprintf('[%d] Earthquake M 6.7 + Hurricane wind\n', test_count+1);
    thefunc_dcr_floor_tuner_real_data('TEST5_earthquake_M6.7.csv', true, 'TEST5_hurricane_wind_50ms.csv');
    test_count = test_count + 1;
    
    fprintf('[%d] Earthquake M 6.7 + Turbulent wind\n', test_count+1);
    thefunc_dcr_floor_tuner_real_data('TEST5_earthquake_M6.7.csv', true, 'TEST2_turbulent_wind_25ms.csv');
    test_count = test_count + 1;
    fprintf('✓ Complete\n\n');
    
    %% TEST CASE 6: STRESS/NOISE/LATENCY
    fprintf('╔═══════════════════════════════════════════════════════╗\n');
    fprintf('║  TEST CASE 6: STRESS/NOISE/LATENCY TESTS              ║\n');
    fprintf('╚═══════════════════════════════════════════════════════╝\n');
    
    fprintf('[%d] Baseline (clean data)\n', test_count+1);
    thefunc_dcr_floor_tuner_real_data('TEST6a_baseline_clean.csv', false);
    test_count = test_count + 1;
    
    fprintf('[%d] With 10%% white noise\n', test_count+1);
    thefunc_dcr_floor_tuner_real_data('TEST6b_with_10pct_noise.csv', false);
    test_count = test_count + 1;
    
    fprintf('[%d] With 50ms latency\n', test_count+1);
    thefunc_dcr_floor_tuner_real_data('TEST6c_with_50ms_latency.csv', false);
    test_count = test_count + 1;
    
    fprintf('[%d] With 5%% dropout\n', test_count+1);
    thefunc_dcr_floor_tuner_real_data('TEST6d_with_5pct_dropout.csv', false);
    test_count = test_count + 1;
    
    fprintf('[%d] Combined stress (15%% noise + 50ms latency + 8%% dropout)\n', test_count+1);
    thefunc_dcr_floor_tuner_real_data('TEST6e_combined_stress.csv', false);
    test_count = test_count + 1;
    fprintf('✓ Complete\n\n');
    
    %% SUMMARY
    fprintf('\n╔═══════════════════════════════════════════════════════╗\n');
    fprintf('║  COMPREHENSIVE TEST COMPLETE                          ║\n');
    fprintf('╚═══════════════════════════════════════════════════════╝\n\n');
    fprintf('Total simulations run: %d\n', test_count);
    fprintf('All JSON output files saved in current directory\n\n');
end

%% ============================================================
%% SPECIFIC TEST
%% ============================================================
function run_specific_test()
    fprintf('\n═══ SELECT SPECIFIC TEST CASE ═══\n\n');
    fprintf('  1. Stationary Wind\n');
    fprintf('  2. Turbulent Wind\n');
    fprintf('  3. Small Earthquake (M 4.5)\n');
    fprintf('  4. Large Earthquake (M 6.9)\n');
    fprintf('  5. Mixed Seismic-Wind Input\n');
    fprintf('  6. Stress/Noise/Latency Tests\n\n');
    
    test_num = input('Enter test case number (1-6): ');
    
    switch test_num
        case 1
            fprintf('\n→ Running Test Case 1: Stationary Wind\n');
            thefunc_dcr_floor_tuner_real_data('el_centro', true, 'TEST1_stationary_wind_12ms.csv');
            
        case 2
            fprintf('\n→ Running Test Case 2: Turbulent Wind\n');
            thefunc_dcr_floor_tuner_real_data('el_centro', true, 'TEST2_turbulent_wind_25ms.csv');
            
        case 3
            fprintf('\n→ Running Test Case 3: Small Earthquake\n');
            thefunc_dcr_floor_tuner_real_data('TEST3_small_earthquake_M4.5.csv', false);
            
        case 4
            fprintf('\n→ Running Test Case 4: Large Earthquake\n');
            thefunc_dcr_floor_tuner_real_data('TEST4_large_earthquake_M6.9.csv', false);
            
        case 5
            fprintf('\n→ Running Test Case 5: Mixed Seismic-Wind\n');
            thefunc_dcr_floor_tuner_real_data('TEST5_earthquake_M6.7.csv', true, 'TEST5_hurricane_wind_50ms.csv');
            
        case 6
            fprintf('\n→ Running Test Case 6: Stress Tests\n');
            fprintf('  Select stress type:\n');
            fprintf('    1. Baseline (clean)\n');
            fprintf('    2. 10%% noise\n');
            fprintf('    3. 50ms latency\n');
            fprintf('    4. 5%% dropout\n');
            fprintf('    5. Combined stress\n');
            fprintf('    6. All of above\n');
            stress_choice = input('  Enter choice (1-6): ');
            
            switch stress_choice
                case 1
                    thefunc_dcr_floor_tuner_real_data('TEST6a_baseline_clean.csv', false);
                case 2
                    thefunc_dcr_floor_tuner_real_data('TEST6b_with_10pct_noise.csv', false);
                case 3
                    thefunc_dcr_floor_tuner_real_data('TEST6c_with_50ms_latency.csv', false);
                case 4
                    thefunc_dcr_floor_tuner_real_data('TEST6d_with_5pct_dropout.csv', false);
                case 5
                    thefunc_dcr_floor_tuner_real_data('TEST6e_combined_stress.csv', false);
                case 6
                    fprintf('Running all stress tests...\n');
                    thefunc_dcr_floor_tuner_real_data('TEST6a_baseline_clean.csv', false);
                    thefunc_dcr_floor_tuner_real_data('TEST6b_with_10pct_noise.csv', false);
                    thefunc_dcr_floor_tuner_real_data('TEST6c_with_50ms_latency.csv', false);
                    thefunc_dcr_floor_tuner_real_data('TEST6d_with_5pct_dropout.csv', false);
                    thefunc_dcr_floor_tuner_real_data('TEST6e_combined_stress.csv', false);
            end
            
        otherwise
            fprintf('Invalid choice.\n');
    end
end

%% ============================================================
%% HELPER FUNCTIONS
%% ============================================================

function exists = check_datasets_exist()
    % Check if all required dataset files exist
    required_files = {
        'TEST1_stationary_wind_12ms.csv'
        'TEST2_turbulent_wind_25ms.csv'
        'TEST3_small_earthquake_M4.5.csv'
        'TEST4_large_earthquake_M6.9.csv'
        'TEST5_earthquake_M6.7.csv'
        'TEST5_hurricane_wind_50ms.csv'
        'TEST6a_baseline_clean.csv'
        'TEST6b_with_10pct_noise.csv'
        'TEST6c_with_50ms_latency.csv'
        'TEST6d_with_5pct_dropout.csv'
        'TEST6e_combined_stress.csv'
    };
    
    exists = true;
    for i = 1:length(required_files)
        if ~isfile(required_files{i})
            exists = false;
            return;
        end
    end
end

%% ============================================================
%% END OF SCRIPT
%% ============================================================