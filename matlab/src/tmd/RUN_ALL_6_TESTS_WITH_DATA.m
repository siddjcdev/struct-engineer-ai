% ============================================================
% RUN ALL 6 TEST CASES WITH V7 ENHANCED TMD TUNER
% ============================================================
% This script runs all 6 test cases using the V7 enhanced optimizer
% 
% Prerequisites: Dataset CSV files must exist!
% Run create_all_6_test_datasets.m first if needed
% ============================================================
function RUN_ALL_6_TESTS_WITH_DATA()
    
    fprintf('\n╔════════════════════════════════════════════════════════╗\n');
    fprintf('║  RUNNING ALL 6 TEST CASES WITH V7 OPTIMIZER           ║\n');
    fprintf('╚════════════════════════════════════════════════════════╝\n\n');
    
    fprintf('V7 IMPROVEMENTS:\n');
    fprintf('  ✓ Multi-objective optimization (DCR + drift + roof)\n');
    fprintf('  ✓ Adaptive parameter grid based on response\n');
    fprintf('  ✓ Intelligent floor selection (modal + DCR)\n');
    fprintf('  ✓ Enhanced DCR calculation (75th percentile)\n');
    fprintf('  ✓ Smart guard constraints\n');
    fprintf('  ✓ Performance rating system\n\n');
    
    % Check datasets
    if ~check_datasets_exist()
        fprintf('❌ Dataset files not found!\n\n');
        response = input('Generate datasets now? (y/n): ', 's');
        if strcmpi(response, 'y')
            create_all_6_test_datasets();
        else
            return;
        end
    end
    
    fprintf('✓ All dataset files found!\n\n');
    
    % Menu
    fprintf('Options:\n');
    fprintf('  1. Quick demo (one test from each case, ~6 min)\n');
    fprintf('  2. Run all comprehensive (~15 min)\n');
    fprintf('  3. Run specific test case\n');
    fprintf('  4. Compare v5/v6 vs v7 performance\n\n');
    
    choice = input('Enter choice (1-4): ');
    
    switch choice
        case 1
            run_quick_demo_v7();
        case 2
            run_all_comprehensive_v7();
        case 3
            run_specific_test_v7();
        case 4
            compare_versions();
        otherwise
            fprintf('Invalid choice. Running quick demo...\n');
            run_quick_demo_v7();
    end
    
    fprintf('\n✓ All simulations complete!\n');
    fprintf('Check JSON files: tmd_v7_simulation_*.json\n\n');
end

%% ============================================================
%% QUICK DEMO WITH V7
%% ============================================================
function run_quick_demo_v7()
    fprintf('\n═══ V7 QUICK DEMO ═══\n\n');
    
    test_count = 0;
    folder = '../../datasets'; % relative folder name

    fprintf('[%d/6] Test 1: Stationary Wind + Earthquake\n', test_count+1);
    thefunc_dcr_floor_tuner_v7('el_centro', true, fullfile(folder,'TEST1_stationary_wind_12ms.csv'));
    test_count = test_count + 1;
    fprintf('✓ Complete\n\n');
    pause(1);
    
    fprintf('[%d/6] Test 2: Turbulent Wind + Earthquake\n', test_count+1);
    thefunc_dcr_floor_tuner_v7('el_centro', true, fullfile(folder,'TEST2_turbulent_wind_25ms.csv'));
    test_count = test_count + 1;
    fprintf('✓ Complete\n\n');
    pause(1);
    
    fprintf('[%d/6] Test 3: Small Earthquake (M 4.5)\n', test_count+1);
    thefunc_dcr_floor_tuner_v7(fullfile(folder,'TEST3_small_earthquake_M4.5.csv'), false);
    test_count = test_count + 1;
    fprintf('✓ Complete\n\n');
    pause(1);
    
    fprintf('[%d/6] Test 4: Large Earthquake (M 6.9)\n', test_count+1);
    thefunc_dcr_floor_tuner_v7(fullfile(folder,'TEST4_large_earthquake_M6.9.csv'), false);
    test_count = test_count + 1;
    fprintf('✓ Complete\n\n');
    pause(1);
    
    fprintf('[%d/6] Test 5: Extreme Combined Loading\n', test_count+1);
    thefunc_dcr_floor_tuner_v7(fullfile(folder,'TEST5_earthquake_M6.7.csv'), true, fullfile(folder, 'TEST5_hurricane_wind_50ms.csv'));
    test_count = test_count + 1;
    fprintf('✓ Complete\n\n');
    pause(1);
    
    fprintf('[%d/6] Test 6: Stress Test (10%% noise)\n', test_count+1);
    thefunc_dcr_floor_tuner_v7(fullfile(folder,'TEST6b_with_10pct_noise.csv'), false);
    test_count = test_count + 1;
    fprintf('✓ Complete\n\n');
    
    fprintf('\n═══ V7 QUICK DEMO COMPLETE ═══\n');
    fprintf('%d simulations completed\n\n', test_count);
end

%% ============================================================
%% COMPREHENSIVE TEST
%% ============================================================
function run_all_comprehensive_v7()
    fprintf('\n═══ COMPREHENSIVE V7 TEST SUITE ═══\n\n');
    
    test_count = 0;
    folder = 'datasets'; % relative folder name
    
    % Test Case 1
    fprintf('╔═══════════════════════════════════════════════════════╗\n');
    fprintf('║  TEST CASE 1: STATIONARY WIND                         ║\n');
    fprintf('╚═══════════════════════════════════════════════════════╝\n');
    thefunc_dcr_floor_tuner_v7('el_centro', true, fullfile(folder,'TEST1_stationary_wind_12ms.csv'));
    test_count = test_count + 1;
    fprintf('\n');
    
    % Test Case 2
    fprintf('╔═══════════════════════════════════════════════════════╗\n');
    fprintf('║  TEST CASE 2: TURBULENT WIND                          ║\n');
    fprintf('╚═══════════════════════════════════════════════════════╝\n');
    thefunc_dcr_floor_tuner_v7('el_centro', true, fullfile(folder,'TEST2_turbulent_wind_25ms.csv'));
    test_count = test_count + 1;
    fprintf('\n');
    
    % Test Case 3
    fprintf('╔═══════════════════════════════════════════════════════╗\n');
    fprintf('║  TEST CASE 3: SMALL EARTHQUAKE                        ║\n');
    fprintf('╚═══════════════════════════════════════════════════════╝\n');
    thefunc_dcr_floor_tuner_v7(fullfile(folder,'TEST3_small_earthquake_M4.5.csv'), false);
    test_count = test_count + 1;
    fprintf('\n');
    
    % Test Case 4
    fprintf('╔═══════════════════════════════════════════════════════╗\n');
    fprintf('║  TEST CASE 4: LARGE EARTHQUAKE                        ║\n');
    fprintf('╚═══════════════════════════════════════════════════════╝\n');
    thefunc_dcr_floor_tuner_v7(fullfile(folder,'TEST4_large_earthquake_M6.9.csv'), false);
    test_count = test_count + 1;
    fprintf('\n');
    
    % Test Case 5
    fprintf('╔═══════════════════════════════════════════════════════╗\n');
    fprintf('║  TEST CASE 5: MIXED LOADING                           ║\n');
    fprintf('╚═══════════════════════════════════════════════════════╝\n');
    thefunc_dcr_floor_tuner_v7(fullfile(folder,'TEST5_earthquake_M6.7.csv'), true, fullfile(folder,'TEST5_hurricane_wind_50ms.csv'));
    test_count = test_count + 1;
    fprintf('\n');
    
    % Test Case 6
    fprintf('╔═══════════════════════════════════════════════════════╗\n');
    fprintf('║  TEST CASE 6: STRESS TESTS                            ║\n');
    fprintf('╚═══════════════════════════════════════════════════════╝\n');
    
    fprintf('  [%d] Baseline (clean)\n', test_count+1);
    thefunc_dcr_floor_tuner_v7(fullfile(folder,'TEST6a_baseline_clean.csv'), false);
    test_count = test_count + 1;
    
    fprintf('  [%d] 10%% noise\n', test_count+1);
    thefunc_dcr_floor_tuner_v7(fullfile(folder,'TEST6b_with_10pct_noise.csv'), false);
    test_count = test_count + 1;
    
    fprintf('  [%d] 50ms latency\n', test_count+1);
    thefunc_dcr_floor_tuner_v7(fullfile(folder,'TEST6c_with_50ms_latency.csv'), false);
    test_count = test_count + 1;
    
    fprintf('  [%d] 5%% dropout\n', test_count+1);
    thefunc_dcr_floor_tuner_v7(fullfile(folder,'TEST6d_with_5pct_dropout.csv'), false);
    test_count = test_count + 1;
    
    fprintf('  [%d] Combined stress\n', test_count+1);
    thefunc_dcr_floor_tuner_v7(fullfile(folder,'TEST6e_combined_stress.csv'), false);
    test_count = test_count + 1;
    
    fprintf('\n╔═══════════════════════════════════════════════════════╗\n');
    fprintf('║  COMPREHENSIVE TEST COMPLETE                          ║\n');
    fprintf('╚═══════════════════════════════════════════════════════╝\n');
    fprintf('Total simulations: %d\n\n', test_count);
end

%% ============================================================
%% SPECIFIC TEST
%% ============================================================
function run_specific_test_v7()
    fprintf('\n═══ SELECT TEST CASE ═══\n\n');
    fprintf('  1. Stationary Wind\n');
    fprintf('  2. Turbulent Wind\n');
    fprintf('  3. Small Earthquake (M 4.5)\n');
    fprintf('  4. Large Earthquake (M 6.9)\n');
    fprintf('  5. Mixed Seismic-Wind\n');
    fprintf('  6. Stress Tests\n\n');
    
    test_num = input('Enter test number (1-6): ');
    folder = '../../datasets'; % relative folder name
    
    switch test_num
        case 1
            thefunc_dcr_floor_tuner_v7('el_centro', true, fullfile(folder,'TEST1_stationary_wind_12ms.csv'));
        case 2
            thefunc_dcr_floor_tuner_v7('el_centro', true, fullfile(folder,'TEST2_turbulent_wind_25ms.csv'));
        case 3
            thefunc_dcr_floor_tuner_v7(fullfile(folder,'TEST3_small_earthquake_M4.5.csv'), false);
        case 4
            thefunc_dcr_floor_tuner_v7(fullfile(folder,'TEST4_large_earthquake_M6.9.csv'), false);
        case 5
            thefunc_dcr_floor_tuner_v7(fullfile(folder,'TEST5_earthquake_M6.7.csv'), true, fullfile(folder,'TEST5_hurricane_wind_50ms.csv'));
        case 6
            fprintf('  Select stress type:\n');
            fprintf('    1. Baseline\n    2. 10%% noise\n');
            fprintf('    3. 50ms latency\n    4. 5%% dropout\n');
            fprintf('    5. Combined\n    6. All\n');
            stress = input('  Choice: ');
            
            switch stress
                case 1, thefunc_dcr_floor_tuner_v7(fullfile(folder,'TEST6a_baseline_clean.csv'), false);
                case 2, thefunc_dcr_floor_tuner_v7(fullfile(folder,'TEST6b_with_10pct_noise.csv'), false);
                case 3, thefunc_dcr_floor_tuner_v7(fullfile(folder,'TEST6c_with_50ms_latency.csv'), false);
                case 4, thefunc_dcr_floor_tuner_v7(fullfile(folder,'TEST6d_with_5pct_dropout.csv'), false);
                case 5, thefunc_dcr_floor_tuner_v7(fullfile(folder,'TEST6e_combined_stress.csv'), false);
                case 6
                    thefunc_dcr_floor_tuner_v7(fullfile(folder,'TEST6a_baseline_clean.csv'), false);
                    thefunc_dcr_floor_tuner_v7(fullfile(folder,'TEST6b_with_10pct_noise.csv'), false);
                    thefunc_dcr_floor_tuner_v7(fullfile(folder,'TEST6c_with_50ms_latency.csv'), false);
                    thefunc_dcr_floor_tuner_v7(fullfile(folder,'TEST6d_with_5pct_dropout.csv'), false);
                    thefunc_dcr_floor_tuner_v7(fullfile(folder,'TEST6e_combined_stress.csv'), false);
            end
        otherwise
            fprintf('Invalid choice.\n');
    end
end

%% ============================================================
%% VERSION COMPARISON
%% ============================================================
function compare_versions()
    fprintf('\n═══ V5/V6 vs V7 COMPARISON ═══\n\n');
    fprintf('This will run the same test with both versions.\n');
    fprintf('Select test case:\n');
    fprintf('  1. Test 1 (Stationary Wind)\n');
    fprintf('  2. Test 4 (Large Earthquake)\n');
    fprintf('  3. Test 5 (Extreme Combined)\n\n');
    
    choice = input('Choice: ');
    folder = 'datasets'; % relative folder name
    
    switch choice
        case 1
            eq = 'el_centro';
            wind_file = fullfile(folder,'TEST1_stationary_wind_12ms.csv');
            use_wind = true;
        case 2
            eq = fullfile(folder,'TEST4_large_earthquake_M6.9.csv');
            wind_file = '';
            use_wind = false;
        case 3
            eq = fullfile(folder,'TEST5_earthquake_M6.7.csv');
            wind_file = fullfile(folder,'TEST5_hurricane_wind_50ms.csv');
            use_wind = true;
        otherwise
            fprintf('Invalid choice.\n');
            return;
    end
    
    fprintf('\n--- Running V5 ---\n');
    if use_wind
        thefunc_dcr_floor_tuner_v5(eq, true, wind_file);
    else
        thefunc_dcr_floor_tuner_v5(eq, false);
    end
    
    fprintf('\n--- Running V7 ---\n');
    if use_wind
        thefunc_dcr_floor_tuner_v7(eq, true, wind_file);
    else
        thefunc_dcr_floor_tuner_v7(eq, false);
    end
    
    fprintf('\n═══ COMPARISON COMPLETE ═══\n');
    fprintf('Compare the two JSON files generated.\n');
    fprintf('V5: tmd_simulation_*.json\n');
    fprintf('V7: tmd_v7_simulation_*.json\n\n');
end

%% ============================================================
%% HELPER
%% ============================================================
function exists = check_datasets_exist()
    % Folder where datasets are stored
    folder = 'datasets';

    % List of required dataset files
    required = {
        'TEST3_small_earthquake_M4.5.csv'
        'TEST4_large_earthquake_M6.9.csv'
        'TEST5_earthquake_M6.7.csv'
        'TEST5_hurricane_wind_50ms.csv'
        'TEST6a_baseline_clean.csv'
        'TEST6b_with_10pct_noise.csv'
        'TEST6c_with_50ms_latency.csv'
    };

    % Assume all exist until proven otherwise
    exists = true;

    % Loop through each required file
    for i = 1:length(required)
        % Build full path to file inside the folder
        filepath = fullfile(['../../' folder], required{i});

        fprintf('filepath : %s',filepath)

        % Check if file exists
        if ~isfile(filepath)
            exists = false;
            return;
        end
    end
end
