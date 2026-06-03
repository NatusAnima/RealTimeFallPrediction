function run_offline_metrics_gui()
% run_offline_metrics_gui
% GUI to evaluate models, train on select subjects, and run full LOSO CV.

clear all; close all; clc;

% --- 1. GUI Setup ---
fig = uifigure('Name', 'Offline Metrics & Training Console', 'Position', [200, 100, 600, 700]);
fig.Color = [0.94 0.94 0.94];

uilabel(fig, 'Position', [100, 650, 400, 30], 'Text', 'Offline Data Metrics Evaluator', ...
    'FontSize', 20, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% Mode Selection
uilabel(fig, 'Position', [50, 600, 100, 30], 'Text', 'Select Mode:', 'FontSize', 14, 'FontWeight', 'bold');
mode_dropdown = uidropdown(fig, 'Position', [160, 605, 200, 25], ...
    'Items', {'Evaluate Pre-trained', 'Evaluate Live Simulation (Continuous)', 'Train on Selection', 'Run Full LOSO'}, ...
    'ValueChangedFcn', @(dd,event) mode_changed(dd, fig));

% Target Selection (Dynamic)
uilabel(fig, 'Position', [50, 550, 100, 30], 'Text', 'Select Target:', 'FontSize', 14, 'FontWeight', 'bold');

raw_data_dir = fullfile(pwd, 'Raw_Data');
table_path = fullfile(raw_data_dir, 'All34_table.mat');
if exist(table_path, 'file')
    load(table_path, 'event_table');
    subjects_list = event_table.Subject';
    dropdown_items = [{'All Subjects'}, arrayfun(@(x) sprintf('Subject %d', x), subjects_list, 'UniformOutput', false)];
    listbox_items = arrayfun(@(x) sprintf('Subject %d', x), subjects_list, 'UniformOutput', false);
else
    dropdown_items = {'All Subjects'};
    listbox_items = {'No Data'};
    subjects_list = [];
end

% Single Dropdown (for Evaluate)
subj_dropdown = uidropdown(fig, 'Position', [160, 555, 200, 25], 'Items', dropdown_items);

% Multi Listbox (for Train)
subj_listbox = uilistbox(fig, 'Position', [160, 450, 200, 130], 'Items', listbox_items, 'MultiSelect', 'on', 'Visible', 'off');

run_btn = uibutton(fig, 'push', 'Position', [400, 555, 120, 40], 'Text', 'Execute', ...
    'FontWeight', 'bold', 'BackgroundColor', [0.2 0.6 0.8], 'FontColor', 'w', ...
    'ButtonPushedFcn', @(btn,event) run_action(fig));

% Results Text Area
results_text = uitextarea(fig, 'Position', [50, 20, 500, 410], 'Editable', 'off', ...
    'FontName', 'Courier New', 'FontSize', 12);
results_text.Value = {'Ready.'};

fig.UserData = struct('results_text', results_text, 'subjects_list', subjects_list, ...
    'raw_data_dir', raw_data_dir, 'mode_dropdown', mode_dropdown, ...
    'subj_dropdown', subj_dropdown, 'subj_listbox', subj_listbox);

end

% --- UI Callbacks ---
function mode_changed(dd, fig)
    val = dd.Value;
    ud = fig.UserData;
    if strcmp(val, 'Evaluate Pre-trained') || strcmp(val, 'Evaluate Live Simulation (Continuous)')
        ud.subj_dropdown.Visible = 'on';
        ud.subj_listbox.Visible = 'off';
    elseif strcmp(val, 'Train on Selection')
        ud.subj_dropdown.Visible = 'off';
        ud.subj_listbox.Visible = 'on';
    elseif strcmp(val, 'Run Full LOSO')
        ud.subj_dropdown.Visible = 'off';
        ud.subj_listbox.Visible = 'off';
    end
end

function run_action(fig)
    ud = fig.UserData;
    mode = ud.mode_dropdown.Value;
    
    if strcmp(mode, 'Evaluate Pre-trained')
        execute_evaluation(fig);
    elseif strcmp(mode, 'Evaluate Live Simulation (Continuous)')
        execute_continuous_evaluation(fig);
    elseif strcmp(mode, 'Train on Selection')
        execute_training(fig);
    elseif strcmp(mode, 'Run Full LOSO')
        execute_loso(fig);
    end
end

% --- Action Implementations ---

% 1. Evaluation Mode
function execute_evaluation(fig)
    ud = fig.UserData;
    results_ta = ud.results_text;
    results_ta.Value = {'Loading model... Please wait.'};
    drawnow;
    
    try
        load('trained_model.mat', 'mdl_eeg', 'mdl_imu', 'mdl_fusion', 'global_eeg_cap', 'b', 'a');
    catch
        results_ta.Value = {'ERROR: trained_model.mat not found.'};
        return;
    end
    
    val = ud.subj_dropdown.Value;
    if strcmp(val, 'All Subjects')
        target_subjs = ud.subjects_list;
    else
        target_subjs = sscanf(val, 'Subject %d');
    end
    
    if isempty(target_subjs), return; end
    
    TP = 0; TN = 0; FP = 0; FN = 0;
    FP_Class0 = 0; FP_Class2 = 0;
    total_events = 0;
    all_p_falls = [];
    all_true_labels = [];
    
    wb = uiprogressdlg(fig, 'Title', 'Evaluating Metrics', 'Message', 'Processing subjects...');
    
    for i = 1:length(target_subjs)
        subj = target_subjs(i);
        wb.Value = i / length(target_subjs);
        wb.Message = sprintf('Evaluating Subject %d (%d/%d)', subj, i, length(target_subjs));
        
        [sub_TP, sub_TN, sub_FP, sub_FN, sub_FP0, sub_FP2, sub_tot, p_f, t_l] = evaluate_subject(subj, ud.raw_data_dir, b, a, global_eeg_cap, mdl_eeg, mdl_imu, mdl_fusion);
        
        all_p_falls = [all_p_falls; p_f];
        all_true_labels = [all_true_labels; t_l];
        
        TP = TP + sub_TP; TN = TN + sub_TN;
        FP = FP + sub_FP; FN = FN + sub_FN;
        FP_Class0 = FP_Class0 + sub_FP0; FP_Class2 = FP_Class2 + sub_FP2;
        total_events = total_events + sub_tot;
    end
    close(wb);
    
    figure('Name', 'p_fall Distributions');
    hold on;
    histogram(all_p_falls(all_true_labels == 1), 'BinWidth', 0.05, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'Normalization', 'probability');
    histogram(all_p_falls(all_true_labels == 0 | all_true_labels == 2), 'BinWidth', 0.05, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'Normalization', 'probability');
    legend('Real Falls (Label 1)', 'Non-Falls (Label 0 & 2)');
    xlabel('p\_fall score');
    ylabel('Probability Density');
    title('Distribution of p\_fall scores (Gate Disabled)');
    grid on;
    hold off;
    
    Sens = TP / max(1, (TP + FN));
    Spec = TN / max(1, (TN + FP));
    BAcc = (Sens + Spec) / 2;
    
    results_ta.Value = {
        '========================================='
        '        OFFLINE EVALUATION METRICS       '
        '========================================='
        sprintf(' Target            : %s', val)
        sprintf(' Total Events Eval : %d', total_events)
        '-----------------------------------------'
        sprintf(' True Positives (TP)  : %d', TP)
        sprintf(' True Negatives (TN)  : %d', TN)
        sprintf(' False Positives (FP) : %d', FP)
        sprintf(' False Negatives (FN) : %d', FN)
        '-----------------------------------------'
        sprintf(' Sensitivity (Recall) : %.4f', Sens)
        sprintf(' Specificity          : %.4f', Spec)
        sprintf(' Balanced Accuracy    : %.4f', BAcc)
        '-----------------------------------------'
        sprintf(' Class 0 (Movement FP): %d', FP_Class0)
        sprintf(' Class 2 (Silence FP) : %d', FP_Class2)
        '========================================='
    };
end

% 2. Train Mode
function execute_training(fig)
    ud = fig.UserData;
    results_ta = ud.results_text;
    selected_items = ud.subj_listbox.Value;
    
    if isempty(selected_items)
        results_ta.Value = {'ERROR: No subjects selected for training.'};
        return;
    end
    
    target_subjs = zeros(1, length(selected_items));
    for k = 1:length(selected_items)
        target_subjs(k) = sscanf(selected_items{k}, 'Subject %d');
    end
    
    results_ta.Value = {'Training new model... Please wait.'};
    drawnow;
    
    fs = 1000;
    [b, a] = butter(2, [2.5, 30] / (fs/2), 'bandpass');
    
    num_t = length(target_subjs);
    subj_eeg_cap = ones(num_t, 1);
    
    wb = uiprogressdlg(fig, 'Title', 'Training Model', 'Message', 'Pass 1: Calculating Global Cap...');
    
    parfor i = 1:num_t
        subj = target_subjs(i);
        subj_eeg_cap(i) = calculate_subject_cap(subj, ud.raw_data_dir, b, a);
    end
    
    global_eeg_cap = max(subj_eeg_cap);
    
    wb.Message = 'Pass 2: Extracting features...';
    wb.Value = 0.3;
    drawnow;
    
    subj_X = cell(num_t, 1);
    subj_X_imu = cell(num_t, 1);
    subj_X_heur = cell(num_t, 1);
    subj_Y = cell(num_t, 1);
    subj_idx_cell = cell(num_t, 1);
    
    parfor i = 1:num_t
        subj = target_subjs(i);
        [local_X, local_X_imu, local_X_heur, local_Y] = extract_subject_features(subj, ud.raw_data_dir, b, a, global_eeg_cap);
        subj_X{i} = local_X;
        subj_X_imu{i} = local_X_imu;
        subj_X_heur{i} = local_X_heur;
        subj_Y{i} = local_Y;
        subj_idx_cell{i} = i * ones(size(local_Y, 1), 1);
    end
    
    wb.Message = 'Aggregating and Training RF...';
    wb.Value = 0.7;
    drawnow;
    
    X_eeg = cell2mat(subj_X);
    X_imu = cell2mat(subj_X_imu);
    X_heur = cell2mat(subj_X_heur);
    Y_train = vertcat(subj_Y{:});
    all_subj_idx = cell2mat(subj_idx_cell);
    
    % Calculate observation weights
    obs_weights = ones(size(Y_train, 1), 1);
    idx_class1 = strcmp(Y_train, '1');
    obs_weights(idx_class1) = 5.0; % 5x weight for minority 'Real Fall' class
    
    K = 5;
    if num_t < K, K = num_t; end
    
    p_eeg_oof = zeros(size(Y_train, 1), 1);
    p_imu_oof = zeros(size(Y_train, 1), 1);
    
    if K > 1
        cv_subj = cvpartition(num_t, 'KFold', K);
        for k = 1:K
            test_subjs_mask = test(cv_subj, k);
            train_subjs_mask = training(cv_subj, k);
            
            test_idx = ismember(all_subj_idx, find(test_subjs_mask));
            train_idx = ismember(all_subj_idx, find(train_subjs_mask));
            
            template_eeg = templateTree('MaxNumSplits', sum(train_idx) - 1);
            mdl_eeg_fold = fitcensemble(X_eeg(train_idx, :), Y_train(train_idx), 'Method', 'Bag', 'NumLearningCycles', 30, 'Learners', template_eeg, 'ClassNames', {'0', '1'}, 'Weights', obs_weights(train_idx));
            
            template_imu = templateTree('MaxNumSplits', sum(train_idx) - 1);
            mdl_imu_fold = fitcensemble(X_imu(train_idx, :), Y_train(train_idx), 'Method', 'Bag', 'NumLearningCycles', 30, 'Learners', template_imu, 'ClassNames', {'0', '1'}, 'Weights', obs_weights(train_idx));
            
            [~, score_eeg] = predict(mdl_eeg_fold, X_eeg(test_idx, :));
            [~, score_imu] = predict(mdl_imu_fold, X_imu(test_idx, :));
            
            idx_1_eeg = find(strcmp(mdl_eeg_fold.ClassNames, '1'));
            idx_1_imu = find(strcmp(mdl_imu_fold.ClassNames, '1'));
            
            p_eeg_oof(test_idx) = score_eeg(:, idx_1_eeg);
            p_imu_oof(test_idx) = score_imu(:, idx_1_imu);
        end
    end
    
    template_eeg = templateTree('MaxNumSplits', size(X_eeg, 1) - 1);
    mdl_eeg = fitcensemble(X_eeg, Y_train, 'Method', 'Bag', 'NumLearningCycles', 30, 'Learners', template_eeg, 'ClassNames', {'0', '1'}, 'Weights', obs_weights);
    
    template_imu = templateTree('MaxNumSplits', size(X_imu, 1) - 1);
    mdl_imu = fitcensemble(X_imu, Y_train, 'Method', 'Bag', 'NumLearningCycles', 30, 'Learners', template_imu, 'ClassNames', {'0', '1'}, 'Weights', obs_weights);
    
    X_fusion_train = [p_eeg_oof, p_imu_oof, X_heur];
    Y_train_cat = categorical(Y_train);
    mdl_fusion = fitglm(X_fusion_train, Y_train_cat == '1', 'Distribution', 'binomial', 'Weights', obs_weights);
    
    % Generate file name based on selected subjects
    if length(target_subjs) > 5
        filename = sprintf('trained_model_custom_%d_subjs.mat', length(target_subjs));
    else
        filename = ['trained_model_subj_', strjoin(arrayfun(@num2str, target_subjs, 'UniformOutput', false), '_'), '.mat'];
    end
    
    save(filename, 'mdl_eeg', 'mdl_imu', 'mdl_fusion', 'global_eeg_cap', 'b', 'a');
    close(wb);
    
    results_ta.Value = {
        '========================================='
        '          TRAINING COMPLETE              '
        '========================================='
        sprintf(' Trained on %d Subjects.', length(target_subjs))
        sprintf(' Total Samples   : %d', size(X_eeg, 1))
        sprintf(' Global EEG Cap  : %.4f', global_eeg_cap)
        '-----------------------------------------'
        sprintf(' Model successfully saved as:')
        sprintf(' %s', filename)
        '========================================='
    };
end

% 3. LOSO Mode
function execute_loso(fig)
    ud = fig.UserData;
    results_ta = ud.results_text;
    all_subjs = ud.subjects_list;
    
    if isempty(all_subjs)
        results_ta.Value = {'ERROR: No subjects available for LOSO.'};
        return;
    end
    
    results_ta.Value = {'Starting Full Leave-One-Subject-Out CV...'};
    drawnow;
    
    fs = 1000;
    [b, a] = butter(2, [2.5, 30] / (fs/2), 'bandpass');
    num_subjs = length(all_subjs);
    
    wb = uiprogressdlg(fig, 'Title', 'LOSO Cross-Validation', 'Message', 'Pass 1: Calculating Global Cap...');
    
    subj_eeg_cap = ones(num_subjs, 1);
    parfor i = 1:num_subjs
        subj_eeg_cap(i) = calculate_subject_cap(all_subjs(i), ud.raw_data_dir, b, a);
    end
    global_eeg_cap = max(subj_eeg_cap);
    
    wb.Message = 'Pass 2: Extracting all features (Parallel)...';
    wb.Value = 0.2;
    
    subj_X = cell(num_subjs, 1);
    subj_X_imu = cell(num_subjs, 1);
    subj_X_heur = cell(num_subjs, 1);
    subj_Y = cell(num_subjs, 1);
    subj_idx_cell = cell(num_subjs, 1);
    
    parfor i = 1:num_subjs
        [subj_X{i}, subj_X_imu{i}, subj_X_heur{i}, subj_Y{i}] = extract_subject_features(all_subjs(i), ud.raw_data_dir, b, a, global_eeg_cap);
        subj_idx_cell{i} = i * ones(size(subj_Y{i}, 1), 1);
    end
    
    TP = 0; TN = 0; FP = 0; FN = 0;
    FP_Class0 = 0; FP_Class2 = 0;
    total_events = 0;
    all_p_falls = [];
    all_true_labels = [];
    
    for test_idx = 1:num_subjs
        wb.Value = min(1.0, 0.2 + (0.8 * test_idx / num_subjs));
        wb.Message = sprintf('LOSO Iteration %d/%d (Held-out: Subj %d)', test_idx, num_subjs, all_subjs(test_idx));
        
        train_mask = true(num_subjs, 1);
        train_mask(test_idx) = false;
        
        X_eeg = cell2mat(subj_X(train_mask));
        X_imu = cell2mat(subj_X_imu(train_mask));
        X_heur = cell2mat(subj_X_heur(train_mask));
        Y_train = vertcat(subj_Y{train_mask});
        all_subj_idx = cell2mat(subj_idx_cell(train_mask));
        
        obs_weights = ones(size(Y_train, 1), 1);
        idx_class1 = strcmp(Y_train, '1');
        obs_weights(idx_class1) = 5.0; % 5x weight for minority 'Real Fall' class
        
        num_t = sum(train_mask);
        K = 5;
        if num_t < K; K = num_t; end
        
        p_eeg_oof = zeros(size(Y_train, 1), 1);
        p_imu_oof = zeros(size(Y_train, 1), 1);
        
        if K > 1
            cv_subj = cvpartition(num_t, 'KFold', K);
            for k = 1:K
                test_subjs_mask = test(cv_subj, k);
                train_subjs_mask = training(cv_subj, k);
                
                t_idx = ismember(all_subj_idx, find(train_subjs_mask));
                ts_idx = ismember(all_subj_idx, find(test_subjs_mask));
                
                template_eeg = templateTree('MaxNumSplits', sum(t_idx) - 1);
                mdl_eeg_fold = fitcensemble(X_eeg(t_idx, :), Y_train(t_idx), 'Method', 'Bag', 'NumLearningCycles', 30, 'Learners', template_eeg, 'ClassNames', {'0', '1'}, 'Weights', obs_weights(t_idx));
                
                template_imu = templateTree('MaxNumSplits', sum(t_idx) - 1);
                mdl_imu_fold = fitcensemble(X_imu(t_idx, :), Y_train(t_idx), 'Method', 'Bag', 'NumLearningCycles', 30, 'Learners', template_imu, 'ClassNames', {'0', '1'}, 'Weights', obs_weights(t_idx));
                
                [~, score_eeg] = predict(mdl_eeg_fold, X_eeg(ts_idx, :));
                [~, score_imu] = predict(mdl_imu_fold, X_imu(ts_idx, :));
                
                idx_1_eeg = find(strcmp(mdl_eeg_fold.ClassNames, '1'));
                idx_1_imu = find(strcmp(mdl_imu_fold.ClassNames, '1'));
                
                p_eeg_oof(ts_idx) = score_eeg(:, idx_1_eeg);
                p_imu_oof(ts_idx) = score_imu(:, idx_1_imu);
            end
        end
        
        template_eeg = templateTree('MaxNumSplits', size(X_eeg, 1) - 1);
        mdl_eeg = fitcensemble(X_eeg, Y_train, 'Method', 'Bag', 'NumLearningCycles', 30, 'Learners', template_eeg, 'ClassNames', {'0', '1'}, 'Weights', obs_weights);
        
        template_imu = templateTree('MaxNumSplits', size(X_imu, 1) - 1);
        mdl_imu = fitcensemble(X_imu, Y_train, 'Method', 'Bag', 'NumLearningCycles', 30, 'Learners', template_imu, 'ClassNames', {'0', '1'}, 'Weights', obs_weights);
        
        X_fusion_train = [p_eeg_oof, p_imu_oof, X_heur];
        Y_train_cat = categorical(Y_train);
        mdl_fusion = fitglm(X_fusion_train, Y_train_cat == '1', 'Distribution', 'binomial', 'Weights', obs_weights);
        
        % Evaluate on the held out subject
        [sub_TP, sub_TN, sub_FP, sub_FN, sub_FP0, sub_FP2, sub_tot, p_f, t_l] = evaluate_subject(all_subjs(test_idx), ud.raw_data_dir, b, a, global_eeg_cap, mdl_eeg, mdl_imu, mdl_fusion);
        
        all_p_falls = [all_p_falls; p_f];
        all_true_labels = [all_true_labels; t_l];
        
        TP = TP + sub_TP; TN = TN + sub_TN;
        FP = FP + sub_FP; FN = FN + sub_FN;
        FP_Class0 = FP_Class0 + sub_FP0; FP_Class2 = FP_Class2 + sub_FP2;
        total_events = total_events + sub_tot;
        
        results_ta.Value = [results_ta.Value; {sprintf('  Finished Subj %d', all_subjs(test_idx))}];
        drawnow;
    end
    
    close(wb);
    
    figure('Name', 'p_fall Distributions (LOSO)');
    hold on;
    histogram(all_p_falls(all_true_labels == 1), 'BinWidth', 0.05, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'Normalization', 'probability');
    histogram(all_p_falls(all_true_labels == 0 | all_true_labels == 2), 'BinWidth', 0.05, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'Normalization', 'probability');
    legend('Real Falls (Label 1)', 'Non-Falls (Label 0 & 2)');
    xlabel('p\_fall score');
    ylabel('Probability Density');
    title('Distribution of p\_fall scores in LOSO (Gate Disabled)');
    grid on;
    hold off;
    
    Sens = TP / max(1, (TP + FN));
    Spec = TN / max(1, (TN + FP));
    BAcc = (Sens + Spec) / 2;
    
    results_ta.Value = {
        '========================================='
        '    FINAL LOSO CV EVALUATION METRICS     '
        '========================================='
        sprintf(' Total Subjects    : %d', num_subjs)
        sprintf(' Total Events Eval : %d', total_events)
        '-----------------------------------------'
        sprintf(' True Positives (TP)  : %d', TP)
        sprintf(' True Negatives (TN)  : %d', TN)
        sprintf(' False Positives (FP) : %d', FP)
        sprintf(' False Negatives (FN) : %d', FN)
        '-----------------------------------------'
        sprintf(' Sensitivity (Recall) : %.4f', Sens)
        sprintf(' Specificity          : %.4f', Spec)
        sprintf(' Balanced Accuracy    : %.4f', BAcc)
        '-----------------------------------------'
        sprintf(' Class 0 (Movement FP): %d', FP_Class0)
        sprintf(' Class 2 (Silence FP) : %d', FP_Class2)
        '========================================='
    };
end

% --- Helper Functions ---
function [TP, TN, FP, FN, FP0, FP2, tot, p_falls, true_labels] = evaluate_subject(subj, raw_data_dir, b, a, global_cap, mdl_eeg, mdl_imu, mdl_fusion)
    TP = 0; TN = 0; FP = 0; FN = 0; FP0 = 0; FP2 = 0; tot = 0;
    p_falls = []; true_labels = [];
    
    load(fullfile(raw_data_dir, 'All34_table.mat'), 'event_table');
    load(fullfile(raw_data_dir, 'Label_Table.mat'), 'label_table');
    filtered_dir = fullfile(raw_data_dir, 'Filtered');
    my_edfs_dir = dir(fullfile(filtered_dir, '*edf'));
    edfs_names = {my_edfs_dir.name};
    
    if subj < 10
        edf_idx = find(contains(edfs_names, ['0', num2str(subj)]));
    else
        edf_idx = find(contains(edfs_names, num2str(subj)));
    end
    if isempty(edf_idx), return; end
    
    edf_name = fullfile(filtered_dir, edfs_names{edf_idx(1)});
    table_idx = find(event_table.Subject == subj);
    subj_events = event_table{table_idx, 3:end};
    subj_labels = label_table{table_idx, 3:end};
    
    valid_idx = ~isnan(subj_events);
    subj_events = subj_events(valid_idx);
    subj_labels = subj_labels(valid_idx);
    
    try
        EEG_whole = edfread(edf_name, 'SelectedSignals', 'R6', 'DataRecordOutputType', 'vector');
        eeg_sig = cat(1, EEG_whole.(1){:})';
        ACC_X = edfread(edf_name, 'SelectedSignals', 'x_dir', 'DataRecordOutputType', 'vector');
        ACC_Y = edfread(edf_name, 'SelectedSignals', 'y_dir', 'DataRecordOutputType', 'vector');
        ACC_Z = edfread(edf_name, 'SelectedSignals', 'z_dir', 'DataRecordOutputType', 'vector');
        acc_sig = [cat(1, ACC_X.(1){:})'; cat(1, ACC_Y.(1){:})'; cat(1, ACC_Z.(1){:})'] ./ 980;
    catch
        return;
    end
    
    eeg_sig_filtered = filtfilt(b, a, eeg_sig);
    num_class2 = length(subj_events);
    class2_onsets = []; attempts = 0;
    while length(class2_onsets) < num_class2 && attempts < 10000
        rand_onset = randi([1000, length(eeg_sig_filtered) - 2000]);
        if all(abs(subj_events - rand_onset) >= 2000) && all(abs(class2_onsets - rand_onset) >= 2000)
            class2_onsets = [class2_onsets, rand_onset];
        end
        attempts = attempts + 1;
    end
    
    all_onsets_test = [subj_events, class2_onsets];
    all_original_labels_test = [subj_labels, 2 * ones(1, length(class2_onsets))];
    
    num_events = length(all_onsets_test);
    local_TP = zeros(num_events, 1);
    local_TN = zeros(num_events, 1);
    local_FP = zeros(num_events, 1);
    local_FN = zeros(num_events, 1);
    local_FP0 = zeros(num_events, 1);
    local_FP2 = zeros(num_events, 1);
    local_total = zeros(num_events, 1);
    local_p_fall = zeros(num_events, 1);
    local_labels = zeros(num_events, 1);
    
    delay_time = 80; window_size = 256; sys_order = 3;
    
    parfor e_idx = 1:num_events
        onset = all_onsets_test(e_idx);
        orig_label = all_original_labels_test(e_idx);
        start_idx = onset + delay_time + 1;
        end_idx = start_idx + window_size - 1;
        
        if end_idx <= length(eeg_sig_filtered)
            acc_epoch = acc_sig(:, start_idx:end_idx);
            acc_norm = normalized_acc(acc_epoch);
            heur_score = calculate_imu_heuristic(acc_norm);
            gate_passed = true; % heur_score >= 1.2; (Gate Disabled for offline evaluation)
            
            if gate_passed
                epoch = eeg_sig_filtered(start_idx:end_idx);
                epoch_bn = (epoch - mean(epoch(1:40))) ./ global_cap;
                feat_eeg = extract_features(epoch_bn, sys_order);
                feat_imu = extract_imu_features(acc_epoch);
                
                p_fall = predict_fall(feat_eeg, feat_imu, heur_score, mdl_eeg, mdl_imu, mdl_fusion);
                local_p_fall(e_idx) = p_fall;
                local_labels(e_idx) = orig_label;
                
                 if p_fall >= 0.20
                     pred_label = '1';
                 else
                     pred_label = '0';
                 end
            else
                pred_label = '0';
                local_p_fall(e_idx) = 0;
                local_labels(e_idx) = orig_label;
            end
            
            if orig_label == 1 && strcmp(pred_label, '1')
                local_TP(e_idx) = 1;
            elseif orig_label == 1 && strcmp(pred_label, '0')
                local_FN(e_idx) = 1;
            elseif (orig_label == 0 || orig_label == 2) && strcmp(pred_label, '0')
                local_TN(e_idx) = 1;
            elseif (orig_label == 0 || orig_label == 2) && strcmp(pred_label, '1')
                local_FP(e_idx) = 1;
                if orig_label == 0, local_FP0(e_idx) = 1;
                elseif orig_label == 2, local_FP2(e_idx) = 1; end
            end
            local_total(e_idx) = 1;
        end
    end
    
    TP = sum(local_TP); TN = sum(local_TN);
    FP = sum(local_FP); FN = sum(local_FN);
    FP0 = sum(local_FP0); FP2 = sum(local_FP2);
    tot = sum(local_total);
    p_falls = local_p_fall(local_total > 0);
    true_labels = local_labels(local_total > 0);
end

function [X_eeg, X_imu, X_heur, Y] = extract_subject_features(subj, raw_data_dir, b, a, global_cap)
    X_eeg = []; X_imu = []; X_heur = []; Y = {};
    
    load(fullfile(raw_data_dir, 'All34_table.mat'), 'event_table');
    load(fullfile(raw_data_dir, 'Label_Table.mat'), 'label_table');
    filtered_dir = fullfile(raw_data_dir, 'Filtered');
    my_edfs_dir = dir(fullfile(filtered_dir, '*edf'));
    edfs_names = {my_edfs_dir.name};
    
    if subj < 10, edf_idx = find(contains(edfs_names, ['0', num2str(subj)]));
    else, edf_idx = find(contains(edfs_names, num2str(subj))); end
    if isempty(edf_idx), return; end
    
    edf_name = fullfile(filtered_dir, edfs_names{edf_idx(1)});
    table_idx = find(event_table.Subject == subj);
    subj_events = event_table{table_idx, 3:end};
    subj_labels = label_table{table_idx, 3:end};
    
    valid_idx = ~isnan(subj_events);
    subj_events = subj_events(valid_idx);
    subj_labels = subj_labels(valid_idx);
    
    try
        EEG_whole = edfread(edf_name, 'SelectedSignals', 'R6', 'DataRecordOutputType', 'vector');
        eeg_sig = cat(1, EEG_whole.(1){:})';
        ACC_X = edfread(edf_name, 'SelectedSignals', 'x_dir', 'DataRecordOutputType', 'vector');
        ACC_Y = edfread(edf_name, 'SelectedSignals', 'y_dir', 'DataRecordOutputType', 'vector');
        ACC_Z = edfread(edf_name, 'SelectedSignals', 'z_dir', 'DataRecordOutputType', 'vector');
        acc_sig = [cat(1, ACC_X.(1){:})'; cat(1, ACC_Y.(1){:})'; cat(1, ACC_Z.(1){:})'] ./ 980;
    catch
        return;
    end
    
    eeg_sig_filtered = filtfilt(b, a, eeg_sig);
    
    num_class2 = length(subj_events);
    class2_onsets = []; attempts = 0;
    while length(class2_onsets) < num_class2 && attempts < 10000
        rand_onset = randi([1000, length(eeg_sig_filtered) - 2000]);
        if all(abs(subj_events - rand_onset) >= 2000) && all(abs(class2_onsets - rand_onset) >= 2000)
            class2_onsets = [class2_onsets, rand_onset];
        end
        attempts = attempts + 1;
    end
    
    all_onsets = [subj_events, class2_onsets];
    all_original_labels = [subj_labels, 2 * ones(1, length(class2_onsets))];
    
    delay_time = 80; window_size = 256; sys_order = 3;
    
    for e_idx = 1:length(all_onsets)
        onset = all_onsets(e_idx);
        orig_label = all_original_labels(e_idx);
        start_idx = onset + delay_time + 1;
        end_idx = start_idx + window_size - 1;
        
        if end_idx <= length(eeg_sig_filtered)
            epoch = eeg_sig_filtered(start_idx:end_idx);
            epoch_bn = (epoch - mean(epoch(1:40))) ./ global_cap;
            feat = extract_features(epoch_bn, sys_order);
            
            acc_epoch = acc_sig(:, start_idx:end_idx);
            acc_norm = normalized_acc(acc_epoch);
            
            imu_feat = extract_imu_features(acc_epoch);
            heur_score = calculate_imu_heuristic(acc_norm);
            
            if orig_label == 1
                Y{end+1, 1} = '1';
            else
                Y{end+1, 1} = '0';
            end
            X_eeg = [X_eeg; feat];
            X_imu = [X_imu; imu_feat];
            X_heur = [X_heur; heur_score];
        end
    end
end

% --- Continuous Live Evaluation Mode ---
function execute_continuous_evaluation(fig)
    ud = fig.UserData;
    results_ta = ud.results_text;
    results_ta.Value = {'Loading model... Please wait.'};
    drawnow;
    
    try
        load('trained_model.mat', 'mdl_eeg', 'mdl_imu', 'mdl_fusion', 'global_eeg_cap', 'b', 'a');
    catch
        results_ta.Value = {'ERROR: trained_model.mat not found.'};
        return;
    end
    
    val = ud.subj_dropdown.Value;
    if strcmp(val, 'All Subjects')
        target_subjs = ud.subjects_list;
    else
        target_subjs = sscanf(val, 'Subject %d');
    end
    
    if isempty(target_subjs), return; end
    
    % Adjustable tolerance window for True Positives
    tolerance_window_ms = 1500; 
    
    TP = 0; FP = 0; FN = 0;
    total_events = 0;
    
    wb = uiprogressdlg(fig, 'Title', 'Continuous Evaluation', 'Message', 'Processing subjects...');
    
    for i = 1:length(target_subjs)
        subj = target_subjs(i);
        wb.Value = i / length(target_subjs);
        wb.Message = sprintf('Evaluating Subject %d (%d/%d)', subj, i, length(target_subjs));
        
        [sub_TP, sub_FP, sub_FN, sub_tot] = continuous_evaluate_subject(subj, ud.raw_data_dir, b, a, global_eeg_cap, mdl_eeg, mdl_imu, mdl_fusion, tolerance_window_ms);
        
        TP = TP + sub_TP; 
        FP = FP + sub_FP; 
        FN = FN + sub_FN;
        total_events = total_events + sub_tot;
    end
    close(wb);
    
    Sens = TP / max(1, (TP + FN));
    Prec = TP / max(1, (TP + FP));
    F1 = 2 * (Sens * Prec) / max(1e-6, Sens + Prec);
    
    results_ta.Value = {
        '========================================='
        '      CONTINUOUS EVALUATION METRICS      '
        '========================================='
        sprintf(' Target            : %s', val)
        sprintf(' Tolerance Window  : +/- %d ms', tolerance_window_ms)
        sprintf(' Total Real Falls  : %d', total_events)
        '-----------------------------------------'
        sprintf(' True Positives (TP)  : %d', TP)
        sprintf(' False Positives (FP) : %d', FP)
        sprintf(' False Negatives (FN) : %d', FN)
        '-----------------------------------------'
        sprintf(' Sensitivity (Recall) : %.4f', Sens)
        sprintf(' Precision            : %.4f', Prec)
        sprintf(' F1 Score             : %.4f', F1)
        '========================================='
    };
end

function [TP, FP, FN, tot] = continuous_evaluate_subject(subj, raw_data_dir, b, a, global_cap, mdl_eeg, mdl_imu, mdl_fusion, tol_ms)
    TP = 0; FP = 0; FN = 0; tot = 0;
    
    load(fullfile(raw_data_dir, 'All34_table.mat'), 'event_table');
    load(fullfile(raw_data_dir, 'Label_Table.mat'), 'label_table');
    filtered_dir = fullfile(raw_data_dir, 'Filtered');
    my_edfs_dir = dir(fullfile(filtered_dir, '*edf'));
    edfs_names = {my_edfs_dir.name};
    
    if subj < 10
        edf_idx = find(contains(edfs_names, ['0', num2str(subj)]));
    else
        edf_idx = find(contains(edfs_names, num2str(subj)));
    end
    if isempty(edf_idx), return; end
    
    edf_name = fullfile(filtered_dir, edfs_names{edf_idx(1)});
    table_idx = find(event_table.Subject == subj);
    subj_events = event_table{table_idx, 3:end};
    subj_labels = label_table{table_idx, 3:end};
    
    valid_idx = ~isnan(subj_events);
    subj_events = subj_events(valid_idx);
    subj_labels = subj_labels(valid_idx);
    
    real_falls = subj_events(subj_labels == 1);
    tot = length(real_falls);
    
    try
        EEG_whole = edfread(edf_name, 'SelectedSignals', 'R6', 'DataRecordOutputType', 'vector');
        eeg_sig = cat(1, EEG_whole.(1){:})';
        ACC_X = edfread(edf_name, 'SelectedSignals', 'x_dir', 'DataRecordOutputType', 'vector');
        ACC_Y = edfread(edf_name, 'SelectedSignals', 'y_dir', 'DataRecordOutputType', 'vector');
        ACC_Z = edfread(edf_name, 'SelectedSignals', 'z_dir', 'DataRecordOutputType', 'vector');
        acc_sig = [cat(1, ACC_X.(1){:})'; cat(1, ACC_Y.(1){:})'; cat(1, ACC_Z.(1){:})'] ./ 980;
    catch
        return;
    end
    
    window_size = 256;
    step_size = 30;
    sys_order = 3;
    blindfold_samples = 1000;
    ms_since_last_detection = blindfold_samples;
    
    total_samples = length(eeg_sig);
    current_sample = 1;
    detections = [];
    
    while current_sample + window_size - 1 <= total_samples
        end_sample = current_sample + window_size - 1;
        eeg_window = eeg_sig(current_sample:end_sample);
        acc_window = acc_sig(:, current_sample:end_sample);
        
        [eeg_norm, acc_mag] = preprocess_signal(eeg_window', acc_window, b, a, global_cap);
        if size(eeg_norm, 1) > size(eeg_norm, 2), eeg_norm = eeg_norm'; end
        
        heur_score = calculate_imu_heuristic(acc_mag);
        ms_since_last_detection = ms_since_last_detection + step_size;
        
        if ms_since_last_detection >= blindfold_samples
            if heur_score >= 1.2
                p_fall = predict_fall_wrapper(eeg_norm, acc_window, heur_score, sys_order, mdl_eeg, mdl_imu, mdl_fusion);
                if p_fall >= 0.20
                    detections(end+1) = current_sample + round(window_size/2);
                    ms_since_last_detection = 0;
                end
            end
        end
        current_sample = current_sample + step_size;
    end
    
    % Score logic
    matched_falls = false(1, length(real_falls));
    
    for d = detections
        matched = false;
        for f_idx = 1:length(real_falls)
            if abs(d - real_falls(f_idx)) <= tol_ms
                matched_falls(f_idx) = true;
                matched = true;
                break;
            end
        end
        if ~matched
            FP = FP + 1;
        end
    end
    
    TP = sum(matched_falls);
    FN = length(real_falls) - TP;
end

function local_cap = calculate_subject_cap(subj, raw_data_dir, b, a)
    local_cap = 1.0;
    
    load(fullfile(raw_data_dir, 'All34_table.mat'), 'event_table');
    filtered_dir = fullfile(raw_data_dir, 'Filtered');
    my_edfs_dir = dir(fullfile(filtered_dir, '*edf'));
    edfs_names = {my_edfs_dir.name};
    
    if subj < 10, edf_idx = find(contains(edfs_names, ['0', num2str(subj)]));
    else, edf_idx = find(contains(edfs_names, num2str(subj))); end
    if isempty(edf_idx), return; end
    
    edf_name = fullfile(filtered_dir, edfs_names{edf_idx(1)});
    table_idx = find(event_table.Subject == subj);
    subj_events = event_table{table_idx, 3:end};
    subj_events = subj_events(~isnan(subj_events));
    
    try
        EEG_whole = edfread(edf_name, 'SelectedSignals', 'R6', 'DataRecordOutputType', 'vector');
        eeg_sig = cat(1, EEG_whole.(1){:})';
    catch
        return;
    end
    
    eeg_sig_filtered = filtfilt(b, a, eeg_sig);
    if ~isempty(subj_events)
        timing_1 = subj_events(1);
        if timing_1 + 1000 <= length(eeg_sig_filtered)
            per_1 = eeg_sig_filtered(timing_1 + 1 : timing_1 + 1000);
            local_cap = max(per_1 - mean(per_1(1:40)));
        end
    end
end
