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
    'Items', {'Evaluate Pre-trained', 'Train on Selection', 'Run Full LOSO'}, ...
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
    if strcmp(val, 'Evaluate Pre-trained')
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
        load('trained_model.mat', 'rf_model', 'acc_threshold', 'global_eeg_cap', 'b', 'a');
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
    
    wb = uiprogressdlg(fig, 'Title', 'Evaluating Metrics', 'Message', 'Processing subjects...');
    
    for i = 1:length(target_subjs)
        subj = target_subjs(i);
        wb.Value = i / length(target_subjs);
        wb.Message = sprintf('Evaluating Subject %d (%d/%d)', subj, i, length(target_subjs));
        
        [sub_TP, sub_TN, sub_FP, sub_FN, sub_FP0, sub_FP2, sub_tot] = evaluate_subject(subj, ud.raw_data_dir, b, a, global_eeg_cap, acc_threshold, rf_model);
        
        TP = TP + sub_TP; TN = TN + sub_TN;
        FP = FP + sub_FP; FN = FN + sub_FN;
        FP_Class0 = FP_Class0 + sub_FP0; FP_Class2 = FP_Class2 + sub_FP2;
        total_events = total_events + sub_tot;
    end
    close(wb);
    
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
    subj_X = cell(num_t, 1);
    subj_Y = cell(num_t, 1);
    subj_acc_max = cell(num_t, 1);
    subj_eeg_cap = zeros(num_t, 1);
    
    wb = uiprogressdlg(fig, 'Title', 'Training Model', 'Message', 'Extracting features (Parallel)...');
    
    parfor i = 1:num_t
        subj = target_subjs(i);
        [local_X, local_Y, local_acc_max, local_cap] = extract_subject_features(subj, ud.raw_data_dir, b, a);
        subj_X{i} = local_X;
        subj_Y{i} = local_Y;
        subj_acc_max{i} = local_acc_max;
        subj_eeg_cap(i) = local_cap;
    end
    
    wb.Message = 'Aggregating and Training RF...';
    wb.Value = 0.8;
    drawnow;
    
    X_train = cell2mat(subj_X);
    Y_train = vertcat(subj_Y{:});
    acc_max_unexpected = cell2mat(subj_acc_max');
    global_eeg_cap = max(subj_eeg_cap);
    
    if ~isempty(acc_max_unexpected)
        acc_threshold = min(acc_max_unexpected);
    else
        acc_threshold = 1.0;
    end
    
    template = templateTree('MaxNumSplits', size(X_train, 1) - 1);
    rf_model = fitcensemble(X_train, Y_train, 'Method', 'Bag', 'NumLearningCycles', 30, 'Learners', template, 'ClassNames', {'0', '1'});
    
    % Generate file name based on selected subjects
    if length(target_subjs) > 5
        filename = sprintf('trained_model_custom_%d_subjs.mat', length(target_subjs));
    else
        filename = ['trained_model_subj_', strjoin(arrayfun(@num2str, target_subjs, 'UniformOutput', false), '_'), '.mat'];
    end
    
    save(filename, 'rf_model', 'acc_threshold', 'global_eeg_cap', 'b', 'a');
    close(wb);
    
    results_ta.Value = {
        '========================================='
        '          TRAINING COMPLETE              '
        '========================================='
        sprintf(' Trained on %d Subjects.', length(target_subjs))
        sprintf(' Total Samples   : %d', size(X_train, 1))
        sprintf(' Global EEG Cap  : %.4f', global_eeg_cap)
        sprintf(' ACC Threshold   : %.4f', acc_threshold)
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
    
    wb = uiprogressdlg(fig, 'Title', 'LOSO Cross-Validation', 'Message', 'Extracting all features (Parallel)...');
    
    % Pre-extract all features to save massive time
    subj_X = cell(num_subjs, 1);
    subj_Y = cell(num_subjs, 1);
    subj_acc_max = cell(num_subjs, 1);
    subj_eeg_cap = zeros(num_subjs, 1);
    
    parfor i = 1:num_subjs
        [subj_X{i}, subj_Y{i}, subj_acc_max{i}, subj_eeg_cap(i)] = extract_subject_features(all_subjs(i), ud.raw_data_dir, b, a);
    end
    
    global_eeg_cap = max(subj_eeg_cap);
    
    TP = 0; TN = 0; FP = 0; FN = 0;
    FP_Class0 = 0; FP_Class2 = 0;
    total_events = 0;
    
    for test_idx = 1:num_subjs
        wb.Value = test_idx / num_subjs;
        wb.Message = sprintf('LOSO Iteration %d/%d (Held-out: Subj %d)', test_idx, num_subjs, all_subjs(test_idx));
        
        train_mask = true(num_subjs, 1);
        train_mask(test_idx) = false;
        
        X_train = cell2mat(subj_X(train_mask));
        Y_train = vertcat(subj_Y{train_mask});
        acc_max_train = cell2mat(subj_acc_max(train_mask)');
        
        if ~isempty(acc_max_train)
            acc_threshold = min(acc_max_train);
        else
            acc_threshold = 1.0;
        end
        
        template = templateTree('MaxNumSplits', size(X_train, 1) - 1);
        rf_model = fitcensemble(X_train, Y_train, 'Method', 'Bag', 'NumLearningCycles', 30, 'Learners', template, 'ClassNames', {'0', '1'});
        
        % Evaluate on the held out subject
        [sub_TP, sub_TN, sub_FP, sub_FN, sub_FP0, sub_FP2, sub_tot] = evaluate_subject(all_subjs(test_idx), ud.raw_data_dir, b, a, global_eeg_cap, acc_threshold, rf_model);
        
        TP = TP + sub_TP; TN = TN + sub_TN;
        FP = FP + sub_FP; FN = FN + sub_FN;
        FP_Class0 = FP_Class0 + sub_FP0; FP_Class2 = FP_Class2 + sub_FP2;
        total_events = total_events + sub_tot;
        
        results_ta.Value = [results_ta.Value; {sprintf('  Finished Subj %d', all_subjs(test_idx))}];
        drawnow;
    end
    
    close(wb);
    
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
function [TP, TN, FP, FN, FP0, FP2, tot] = evaluate_subject(subj, raw_data_dir, b, a, global_cap, acc_thresh, rf_model)
    TP = 0; TN = 0; FP = 0; FN = 0; FP0 = 0; FP2 = 0; tot = 0;
    
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
    
    delay_time = 80; window_size = 256; sys_order = 3;
    
    parfor e_idx = 1:num_events
        onset = all_onsets_test(e_idx);
        orig_label = all_original_labels_test(e_idx);
        start_idx = onset + delay_time + 1;
        end_idx = start_idx + window_size - 1;
        
        if end_idx <= length(eeg_sig_filtered)
            acc_epoch = acc_sig(:, start_idx:end_idx);
            acc_norm = normalized_acc(acc_epoch);
            [gate_passed, ~] = evaluate_acc_gate(acc_norm, acc_thresh);
            
            if gate_passed
                epoch = eeg_sig_filtered(start_idx:end_idx);
                epoch_bn = (epoch - mean(epoch(1:40))) ./ global_cap;
                feat = extract_features(epoch_bn, sys_order);
                pred_label = predict_fall(feat, rf_model);
            else
                pred_label = '0';
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
end

function [X, Y, acc_max, local_cap] = extract_subject_features(subj, raw_data_dir, b, a)
    X = []; Y = {}; acc_max = []; local_cap = 1.0;
    
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
    if ~isempty(subj_events)
        timing_1 = subj_events(1);
        if timing_1 + 1000 <= length(eeg_sig_filtered)
            per_1 = eeg_sig_filtered(timing_1 + 1 : timing_1 + 1000);
            local_cap = max(per_1 - mean(per_1(1:40)));
        end
    end
    
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
            epoch_bn = (epoch - mean(epoch(1:40))) ./ local_cap;
            feat = extract_features(epoch_bn, sys_order);
            
            acc_epoch = acc_sig(:, start_idx:end_idx);
            acc_norm = normalized_acc(acc_epoch);
            
            if orig_label == 1
                Y{end+1, 1} = '1';
                acc_max(end+1) = max(acc_norm);
            else
                Y{end+1, 1} = '0';
            end
            X = [X; feat];
        end
    end
end
