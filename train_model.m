% train_model.m
% Script to generate and save a trained model based on all available data.
% Uses Parallel Computing Toolbox (parfor) to significantly speed up offline training.

clear all; close all; clc;

disp('--- 1. Configuration & Setup ---');
data_dir = pwd; 
raw_data_dir = fullfile(data_dir, 'Raw_Data');
filtered_dir = fullfile(raw_data_dir, 'Filtered');

% Pipeline parameters
channel = 'R6';
fs = 1000; 
window_size = 256; 
delay_time = 80; 
sys_order = 3; 

disp('--- 2. Data Loading ---');
all34_table_loc = fullfile(raw_data_dir, 'All34_table.mat');
label_table_loc = fullfile(raw_data_dir, 'Label_Table.mat');

if ~exist(all34_table_loc, 'file') || ~exist(label_table_loc, 'file')
    error('Cannot find All34_table.mat or Label_Table.mat. Ensure the script is running in the project root.');
end

load(all34_table_loc, 'event_table');
load(label_table_loc, 'label_table');

my_edfs_dir = dir(fullfile(filtered_dir, '*edf'));
edfs_names = {my_edfs_dir.name};
subjects_list = event_table.Subject'; 
num_subjects = length(subjects_list);

disp('--- 3. Preprocessing Full Data (Pass 1: Global Cap Calculation) ---');
[b, a] = butter(2, [2.5, 30] / (fs/2), 'bandpass');

subj_eeg_cap = ones(num_subjects, 1);

parfor i = 1:num_subjects
    subj = subjects_list(i);
    
    if subj < 10, edf_idx = find(contains(edfs_names, ['0', num2str(subj)]));
    else, edf_idx = find(contains(edfs_names, num2str(subj))); end
    
    if isempty(edf_idx), continue; end
    edf_name = fullfile(filtered_dir, edfs_names{edf_idx(1)});
    
    table_idx = find(event_table.Subject == subj);
    subj_events = event_table{table_idx, 3:end};
    subj_events = subj_events(~isnan(subj_events));
    
    try
        EEG_whole = edfread(edf_name, 'SelectedSignals', channel, 'DataRecordOutputType', 'vector');
        eeg_sig = cat(1, EEG_whole.(1){:})';
    catch ME
        continue;
    end
    
    eeg_sig_filtered = filtfilt(b, a, eeg_sig);
    
    if ~isempty(subj_events)
        timing_1 = subj_events(1);
        if timing_1 + 1000 <= length(eeg_sig_filtered)
            per_1 = eeg_sig_filtered(timing_1 + 1 : timing_1 + 1000);
            subj_eeg_cap(i) = max(per_1 - mean(per_1(1:40)));
        end
    end
end

global_eeg_cap = max(subj_eeg_cap);
fprintf('Global EEG Cap calculated: %.4f\n', global_eeg_cap);

disp('--- 4. Preprocessing Full Data (Pass 2: Feature Extraction) ---');
subj_X = cell(num_subjects, 1);
subj_X_imu = cell(num_subjects, 1);
subj_X_heur = cell(num_subjects, 1);
subj_Y = cell(num_subjects, 1);
subj_idx_cell = cell(num_subjects, 1);

parfor i = 1:num_subjects
    subj = subjects_list(i);
    fprintf('Processing Subject %d (%d/%d)...\n', subj, i, num_subjects);
    
    if subj < 10, edf_idx = find(contains(edfs_names, ['0', num2str(subj)]));
    else, edf_idx = find(contains(edfs_names, num2str(subj))); end
    
    if isempty(edf_idx), continue; end
    edf_name = fullfile(filtered_dir, edfs_names{edf_idx(1)});
    
    table_idx = find(event_table.Subject == subj);
    subj_events = event_table{table_idx, 3:end};
    subj_labels = label_table{table_idx, 3:end};
    
    valid_idx = ~isnan(subj_events);
    subj_events = subj_events(valid_idx);
    subj_labels = subj_labels(valid_idx);
    
    try
        EEG_whole = edfread(edf_name, 'SelectedSignals', channel, 'DataRecordOutputType', 'vector');
        eeg_sig = cat(1, EEG_whole.(1){:})';
        
        ACC_X = edfread(edf_name, 'SelectedSignals', 'x_dir', 'DataRecordOutputType', 'vector');
        ACC_Y = edfread(edf_name, 'SelectedSignals', 'y_dir', 'DataRecordOutputType', 'vector');
        ACC_Z = edfread(edf_name, 'SelectedSignals', 'z_dir', 'DataRecordOutputType', 'vector');
        acc_sig = [cat(1, ACC_X.(1){:})'; cat(1, ACC_Y.(1){:})'; cat(1, ACC_Z.(1){:})'] ./ 980;
    catch ME
        continue;
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
    
    local_X = []; local_X_imu = []; local_X_heur = []; local_Y = {};
    
    for e_idx = 1:length(all_onsets)
        onset = all_onsets(e_idx);
        orig_label = all_original_labels(e_idx);
        start_idx = onset + delay_time + 1;
        end_idx = start_idx + window_size - 1;
        
        if end_idx <= length(eeg_sig_filtered)
            epoch = eeg_sig_filtered(start_idx:end_idx);
            epoch_bn = (epoch - mean(epoch(1:40))) ./ global_eeg_cap;
            feat = extract_features(epoch_bn, sys_order);
            
            acc_epoch = acc_sig(:, start_idx:end_idx);
            acc_norm = normalized_acc(acc_epoch);
            
            imu_feat = extract_imu_features(acc_epoch);
            heur_score = calculate_imu_heuristic(acc_norm);
            
            if orig_label == 1
                local_Y{end+1, 1} = '1';
            else
                local_Y{end+1, 1} = '0';
            end
            local_X = [local_X; feat];
            local_X_imu = [local_X_imu; imu_feat];
            local_X_heur = [local_X_heur; heur_score];
        end
    end
    
    subj_X{i} = local_X;
    subj_X_imu{i} = local_X_imu;
    subj_X_heur{i} = local_X_heur;
    subj_Y{i} = local_Y;
    subj_idx_cell{i} = i * ones(size(local_Y, 1), 1);
end

disp('--- 5. Aggregating Parallel Results ---');
X_eeg = cell2mat(subj_X);
X_imu = cell2mat(subj_X_imu);
X_heur = cell2mat(subj_X_heur);
Y_train = vertcat(subj_Y{:});
all_subj_idx = cell2mat(subj_idx_cell);

disp('--- 6. Model Training (Subject-Wise OOF Late Fusion) ---');
fprintf('Total samples: %d\n', size(X_eeg, 1));

% Calculate observation weights
obs_weights = ones(size(Y_train, 1), 1);
idx_class1 = strcmp(Y_train, '1');
obs_weights(idx_class1) = 5.0; % 5x weight for minority 'Real Fall' class

K = 5;
if num_subjects < K, K = num_subjects; end
cv_subj = cvpartition(num_subjects, 'KFold', K);

p_eeg_oof = zeros(size(Y_train, 1), 1);
p_imu_oof = zeros(size(Y_train, 1), 1);

fprintf('Generating OOF Predictions using Subject-Wise %d-Fold CV...\n', K);
for k = 1:K
    test_subjs_mask = test(cv_subj, k);
    train_subjs_mask = training(cv_subj, k);
    
    test_idx = ismember(all_subj_idx, find(test_subjs_mask));
    train_idx = ismember(all_subj_idx, find(train_subjs_mask));
    
    template_eeg = templateTree('MinLeafSize', 5);
    mdl_eeg_fold = fitcensemble(X_eeg(train_idx, :), Y_train(train_idx), 'Method', 'Bag', 'NumLearningCycles', 100, 'Learners', template_eeg, 'ClassNames', {'0', '1'}, 'Weights', obs_weights(train_idx));
    
    template_imu = templateTree('MinLeafSize', 5);
    mdl_imu_fold = fitcensemble(X_imu(train_idx, :), Y_train(train_idx), 'Method', 'Bag', 'NumLearningCycles', 100, 'Learners', template_imu, 'ClassNames', {'0', '1'}, 'Weights', obs_weights(train_idx));
    
    [~, score_eeg] = predict(mdl_eeg_fold, X_eeg(test_idx, :));
    [~, score_imu] = predict(mdl_imu_fold, X_imu(test_idx, :));
    
    idx_1_eeg = find(strcmp(mdl_eeg_fold.ClassNames, '1'));
    idx_1_imu = find(strcmp(mdl_imu_fold.ClassNames, '1'));
    
    p_eeg_oof(test_idx) = score_eeg(:, idx_1_eeg);
    p_imu_oof(test_idx) = score_imu(:, idx_1_imu);
end

fprintf('Training final Random Forests on full dataset...\n');
template_eeg = templateTree('MinLeafSize', 5);
mdl_eeg = fitcensemble(X_eeg, Y_train, 'Method', 'Bag', 'NumLearningCycles', 100, 'Learners', template_eeg, 'ClassNames', {'0', '1'}, 'Weights', obs_weights);

template_imu = templateTree('MinLeafSize', 5);
mdl_imu = fitcensemble(X_imu, Y_train, 'Method', 'Bag', 'NumLearningCycles', 100, 'Learners', template_imu, 'ClassNames', {'0', '1'}, 'Weights', obs_weights);

fprintf('Training Fusion Logistic Regression...\n');
X_fusion_train = [p_eeg_oof, p_imu_oof, X_heur];
Y_train_cat = categorical(Y_train);
mdl_fusion = fitglm(X_fusion_train, Y_train_cat == '1', 'Distribution', 'binomial', 'Weights', obs_weights);

disp('Training Complete. Saving models to trained_model.mat...');
save('trained_model.mat', 'mdl_eeg', 'mdl_imu', 'mdl_fusion', 'global_eeg_cap', 'b', 'a');
disp('Model saved successfully!');
