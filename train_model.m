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

disp('--- 3. Preprocessing Full Data (Parallel) ---');
[b, a] = butter(2, [2.5, 30] / (fs/2), 'bandpass');

% Pre-allocate cell arrays for parfor collection
subj_X = cell(num_subjects, 1);
subj_Y = cell(num_subjects, 1);
subj_acc_max = cell(num_subjects, 1);
subj_eeg_cap = zeros(num_subjects, 1);

parfor i = 1:num_subjects
    subj = subjects_list(i);
    fprintf('Processing Subject %d (%d/%d)...\n', subj, i, num_subjects);
    
    if subj < 10
        edf_idx = find(contains(edfs_names, ['0', num2str(subj)]));
    else
        edf_idx = find(contains(edfs_names, num2str(subj)));
    end
    
    if isempty(edf_idx)
        fprintf('Missing EDF for Subject %d. Skipping.\n', subj); 
        continue; 
    end
    edf_name = fullfile(filtered_dir, edfs_names{edf_idx(1)});
    
    % Get subject specific events and labels
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
        fprintf('Failed to load Subject %d. Skipping.\n', subj);
        continue;
    end
    
    eeg_sig_filtered = filtfilt(b, a, eeg_sig);
    
    % Baseline cap calculation
    local_eeg_cap = 1; % Default fallback
    if ~isempty(subj_events)
        timing_1 = subj_events(1);
        if timing_1 + 1000 <= length(eeg_sig_filtered)
            per_1 = eeg_sig_filtered(timing_1 + 1 : timing_1 + 1000);
            local_eeg_cap = max(per_1 - mean(per_1(1:40)));
        end
    end
    
    % Generate Silence (Class 2) onsets
    num_class2 = length(subj_events);
    class2_onsets = [];
    attempts = 0;
    while length(class2_onsets) < num_class2 && attempts < 10000
        rand_onset = randi([1000, length(eeg_sig_filtered) - 2000]);
        if all(abs(subj_events - rand_onset) >= 2000) && all(abs(class2_onsets - rand_onset) >= 2000)
            class2_onsets = [class2_onsets, rand_onset];
        end
        attempts = attempts + 1;
    end
    
    % Extract Training Features
    all_onsets = [subj_events, class2_onsets];
    all_original_labels = [subj_labels, 2 * ones(1, length(class2_onsets))];
    
    local_X = [];
    local_Y = {};
    local_acc_max = [];
    
    for e_idx = 1:length(all_onsets)
        onset = all_onsets(e_idx);
        orig_label = all_original_labels(e_idx);
        start_idx = onset + delay_time + 1;
        end_idx = start_idx + window_size - 1;
        
        if end_idx <= length(eeg_sig_filtered)
            epoch = eeg_sig_filtered(start_idx:end_idx);
            epoch_bn = (epoch - mean(epoch(1:40))) ./ local_eeg_cap;
            feat = extract_features(epoch_bn, sys_order);
            
            acc_epoch = acc_sig(:, start_idx:end_idx);
            acc_norm = normalized_acc(acc_epoch);
            max_acc_val = max(acc_norm);
            
            if orig_label == 1
                local_Y{end+1, 1} = '1';
                local_acc_max(end+1) = max_acc_val;
            else
                local_Y{end+1, 1} = '0';
            end
            local_X = [local_X; feat];
        end
    end
    
    % Store in cell arrays for parfor transparency
    subj_X{i} = local_X;
    subj_Y{i} = local_Y;
    subj_acc_max{i} = local_acc_max;
    subj_eeg_cap(i) = local_eeg_cap;
end

disp('--- 4. Aggregating Parallel Results ---');
X_train = cell2mat(subj_X);
Y_train = vertcat(subj_Y{:});
acc_max_unexpected = cell2mat(subj_acc_max');
global_eeg_cap = max(subj_eeg_cap);

disp('--- 5. Model Training ---');
if ~isempty(acc_max_unexpected)
    acc_threshold = min(acc_max_unexpected);
else
    acc_threshold = 1.0; % Fallback
end

fprintf('ACC Gate Threshold calculated: %.4f\n', acc_threshold);
fprintf('Global EEG Cap calculated: %.4f\n', global_eeg_cap);
fprintf('Training fitcensemble on %d total samples...\n', size(X_train, 1));

template = templateTree('MaxNumSplits', size(X_train, 1) - 1);
rf_model = fitcensemble(X_train, Y_train, 'Method', 'Bag', 'NumLearningCycles', 30, 'Learners', template, 'ClassNames', {'0', '1'});

disp('Training Complete. Saving model to trained_model.mat...');
save('trained_model.mat', 'rf_model', 'acc_threshold', 'global_eeg_cap', 'b', 'a');
disp('Model saved successfully!');
