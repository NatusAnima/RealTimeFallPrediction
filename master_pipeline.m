% master_pipeline.m
% Unified MATLAB script to rigorously evaluate EEG-based fall detection 
% using N4SID state-space features, gated by accelerometer threshold.
%
% Architecture: Strict Leave-One-Subject-Out (LOSO) cross-validation.

clear all; close all; clc;

%% 1. Configuration & Setup
disp('--- 1. Configuration & Setup ---');
data_dir = pwd; % Assumes script is run from inside data folder
raw_data_dir = fullfile(data_dir, 'Raw_Data');
filtered_dir = fullfile(raw_data_dir, 'Filtered');

% Pipeline parameters
channel = 'R6';
fs = 1000; % 1000 Hz
window_size = 256; % 256 ms
delay_time = 80; % 80 ms post-onset
step_size = 30; % 30 ms sliding window step
sys_order = 3; % 3rd order for N4SID
blindfold_ms = 1000; % 1 second blindfold after fall detection

%% 2. Data Loading
disp('--- 2. Data Loading ---');
% Load event and label tables
all34_table_loc = fullfile(raw_data_dir, 'All34_table.mat');
label_table_loc = fullfile(raw_data_dir, 'Label_Table.mat');

if ~exist(all34_table_loc, 'file') || ~exist(label_table_loc, 'file')
    error('Cannot find All34_table.mat or Label_Table.mat. Ensure the script is running in the data folder.');
end

load(all34_table_loc, 'event_table');
load(label_table_loc, 'label_table');

% Locate all .edf files
where_my_edfs = fullfile(filtered_dir, '*edf');
my_edfs_dir = dir(where_my_edfs);
edfs_names = {my_edfs_dir.name};
if isempty(edfs_names)
    error('No .edf files found in Raw_Data/Filtered/');
end

% Get valid subject numbers from the table
subjects_list = event_table.Subject'; 
num_subjects = length(subjects_list);
disp(['Found ', num2str(num_subjects), ' subjects in the table.']);

%% 3. Global Preprocessing
disp('--- 3. Preprocessing Full Continuous Signals & Caps ---');

% Pre-allocate storage for each subject to avoid redundant I/O
subject_data = struct();

% Setup filter (2nd-order Butterworth bandpass 2.5 Hz - 30 Hz)
[b, a] = butter(2, [2.5, 30] / (fs/2), 'bandpass');

for i = 1:num_subjects
    subj = subjects_list(i);
    fprintf('Preprocessing Subject %d (%d/%d)... ', subj, i, num_subjects);
    
    % Find correct EDF
    if subj < 10
        edf_idx = find(contains(edfs_names, ['0', num2str(subj)]));
    else
        edf_idx = find(contains(edfs_names, num2str(subj)));
    end
    
    if isempty(edf_idx)
        fprintf('EDF file not found. Skipping.\n');
        continue;
    end
    edf_name = fullfile(filtered_dir, edfs_names{edf_idx(1)});
    
    % Get subject specific events and labels
    table_idx = find(event_table.Subject == subj);
    subj_events = event_table{table_idx, 3:end};
    subj_labels = label_table{table_idx, 3:end};
    
    % Remove NaN events
    valid_idx = ~isnan(subj_events);
    subj_events = subj_events(valid_idx);
    subj_labels = subj_labels(valid_idx);
    
    % Load EEG and ACC data
    try
        EEG_whole = edfread(edf_name, 'SelectedSignals', channel, 'DataRecordOutputType', 'vector');
        eeg_sig = cat(1, EEG_whole.(1){:})';
        
        ACC_X = edfread(edf_name, 'SelectedSignals', 'x_dir', 'DataRecordOutputType', 'vector');
        acc_sig_x = cat(1, ACC_X.(1){:})' ./ 980;
        
        ACC_Y = edfread(edf_name, 'SelectedSignals', 'y_dir', 'DataRecordOutputType', 'vector');
        acc_sig_y = cat(1, ACC_Y.(1){:})' ./ 980;
        
        ACC_Z = edfread(edf_name, 'SelectedSignals', 'z_dir', 'DataRecordOutputType', 'vector');
        acc_sig_z = cat(1, ACC_Z.(1){:})' ./ 980;
        
        acc_sig = [acc_sig_x; acc_sig_y; acc_sig_z];
    catch ME
        fprintf('Failed to load EDF data. Skipping.\n');
        continue;
    end
    
    % Apply bandpass filter
    eeg_sig_filtered = filtfilt(b, a, eeg_sig);
    
    % Baseline cap calculation (from the first event perturbation)
    timing_1 = subj_events(1);
    if timing_1 + 1000 <= length(eeg_sig_filtered)
        per_1 = eeg_sig_filtered(timing_1 + 1 : timing_1 + 1000);
        per_1b = per_1 - mean(per_1(1:40));
        eeg_cap = max(per_1b);
    else
        eeg_cap = 1; % Fallback normalization
    end
    
    % Store subject structures safely
    subject_data(i).subj = subj;
    subject_data(i).eeg_sig_filtered = eeg_sig_filtered;
    subject_data(i).acc_sig = acc_sig;
    subject_data(i).eeg_cap = eeg_cap;
    subject_data(i).events = subj_events;
    subject_data(i).labels = subj_labels;
    
    % Determine random "Silence/Normal Walking" baseline events (Class 2)
    num_class2 = length(subj_events);
    class2_onsets = [];
    min_dist = 2000;
    attempts = 0;
    while length(class2_onsets) < num_class2 && attempts < 10000
        rand_onset = randi([1000, length(eeg_sig_filtered) - 2000]);
        valid = true;
        for ev = subj_events
            if abs(rand_onset - ev) < min_dist
                valid = false;
                break;
            end
        end
        if valid
            for ev2 = class2_onsets
                if abs(rand_onset - ev2) < min_dist
                    valid = false;
                    break;
                end
            end
        end
        if valid
            class2_onsets = [class2_onsets, rand_onset];
        end
        attempts = attempts + 1;
    end
    
    subject_data(i).class2_onsets = class2_onsets;
    fprintf('Done.\n');
end

% Remove empty struct entries due to missing files
valid_subject_mask = arrayfun(@(x) ~isempty(x.subj), subject_data);
subject_data = subject_data(valid_subject_mask);
num_subjects = length(subject_data);

disp(['Successfully preprocessed ', num2str(num_subjects), ' subjects.']);

%% Helper Functions
extract_n4sid = @(epoch) extract_features(epoch, sys_order);

%% 4. Feature Extraction & LOSO Training (Offline)
disp('--- 4 & 5. LOSO Cross-Validation & Real-Time Simulation ---');

% Global Metrics Setup
global_TP = 0; global_TN = 0; global_FP = 0; global_FN = 0;
global_FP_Class0 = 0; global_FP_Class2 = 0;
total_events = 0;

% Real-Time metrics Setup
rt_global_FP = 0;

for test_idx = 1:num_subjects
    test_subj_data = subject_data(test_idx);
    train_subjects_data = subject_data([1:test_idx-1, test_idx+1:end]);
    
    % Subject-specific metric trackers
    subj_TP = 0; subj_TN = 0; subj_FP = 0; subj_FN = 0;
    subj_FP_Class0 = 0; subj_FP_Class2 = 0;
    
    fprintf('\n------------------------------------------------\n');
    fprintf('LOSO Iteration %d/%d (Held-out Subject %d)\n', test_idx, num_subjects, test_subj_data.subj);
    fprintf('------------------------------------------------\n');
    
    % --- TRAINING PHASE ---
    fprintf('  [1/3] Extracting Training Features... ');
    X_train = [];
    Y_train = {};
    acc_max_unexpected = []; % For threshold calculation
    
    for tr_idx = 1:length(train_subjects_data)
        tr_data = train_subjects_data(tr_idx);
        
        all_onsets = [tr_data.events, tr_data.class2_onsets];
        all_original_labels = [tr_data.labels, 2 * ones(1, length(tr_data.class2_onsets))];
        
        for e_idx = 1:length(all_onsets)
            onset = all_onsets(e_idx);
            orig_label = all_original_labels(e_idx);
            
            start_idx = onset + delay_time + 1;
            end_idx = start_idx + window_size - 1;
            
            if end_idx <= length(tr_data.eeg_sig_filtered)
                % EEG Feature Extraction
                epoch = tr_data.eeg_sig_filtered(start_idx:end_idx);
                epoch_bn = (epoch - mean(epoch(1:40))) ./ tr_data.eeg_cap;
                feat = extract_n4sid(epoch_bn);
                
                % ACC Threshold calculation
                acc_epoch = tr_data.acc_sig(:, start_idx:end_idx);
                acc_norm = normalized_acc(acc_epoch);
                max_acc_val = max(acc_norm);
                
                if orig_label == 1
                    mapped_label = '1';
                    acc_max_unexpected(end+1) = max_acc_val;
                else
                    mapped_label = '0';
                end
                
                X_train = [X_train; feat];
                Y_train{end+1, 1} = mapped_label;
            end
        end
    end
    
    % Calculate the Accelerometer Gate Threshold
    if ~isempty(acc_max_unexpected)
        acc_threshold = min(acc_max_unexpected);
    else
        acc_threshold = 0; % Fallback
    end
    fprintf('Done (ACC Threshold: %.4f)\n', acc_threshold);
    
    fprintf('  [2/3] Training fitcensemble... ');
    % Using fitcensemble with 30 Learning Cycles to match original trainer_class.m
    template = templateTree('MaxNumSplits', size(X_train, 1) - 1);
    rf_model = fitcensemble(X_train, Y_train, 'Method', 'Bag', 'NumLearningCycles', 30, 'Learners', template, 'ClassNames', {'0', '1'});
    fprintf('Done\n');
    
    % --- OFFLINE EVALUATION PHASE ---
    fprintf('  [3/3] Offline Evaluation & Real-Time Simulation...\n');
    all_onsets_test = [test_subj_data.events, test_subj_data.class2_onsets];
    all_original_labels_test = [test_subj_data.labels, 2 * ones(1, length(test_subj_data.class2_onsets))];
    
    for e_idx = 1:length(all_onsets_test)
        onset = all_onsets_test(e_idx);
        orig_label = all_original_labels_test(e_idx);
        
        start_idx = onset + delay_time + 1;
        end_idx = start_idx + window_size - 1;
        
        if end_idx <= length(test_subj_data.eeg_sig_filtered)
            % ACC Gate Check
            acc_epoch = test_subj_data.acc_sig(:, start_idx:end_idx);
            acc_norm = normalized_acc(acc_epoch);
            if max(acc_norm) >= acc_threshold
                % EEG Prediction
                epoch = test_subj_data.eeg_sig_filtered(start_idx:end_idx);
                epoch_bn = (epoch - mean(epoch(1:40))) ./ test_subj_data.eeg_cap;
                
                feat = extract_n4sid(epoch_bn);
                pred_cell = predict(rf_model, feat);
                pred_label = pred_cell{1};
            else
                pred_label = '0'; % Gated out by accelerometer
            end
            
            % Update per-subject and global evaluation metrics
            if orig_label == 1 && strcmp(pred_label, '1')
                global_TP = global_TP + 1; subj_TP = subj_TP + 1;
            elseif orig_label == 1 && strcmp(pred_label, '0')
                global_FN = global_FN + 1; subj_FN = subj_FN + 1;
            elseif (orig_label == 0 || orig_label == 2) && strcmp(pred_label, '0')
                global_TN = global_TN + 1; subj_TN = subj_TN + 1;
            elseif (orig_label == 0 || orig_label == 2) && strcmp(pred_label, '1')
                global_FP = global_FP + 1; subj_FP = subj_FP + 1;
                % Breakdown False Positives
                if orig_label == 0
                    global_FP_Class0 = global_FP_Class0 + 1; subj_FP_Class0 = subj_FP_Class0 + 1;
                elseif orig_label == 2
                    global_FP_Class2 = global_FP_Class2 + 1; subj_FP_Class2 = subj_FP_Class2 + 1;
                end
            end
            total_events = total_events + 1;
        end
    end
    
    % --- REAL-TIME CONTINUOUS SIMULATION ---
    sig_len = length(test_subj_data.eeg_sig_filtered);
    num_windows = floor((sig_len - window_size) / step_size);
    
    rt_subject_FP = 0;
    
    % Mask purely silent zones as everything outside of [onset-1000, onset+2000]
    is_silence = true(1, sig_len);
    for e_idx = 1:length(test_subj_data.events)
        onset = test_subj_data.events(e_idx);
        start_sil_remove = max(1, onset - 1000);
        end_sil_remove = min(sig_len, onset + 2000);
        is_silence(start_sil_remove:end_sil_remove) = false;
    end
    
    rt_windows_total = 0;
    rt_windows_baseline = 0;
    blindfold_until = 0;
    
    fprintf('        Simulating Real-Time Feed: [');
    prog_chars = 0;
    
    for w = 1:num_windows
        start_idx = 1 + (w - 1) * step_size;
        end_idx = start_idx + window_size - 1;
        
        rt_windows_total = rt_windows_total + 1;
        
        % Print progress bar (50 characters wide)
        progress = w / num_windows;
        desired_chars = floor(progress * 50);
        if desired_chars > prog_chars
            fprintf(repmat('=', 1, desired_chars - prog_chars));
            prog_chars = desired_chars;
        end
        
        % Check 1-second Blindfold
        if start_idx < blindfold_until
            continue; 
        end
        
        if all(is_silence(start_idx:end_idx))
            rt_windows_baseline = rt_windows_baseline + 1;
            
            % ACC Gate Check
            acc_epoch = test_subj_data.acc_sig(:, start_idx:end_idx);
            acc_norm = normalized_acc(acc_epoch);
            if max(acc_norm) >= acc_threshold
                % EEG Feature Extraction & Prediction
                epoch = test_subj_data.eeg_sig_filtered(start_idx:end_idx);
                epoch_bn = (epoch - mean(epoch(1:40))) ./ test_subj_data.eeg_cap;
                
                feat = extract_n4sid(epoch_bn);
                pred_cell = predict(rf_model, feat);
                pred_label = pred_cell{1};
                
                if strcmp(pred_label, '1')
                    rt_subject_FP = rt_subject_FP + 1;
                    rt_global_FP = rt_global_FP + 1;
                    % Activate 1-second blindfold
                    blindfold_until = start_idx + blindfold_ms;
                end
            end
        end
    end
    fprintf(']\n');
    
    % --- PRINT SUBJECT-SPECIFIC RESULTS ---
    subj_Sens = subj_TP / max(1, (subj_TP + subj_FN));
    subj_Spec = subj_TN / max(1, (subj_TN + subj_FP));
    subj_BAcc = (subj_Sens + subj_Spec) / 2;
    
    fprintf('        --> Subject Offline Results: TP=%d, TN=%d, FP=%d, FN=%d | BAcc: %.4f\n', ...
        subj_TP, subj_TN, subj_FP, subj_FN, subj_BAcc);
    fprintf('        --> Offline FP Breakdown   : FP_Class0=%d | FP_Class2=%d\n', ...
        subj_FP_Class0, subj_FP_Class2);
    fprintf('        --> Real-Time Baseline FPs : %d (Tested on %d pure silence windows)\n', ...
        rt_subject_FP, rt_windows_baseline);
end

%% 6. Final Offline Metrics Display
disp('======================================================');
disp('FINAL OFFLINE LOSO EVALUATION METRICS');
disp('======================================================');

Sensitivity = global_TP / max(1, (global_TP + global_FN));
Specificity = global_TN / max(1, (global_TN + global_FP));
Balanced_Accuracy = (Sensitivity + Specificity) / 2;

fprintf('Total Events Evaluated : %d\n', total_events);
fprintf('True Positives (TP)    : %d\n', global_TP);
fprintf('True Negatives (TN)    : %d\n', global_TN);
fprintf('False Positives (FP)   : %d\n', global_FP);
fprintf('False Negatives (FN)   : %d\n\n', global_FN);

fprintf('Sensitivity (Recall)   : %.4f\n', Sensitivity);
fprintf('Specificity            : %.4f\n', Specificity);
fprintf('Balanced Accuracy      : %.4f\n\n', Balanced_Accuracy);

disp('--- FALSE POSITIVE BREAKDOWN ---');
fprintf('FP_Class0 (Predicted misclassified as Real) : %d\n', global_FP_Class0);
fprintf('FP_Class2 (Silence misclassified as Real)   : %d\n\n', global_FP_Class2);

disp('--- REAL-TIME SIMULATION METRICS ---');
fprintf('Continuous False Positives during Baseline : %d\n', rt_global_FP);
disp('======================================================');

if global_FP_Class0 > 0
    fprintf('\n*** WARNING ***\n');
    fprintf('Algorithm is misclassifying predicted falls as real falls.\n');
    fprintf('N4SID is failing to isolate unpredicted panic, and is merely acting as a motion detector.\n');
else
    fprintf('\n*** SUCCESS ***\n');
    fprintf('Zero FP_Class0 detected. Algorithm successfully differentiates predicted vs unpredicted falls.\n');
end

%% Helper Functions

function my_feature = extract_features(Data, sys_order)
    % Extracts n4sid feature mapping
    % Ts = 1 ms sampling time
    dy_sys = iddata(Data(:), [], 1, 'TimeUnit', 'milliseconds', 'Tstart', 0);
    sys_ss = n4sid(dy_sys, sys_order, 'Display', 'off');
    
    sysA = [];
    for k = 1:sys_order
        sysA = [sysA, sys_ss.A(k, :)];
    end
    
    my_feature = [sysA, sys_ss.C, sys_ss.K'];
end

function acc_epoch_norm = normalized_acc(acc_epoch)
    % Normalized the acc signal
    acc_epoch_x = sgolayfilt(acc_epoch(1,:), 3, 21);
    acc_epoch_y = sgolayfilt(acc_epoch(2,:), 3, 21);
    acc_epoch_z = sgolayfilt(acc_epoch(3,:), 3, 21);
    acc_epoch_norm = sqrt(acc_epoch_x.^2 + acc_epoch_y.^2 + acc_epoch_z.^2);
end
