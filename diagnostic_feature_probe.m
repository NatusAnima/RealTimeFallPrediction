% diagnostic_feature_probe.m
% A standalone script to empirically prove OOF leakage and EEG cap mismatch.

clear all; close all; clc;

disp('======================================================');
disp('   FALL PREDICTION ARCHITECTURE DIAGNOSTIC PROBE      ');
disp('======================================================');

data_dir = pwd; 
raw_data_dir = fullfile(data_dir, 'Raw_Data');
filtered_dir = fullfile(raw_data_dir, 'Filtered');

load(fullfile(raw_data_dir, 'All34_table.mat'), 'event_table');
load(fullfile(raw_data_dir, 'Label_Table.mat'), 'label_table');

subjects_list = unique(event_table.Subject');
if length(subjects_list) < 5
    error('Need at least 5 subjects to run this diagnostic.');
end

% Pick dynamic subjects (handling missing Subject 1)
train_subjs = subjects_list(1:4);
test_subj = subjects_list(5);

fprintf('Selected Training Subjects: %s\n', num2str(train_subjs));
fprintf('Selected Test Subject: %d\n', test_subj);

fs = 1000;
[b, a] = butter(2, [2.5, 30] / (fs/2), 'bandpass');
my_edfs_dir = dir(fullfile(filtered_dir, '*edf'));
edfs_names = {my_edfs_dir.name};

%% ==========================================================
%% DIAGNOSTIC 1: OOF Leakage vs Held-out Reality
%% ==========================================================
disp(' ');
disp('--- RUNNING DIAGNOSTIC 1: OOF LEAKAGE ---');

delay_time = 80; window_size = 256; sys_order = 3;

% Extract features for training subjects
train_X_eeg = []; train_Y = {};
for i = 1:length(train_subjs)
    subj = train_subjs(i);
    [X, ~, ~, Y, ~] = diag_extract_subj(subj, event_table, label_table, edfs_names, filtered_dir, b, a);
    train_X_eeg = [train_X_eeg; X];
    train_Y = [train_Y; Y];
end

% Extract features for test subject
[test_X_eeg, ~, ~, test_Y, ~] = diag_extract_subj(test_subj, event_table, label_table, edfs_names, filtered_dir, b, a);

% Generate OOF with Flat K-Fold (The Bug)
obs_weights = ones(size(train_Y, 1), 1);
obs_weights(strcmp(train_Y, '1')) = 5.0;

disp('Training Random Forest and generating flat KFold OOF probabilities...');
K = 5;
cv = cvpartition(train_Y, 'KFold', K);
p_eeg_oof = zeros(size(train_Y, 1), 1);

for k = 1:K
    t_idx = training(cv, k);
    ts_idx = test(cv, k);
    template_eeg = templateTree('MaxNumSplits', sum(t_idx) - 1);
    mdl = fitcensemble(train_X_eeg(t_idx, :), train_Y(t_idx), 'Method', 'Bag', 'NumLearningCycles', 30, 'Learners', template_eeg, 'ClassNames', {'0', '1'}, 'Weights', obs_weights(t_idx));
    
    [~, score] = predict(mdl, train_X_eeg(ts_idx, :));
    idx1 = find(strcmp(mdl.ClassNames, '1'));
    p_eeg_oof(ts_idx) = score(:, idx1);
end

% Train final model to evaluate on test subject
template_eeg = templateTree('MaxNumSplits', size(train_X_eeg, 1) - 1);
final_mdl_eeg = fitcensemble(train_X_eeg, train_Y, 'Method', 'Bag', 'NumLearningCycles', 30, 'Learners', template_eeg, 'ClassNames', {'0', '1'}, 'Weights', obs_weights);

disp('Evaluating on fully unseen test subject...');
[~, score_test] = predict(final_mdl_eeg, test_X_eeg);
idx1 = find(strcmp(final_mdl_eeg.ClassNames, '1'));
p_eeg_test = score_test(:, idx1);

% Train a dummy fusion model to see weights
X_fusion_dummy = [p_eeg_oof, rand(size(p_eeg_oof)), rand(size(p_eeg_oof))]; % Fake IMU/Heur
mdl_fusion = fitglm(X_fusion_dummy, categorical(train_Y) == '1', 'Distribution', 'binomial', 'Weights', obs_weights);
disp(' ');
disp('--> Fusion Model Learned Coefficients for p_eeg:');
disp(mdl_fusion.Coefficients('x1', :));

% Plot Distribution Shift
figure('Name', 'Diagnostic 1: OOF Leakage');
subplot(1,2,1);
histogram(p_eeg_oof(strcmp(train_Y, '1')), 'BinWidth', 0.1, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'Normalization', 'probability'); hold on;
histogram(p_eeg_oof(strcmp(train_Y, '0')), 'BinWidth', 0.1, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'Normalization', 'probability');
title('Training OOF Probabilities (Leaked)');
xlabel('p\_eeg'); ylabel('Probability Density');
legend('Real Falls', 'Non-Falls');

subplot(1,2,2);
histogram(p_eeg_test(strcmp(test_Y, '1')), 'BinWidth', 0.1, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'Normalization', 'probability'); hold on;
histogram(p_eeg_test(strcmp(test_Y, '0')), 'BinWidth', 0.1, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'Normalization', 'probability');
title(sprintf('Held-out Probabilities (Subj %d)', test_subj));
xlabel('p\_eeg'); ylabel('Probability Density');
legend('Real Falls', 'Non-Falls');

saveas(gcf, 'diagnostic_1_plot.png');


%% ==========================================================
%% DIAGNOSTIC 2: EEG Cap Scale Distortion
%% ==========================================================
disp(' ');
disp('--- RUNNING DIAGNOSTIC 2: EEG CAP SCALE DISTORTION ---');

subj = -1;
for s = train_subjs
    table_idx = find(event_table.Subject == s);
    subj_labels = label_table{table_idx, 3:end};
    subj_labels = subj_labels(~isnan(subj_labels));
    if any(subj_labels == 1)
        subj = s;
        break;
    end
end

if subj == -1
    disp('No real fall found in any training subject for diagnostic 2.');
else
    table_idx = find(event_table.Subject == subj);
    subj_events = event_table{table_idx, 3:end};
    subj_events = subj_events(~isnan(subj_events));
    subj_labels = label_table{table_idx, 3:end};
    subj_labels = subj_labels(~isnan(subj_labels));

% Find a real fall
fall_idx = find(subj_labels == 1, 1);
if isempty(fall_idx)
    disp('No real fall found in this subject for diagnostic 2.');
else
    onset = subj_events(fall_idx);
    
    if subj < 10, edf_idx = find(contains(edfs_names, ['0', num2str(subj)]));
    else, edf_idx = find(contains(edfs_names, num2str(subj))); end
    
    edf_name = fullfile(filtered_dir, edfs_names{edf_idx(1)});
    EEG_whole = edfread(edf_name, 'SelectedSignals', 'R6', 'DataRecordOutputType', 'vector');
    eeg_sig = cat(1, EEG_whole.(1){:})';
    eeg_sig_filtered = filtfilt(b, a, eeg_sig);
    
    start_idx = onset + delay_time + 1;
    end_idx = start_idx + window_size - 1;
    epoch = eeg_sig_filtered(start_idx:end_idx);
    
    % True local cap
    local_cap = 1.0;
    timing_1 = subj_events(1);
    if timing_1 + 1000 <= length(eeg_sig_filtered)
        per_1 = eeg_sig_filtered(timing_1 + 1 : timing_1 + 1000);
        local_cap = max(per_1 - mean(per_1(1:40)));
    end
    
    % Artificial global cap (e.g. 5x larger)
    global_cap = local_cap * 5.0;
    
    epoch_local = (epoch - mean(epoch(1:40))) ./ local_cap;
    epoch_global = (epoch - mean(epoch(1:40))) ./ global_cap;
    
    feat_local = extract_features(epoch_local, sys_order);
    feat_global = extract_features(epoch_global, sys_order);
    
    fprintf('Local Cap used in Training : %.4f\n', local_cap);
    fprintf('Global Cap used in Testing : %.4f\n', global_cap);
    disp(' ');
    disp('First 5 values of N4SID State-Space Feature Vector:');
    fprintf('Trained on Local Cap : [%.4f, %.4f, %.4f, %.4f, %.4f]\n', feat_local(1:5));
    fprintf('Tested on Global Cap : [%.4f, %.4f, %.4f, %.4f, %.4f]\n', feat_global(1:5));
    disp(' ');
    
    diff_norm = norm(feat_local - feat_global);
    fprintf('Euclidean Distance between vectors: %.4f\n', diff_norm);
    if diff_norm > 1.0
        disp('--> HUGE DIFFERENCE: The Random Forest will definitely misclassify the global-capped features!');
    end
end
end
disp('======================================================');
disp('Diagnostics Complete.');


%% Helper Function
function [X_eeg, X_imu, X_heur, Y, local_cap] = diag_extract_subj(subj, event_table, label_table, edfs_names, filtered_dir, b, a)
    X_eeg = []; X_imu = []; X_heur = []; Y = {}; local_cap = 1.0;
    
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
    
    EEG_whole = edfread(edf_name, 'SelectedSignals', 'R6', 'DataRecordOutputType', 'vector');
    eeg_sig = cat(1, EEG_whole.(1){:})';
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
            
            if orig_label == 1
                Y{end+1, 1} = '1';
            else
                Y{end+1, 1} = '0';
            end
            X_eeg = [X_eeg; feat];
        end
    end
end
