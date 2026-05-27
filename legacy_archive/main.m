%% ============================================================
%  FallDetectionRealTime - LOSO with Threshold Tuning
%  L6 + R6 + ACC
%  Fixed: window_size = 256, sys_order = 3
% ============================================================

clear;
clc;
close all;
warning off;

%% Paths

project_root = pwd;
addpath(genpath(project_root));

raw_data_dir = fullfile(project_root, 'Raw_Data');
edf_dir = fullfile(raw_data_dir, 'Filtered');
results_dir = fullfile(project_root, 'Results');

if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

fprintf('\n====================================\n');
fprintf('LOSO THRESHOLD TUNING - L6 + R6 + ACC\n');
fprintf('====================================\n');

%% Load data

load(fullfile(raw_data_dir, 'All34_table.mat'));   % event_table
load(fullfile(raw_data_dir, 'Label_table.mat'));   % label_table

edf_files = dir(fullfile(edf_dir, '*.edf'));

if isempty(edf_files)
    error('No EDF files found inside Raw_Data/Filtered.');
end

edfs_names = fullfile({edf_files.folder}, {edf_files.name});

fprintf('Tables loaded successfully.\n');
fprintf('EDF files found: %d\n', length(edfs_names));

%% Parameters

subjects_list = 2:41;

window_size = 256;
sys_order = 3;
delay_time = 80;
overlap = 1;
train_nofall = 'True';

threshold_list = 0.30:0.05:0.70;

fprintf('\nSubjects      : %s\n', mat2str(subjects_list));
fprintf('Channels      : L6 + R6 + ACC\n');
fprintf('Window size   : %d\n', window_size);
fprintf('System order  : %d\n', sys_order);
fprintf('Delay time    : %d\n', delay_time);
fprintf('Thresholds    : %s\n', mat2str(threshold_list));

%% Preprocessing

fprintf('\n====================================\n');
fprintf('PREPROCESSING L6 + R6 + ACC\n');
fprintf('====================================\n');

[eeg_windows_L6, eeg_windows_R6, acc_windows, mapped_labels, numeric_labels, subject_per_window] = ...
    preprocessing.create_l6_r6_acc_windows( ...
    subjects_list, ...
    edfs_names, ...
    event_table, ...
    label_table, ...
    window_size, ...
    delay_time, ...
    overlap, ...
    train_nofall);

fprintf('\nPreprocessing complete.\n');
fprintf('L6 windows  : %d x %d\n', size(eeg_windows_L6,1), size(eeg_windows_L6,2));
fprintf('R6 windows  : %d x %d\n', size(eeg_windows_R6,1), size(eeg_windows_R6,2));
fprintf('ACC windows : %d x %d\n', size(acc_windows,1), size(acc_windows,2));
fprintf('Labels      : %d\n', length(mapped_labels));

%% Feature extraction

fprintf('\n====================================\n');
fprintf('N4SID FEATURE EXTRACTION\n');
fprintf('====================================\n');

tic;

fprintf('\nExtracting L6 features...\n');
features_L6 = preprocessing.feature_extraction(eeg_windows_L6, sys_order);

fprintf('\nExtracting R6 features...\n');
features_R6 = preprocessing.feature_extraction(eeg_windows_R6, sys_order);

fprintf('\nExtracting ACC features...\n');
features_ACC = preprocessing.feature_extraction(acc_windows, sys_order);

features = [features_L6, features_R6, features_ACC];

feature_time = toc;

fprintf('\nFeature extraction complete.\n');
fprintf('Final feature matrix : %d x %d\n', size(features,1), size(features,2));
fprintf('Feature extraction time : %.2f seconds\n', feature_time);

%% LOSO: collect fall scores for every unseen subject

fprintf('\n====================================\n');
fprintf('LOSO SCORE COLLECTION\n');
fprintf('====================================\n');

all_true = {};
all_scores = [];

per_subject_default_results = table();

for test_subject = subjects_list

    fprintf('\n------------------------------------\n');
    fprintf('TEST SUBJECT: %d\n', test_subject);
    fprintf('------------------------------------\n');

    test_idx = subject_per_window == test_subject;
    train_idx = ~test_idx;

    X_train = features(train_idx, :);
    y_train = mapped_labels(train_idx);

    X_test = features(test_idx, :);
    y_test = mapped_labels(test_idx);

    fprintf('Train samples: %d\n', size(X_train,1));
    fprintf('Test samples : %d\n', size(X_test,1));

    trainedClassifier = trainer_class.train_offline_classifier(X_train, y_train);

    X_test_table = array2table(X_test, ...
        'VariableNames', trainedClassifier.PredictorNames);

    [default_pred, scores] = predict(trainedClassifier.ClassificationEnsemble, X_test_table);

    fall_score = scores(:, 2);   % class 2 = Unexpected/Fall

    default_metrics = trainer_class.evaluate_predictions(y_test, default_pred);
    default_metrics.Subject = test_subject;
    default_metrics = movevars(default_metrics, 'Subject', 'Before', 1);

    per_subject_default_results = [per_subject_default_results; default_metrics];

    all_true = [all_true; y_test(:)];
    all_scores = [all_scores; fall_score(:)];

    disp(default_metrics);
end

%% Threshold tuning

fprintf('\n====================================\n');
fprintf('THRESHOLD TUNING RESULTS\n');
fprintf('====================================\n');

threshold_results = table();

for threshold = threshold_list

    y_pred_threshold = repmat({'Expected'}, size(all_scores));
    y_pred_threshold(all_scores >= threshold) = {'Unexpected'};

    metrics = trainer_class.evaluate_predictions(all_true, y_pred_threshold);

    metrics.Threshold = threshold;
    metrics = movevars(metrics, 'Threshold', 'Before', 1);

    threshold_results = [threshold_results; metrics];

    disp(metrics);
end

threshold_results = sortrows(threshold_results, 'balanced_accuracy', 'descend');

fprintf('\n====================================\n');
fprintf('BEST THRESHOLD RESULTS\n');
fprintf('====================================\n');

disp(threshold_results);

best_threshold = threshold_results.Threshold(1);

fprintf('\nBest threshold based on balanced accuracy: %.2f\n', best_threshold);

%% Save results

fprintf('\n====================================\n');
fprintf('SAVE RESULTS\n');
fprintf('====================================\n');

default_results_path = fullfile(results_dir, 'loso_default_results_L6_R6_ACC.csv');
threshold_results_path = fullfile(results_dir, 'threshold_tuning_results_L6_R6_ACC.csv');
scores_path = fullfile(results_dir, 'loso_fall_scores_L6_R6_ACC.csv');

writetable(per_subject_default_results, default_results_path);
writetable(threshold_results, threshold_results_path);

score_table = table( ...
    all_true(:), ...
    all_scores(:), ...
    'VariableNames', {'TrueLabel', 'FallScore'});

writetable(score_table, scores_path);

fprintf('Saved default LOSO results    : %s\n', default_results_path);
fprintf('Saved threshold tuning results: %s\n', threshold_results_path);
fprintf('Saved LOSO fall scores        : %s\n', scores_path);

fprintf('\n====================================\n');
fprintf('THRESHOLD TUNING FINISHED\n');
fprintf('====================================\n');