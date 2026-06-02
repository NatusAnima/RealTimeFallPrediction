% run_live_system.m
% Orchestrates the live data feed and real-time prediction using GUI.
% Directly connects to BrainBit using BrainFlow SDK.

clear all; close all; clc;

%% 1. Configuration & Setup
disp('--- 1. Configuration & Setup ---');
sys_order = 3;        
model_path = 'trained_model.mat'; 

if ~exist(model_path, 'file')
    error('Model file trained_model.mat not found. Please run train_model.m first.');
end

% Load Model and Parameters
load(model_path, 'mdl_eeg', 'mdl_imu', 'mdl_fusion', 'global_eeg_cap', 'b', 'a');
fprintf('Loaded pre-trained Late Fusion models.\n');

%% 2. Connect to BrainBit Hardware via BrainFlow
disp('--- 2. Connecting to BrainBit via BrainFlow ---');
addpath(genpath('C:\brainflow\brainflow'));

try
    BoardShim.enable_dev_board_logger();
    params = BrainFlowInputParams();
    params.mac_address = 'C4:29:B1:67:F8:A8';
    board_id = int32(BoardIds.BRAINBIT_BOARD);
    preset = int32(BrainFlowPresets.DEFAULT_PRESET);
    
    disp('Creating board...');
    board_shim = BoardShim(board_id, params);
    
    disp('Preparing session...');
    board_shim.prepare_session();
    
    disp('Starting stream...');
    board_shim.start_stream(45000, '');
    
    eeg_channels = BoardShim.get_eeg_channels(board_id, preset);
    accel_channels = BoardShim.get_accel_channels(board_id, preset);
    sampling_rate = BoardShim.get_sampling_rate(board_id, preset);
    
    fprintf('Connected! Sampling Rate: %d Hz\n', sampling_rate);
    
    % Ensure board is safely shut down on error or script close
    cleanupObj = onCleanup(@() cleanup_board(board_shim));
    
catch ME
    disp('Failed to connect to BrainBit.');
    disp(ME.message);
    return;
end

%% 3. GUI Setup
fig = uifigure('Name', 'Real-Time Fall Prediction Monitor', 'Position', [100, 100, 800, 700]);
fig.Color = [0.94 0.94 0.94];

% EEG Axes
ax_eeg = uiaxes(fig, 'Position', [50, 400, 700, 250]);
title(ax_eeg, 'Real-Time EEG Signals (All Channels)');
xlabel(ax_eeg, 'Samples (256ms Window, Upsampled to 1000Hz)');
ylabel(ax_eeg, 'Normalized Amplitude');
grid(ax_eeg, 'on');
hold(ax_eeg, 'on');
colors = lines(length(eeg_channels));
eeg_lines = gobjects(length(eeg_channels), 1);
for c = 1:length(eeg_channels)
    eeg_lines(c) = plot(ax_eeg, nan(1, 256), 'Color', colors(c,:), 'LineWidth', 1.5, 'DisplayName', sprintf('Ch %d', c));
end
legend(ax_eeg, 'Location', 'northeast');
hold(ax_eeg, 'off');
ax_eeg.YLim = [-2, 2];

% ACC Axes
ax_acc = uiaxes(fig, 'Position', [50, 100, 700, 250]);
title(ax_acc, 'Real-Time ACC Magnitude');
xlabel(ax_acc, 'Samples (Upsampled to 1000Hz)');
ylabel(ax_acc, 'Magnitude (g)');
grid(ax_acc, 'on');
hold(ax_acc, 'on');
acc_line = plot(ax_acc, nan(1, 256), 'k', 'LineWidth', 1.5);
yline(ax_acc, 0.7, 'r--', 'Heuristic Trigger', 'LineWidth', 2);
hold(ax_acc, 'off');
ax_acc.YLim = [0, 3];

% Status Indicator
status_lbl = uilabel(fig, 'Position', [250, 20, 300, 50], 'Text', 'STATUS: BUFFERING...', ...
    'FontSize', 20, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
    'FontColor', [0.5, 0.5, 0.5]);

%% 4. Real-Time Processing Loop
disp('--- 3. Starting Real-Time Loop ---');

blindfold_samples = 1000; % Time in equivalent 1000Hz samples (1 second)
ms_since_last_detection = blindfold_samples; 
pred_future = []; % Variable to hold the parfeval Future object

% We need 256ms of data.
samples_needed_raw = round(0.256 * sampling_rate);

% Wait briefly for buffer to fill
pause(1);

while isvalid(fig)
    try
        % Fetch data count from BrainFlow buffer
        data_count = board_shim.get_board_data_count(preset);
        
        if data_count < samples_needed_raw
            pause(0.01);
            continue;
        end
        
        if strcmp(status_lbl.Text, 'STATUS: BUFFERING...')
            status_lbl.Text = 'STATUS: MONITORING (SAFE)';
            status_lbl.FontColor = [0.1, 0.6, 0.1];
        end
        
        % Get only the most recent samples (256ms equivalent) - doesn't consume buffer
        data = board_shim.get_current_board_data(samples_needed_raw, preset);
        
        % MATLAB uses 1-based indexing for BrainFlow output rows
        eeg_data = data(eeg_channels + 1, :); 
        acc_data = data(accel_channels + 1, :); 
        
        % Time vectors for upsampling
        t_original = linspace(0, 1, samples_needed_raw);
        t_upsampled = linspace(0, 1, 256);
        
        % 1. Upsample ACC to 256 samples
        acc_upsampled = zeros(3, 256);
        for i = 1:3
            acc_upsampled(i, :) = interp1(t_original, acc_data(i, :), t_upsampled, 'spline');
        end
        
        % 2. Process all EEG channels (Upsample & Filter)
        eeg_norm_all = zeros(length(eeg_channels), 256);
        for c = 1:length(eeg_channels)
            % Upsample
            eeg_up = interp1(t_original, eeg_data(c, :), t_upsampled, 'spline');
            
            % Process
            [eeg_norm, ~] = preprocess_signal(eeg_up', acc_upsampled, b, a, global_eeg_cap);
            eeg_norm_all(c, :) = eeg_norm;
            
            % Update Plot
            eeg_lines(c).YData = eeg_norm;
        end
        
        % Process ACC magnitude (using the unified preprocess function which returns mag)
        [~, acc_mag] = preprocess_signal(eeg_norm_all(1, :)', acc_upsampled, b, a, global_eeg_cap);
        
        heur_score = calculate_imu_heuristic(acc_mag);
        
        % Update GUI Plots
        acc_line.YData = acc_mag;
        drawnow limitrate;
        
        ms_since_last_detection = ms_since_last_detection + 30; % Approx 30ms loop slide
        
        % Gate Check
        if ms_since_last_detection >= blindfold_samples
            gate_passed = heur_score >= 1.2;
            
            if gate_passed && (isempty(pred_future) || strcmp(pred_future.State, 'finished'))
                % Send N4SID Feature Extraction & Prediction to Background Pool
                status_lbl.Text = 'STATUS: GATE OPEN (EVALUATING...)';
                status_lbl.FontColor = [0.8, 0.5, 0.1];
                
                % Pass all inputs to predict_fall_wrapper
                pred_future = parfeval(backgroundPool, @predict_fall_wrapper, 1, eeg_norm_all(1, :), acc_upsampled, heur_score, sys_order, mdl_eeg, mdl_imu, mdl_fusion);
                
            elseif ~gate_passed && (isempty(pred_future) || strcmp(pred_future.State, 'finished'))
                % Reset text slowly back to monitoring if blindfold is over
                status_lbl.Text = 'STATUS: MONITORING (SAFE)';
                status_lbl.FontColor = [0.1, 0.6, 0.1];
            end
        end
        
        % Check if a background prediction finished
        if ~isempty(pred_future) && strcmp(pred_future.State, 'finished')
            try
                p_fall = fetchOutputs(pred_future);
                if p_fall >= 0.20
                    status_lbl.Text = sprintf('STATUS: FALL DETECTED! (%.2f)', p_fall);
                    status_lbl.FontColor = [0.8, 0.1, 0.1];
                    ms_since_last_detection = 0; % Activate Blindfold
                else
                    status_lbl.Text = sprintf('STATUS: GATE OPEN (SAFE %.2f)', p_fall);
                    status_lbl.FontColor = [0.8, 0.5, 0.1];
                end
            catch ME
                disp(['Background Prediction Error: ', ME.message]);
            end
            pred_future = []; % Reset future
        end
        
        pause(0.01); % Throttle CPU and allow buffer to fill (simulating 30ms slide)
        
    catch ME
        disp('Runtime Error:');
        disp(ME.message);
        break;
    end
end
disp('System Terminated.');

% --- Helper ---
function cleanup_board(board_shim)
    disp('Cleaning up BrainFlow session...');
    try
        board_shim.stop_stream();
        pause(1);
        board_shim.release_session();
    catch
    end
end
