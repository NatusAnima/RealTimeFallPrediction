% realtime_pipeline.m
% Decoupled Real-Time MATLAB Pipeline for EEG-based Fall Prediction
% This script connects to the Python DAQ Transfer Layer and processes the continuous data stream.

clear all; close all; clc;

%% 1. Configuration & Setup
disp('--- 1. Configuration & Setup ---');

% Python Transfer Layer Address
TCP_HOST = '127.0.0.1';
TCP_PORT = 5005;

% Pipeline parameters
channel_idx = 1;      % Which EEG channel to use from the buffer
sys_order = 3;        % State-space order for N4SID
acc_threshold = 1.5;  % Constant for now: Threshold to trigger N4SID prediction
eeg_cap = 1.0;        % Constant for now: Normalization cap for EEG (could be dynamic later)
model_path = 'trained_model.mat'; % Pre-trained fitcensemble model

fs = 1000; % EEG sampling rate
[b, a] = butter(2, [2.5, 30] / (fs/2), 'bandpass');

% Optional: Load Model if it exists
if exist(model_path, 'file')
    load(model_path, 'rf_model');
    disp('Loaded pre-trained model.');
    has_model = true;
else
    warning('No pre-trained model found. Feature extraction will run but prediction will be skipped.');
    has_model = false;
end

%% 2. Connect to Python DAQ
disp('--- 2. Connecting to Python Transfer Layer ---');
try
    t = tcpclient(TCP_HOST, TCP_PORT);
    disp('Successfully connected to Python TCP Server.');
catch ME
    error(['Failed to connect to Python TCP Server at ', TCP_HOST, ':', num2str(TCP_PORT), '. Ensure daq_synchronizer.py is running.']);
end

%% 3. Real-Time Processing Loop
disp('--- 3. Starting Real-Time Loop ---');

blindfold_samples = 1000; % 1 second blindfold after a fall detection (relative to ms)
ms_since_last_detection = blindfold_samples; % Initialize outside of blindfold

while true
    try
        % 1. Request latest buffer from Python
        write(t, uint8('GET'));
        
        % 2. Read response (newline delimited JSON)
        data = readline(t);
        if isempty(data)
            continue;
        end
        
        % 3. Decode the JSON dictionary
        buffer_dict = jsondecode(data);
        
        if strcmp(buffer_dict.status, 'waiting')
            pause(0.01); % Yield slightly if buffer is not full yet
            continue;
        end
        
        % eeg_buffer is [256 x 4] matrix, acc_buffer is [~25 x 3] matrix
        eeg_buffer = buffer_dict.eeg;
        acc_buffer = buffer_dict.acc;
        
        % Advance time tracker (assuming loop ticks roughly every 30ms-50ms)
        % For precise timing, you could use tic/toc or read Python timestamps.
        ms_since_last_detection = ms_since_last_detection + 30; 
        
        % Manage blindfold logic
        if ms_since_last_detection < blindfold_samples
             continue; 
        end
        
        % 4. Evaluate Accelerometer Gate
        % Transpose acc_buffer to [3 x N] to match normalized_acc expectations
        acc_norm = normalized_acc(acc_buffer');
        max_acc = max(acc_norm);
        
        if max_acc >= acc_threshold
            % --- GATE OPENED ---
            
            % Extract the specific channel epoch
            raw_eeg_epoch = eeg_buffer(:, channel_idx);
            
            % Apply bandpass filter
            eeg_sig_filtered = filtfilt(b, a, raw_eeg_epoch);
            
            % Normalize and Baseline Cap
            epoch_bn = (eeg_sig_filtered - mean(eeg_sig_filtered(1:40))) ./ eeg_cap;
            
            % Extract N4SID Features
            feat = extract_features(epoch_bn, sys_order);
            
            % 5. Predict using the trained model
            if has_model
                pred_cell = predict(rf_model, feat);
                pred_label = pred_cell{1};
                
                if strcmp(pred_label, '1')
                    fprintf('\n[!!!] FALL PREDICTED! (ACC Max: %.2f) [!!!]\n', max_acc);
                    % Activate 1-second blindfold
                    ms_since_last_detection = 0; 
                else
                    fprintf('Gate Opened (ACC=%.2f) - Prediction: Safe / Non-Fall\n', max_acc);
                end
            else
                fprintf('Gate Opened (ACC=%.2f)! Features extracted. size=[%d %d]\n', max_acc, size(feat,1), size(feat,2));
            end
        end
        
        % Small pause to prevent MATLAB from freezing the OS thread
        pause(0.01);
        
    catch ME
        % Handle closed connection or JSON parsing errors safely
        disp(['Connection lost or error: ', ME.message]);
        disp('Reconnecting in 2 seconds...');
        pause(2);
        try
            t = tcpclient(TCP_HOST, TCP_PORT);
            disp('Reconnected.');
        catch
        end
    end
end
