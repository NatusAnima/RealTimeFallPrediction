% run_live_system.m
% Orchestrates the live data feed and real-time prediction using GUI.
% Uses parfeval on the backgroundPool to prevent UI freezing during heavy N4SID calculations.

clear all; close all; clc;

%% 1. Configuration & Setup
disp('--- 1. Configuration & Setup ---');
TCP_HOST = '127.0.0.1';
TCP_PORT = 5005;
channel_idx = 1;      
sys_order = 3;        
model_path = 'trained_model.mat'; 

if ~exist(model_path, 'file')
    error('Model file trained_model.mat not found. Please run train_model.m first.');
end

% Load Model and Parameters
load(model_path, 'rf_model', 'acc_threshold', 'global_eeg_cap', 'b', 'a');
fprintf('Loaded pre-trained model. ACC Threshold: %.2f\n', acc_threshold);

%% 2. Connect to Python DAQ
disp('--- 2. Connecting to Python Transfer Layer ---');
try
    t = tcpclient(TCP_HOST, TCP_PORT);
    disp('Successfully connected to Python TCP Server.');
catch ME
    error(['Failed to connect to Python TCP Server at ', TCP_HOST, ':', num2str(TCP_PORT)]);
end

%% 3. GUI Setup
fig = uifigure('Name', 'Real-Time Fall Prediction Monitor', 'Position', [100, 100, 800, 600]);
fig.Color = [0.94 0.94 0.94];

% EEG Axes
ax_eeg = uiaxes(fig, 'Position', [50, 350, 700, 200]);
title(ax_eeg, 'Real-Time EEG Signal');
xlabel(ax_eeg, 'Samples (256ms Window)');
ylabel(ax_eeg, 'Normalized Amplitude');
grid(ax_eeg, 'on');

% ACC Axes
ax_acc = uiaxes(fig, 'Position', [50, 100, 700, 200]);
title(ax_acc, 'Real-Time ACC Magnitude');
xlabel(ax_acc, 'Samples');
ylabel(ax_acc, 'Magnitude (g)');
grid(ax_acc, 'on');
hold(ax_acc, 'on');
yline(ax_acc, acc_threshold, 'r--', 'Gate Threshold', 'LineWidth', 2);
hold(ax_acc, 'off');

% Status Indicator
status_lbl = uilabel(fig, 'Position', [250, 20, 300, 50], 'Text', 'STATUS: MONITORING (SAFE)', ...
    'FontSize', 20, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
    'FontColor', [0.1, 0.6, 0.1]);

% Initial Plot Lines
eeg_line = plot(ax_eeg, nan(1, 256), 'b', 'LineWidth', 1.5);
acc_line = plot(ax_acc, nan(1, 256), 'k', 'LineWidth', 1.5);
ax_eeg.YLim = [-2, 2];
ax_acc.YLim = [0, max(2, acc_threshold + 1)];

%% 4. Real-Time Processing Loop
disp('--- 3. Starting Real-Time Loop ---');

blindfold_samples = 1000; 
ms_since_last_detection = blindfold_samples; 
pred_future = []; % Variable to hold the parfeval Future object

while isvalid(fig)
    try
        % Request latest buffer from Python
        write(t, uint8('GET'));
        data = readline(t);
        
        if isempty(data)
            pause(0.01);
            continue;
        end
        
        buffer_dict = jsondecode(data);
        if strcmp(buffer_dict.status, 'waiting')
            pause(0.01); 
            continue;
        end
        
        eeg_buffer = buffer_dict.eeg;
        acc_buffer = buffer_dict.acc;
        
        ms_since_last_detection = ms_since_last_detection + 30; 
        
        % Preprocess Signal Module
        raw_eeg_epoch = eeg_buffer(:, channel_idx);
        if size(acc_buffer, 1) > size(acc_buffer, 2)
            acc_buffer = acc_buffer';
        end
        [eeg_norm, acc_mag] = preprocess_signal(raw_eeg_epoch, acc_buffer, b, a, global_eeg_cap);
        
        % Update GUI Plots
        eeg_line.YData = eeg_norm;
        acc_mag_interp = interp1(linspace(1,256,length(acc_mag)), acc_mag, 1:256, 'linear', 'extrap');
        acc_line.YData = acc_mag_interp;
        drawnow limitrate;
        
        % Gate Check
        if ms_since_last_detection >= blindfold_samples
            [gate_passed, max_acc] = evaluate_acc_gate(acc_mag, acc_threshold);
            
            if gate_passed && (isempty(pred_future) || strcmp(pred_future.State, 'finished'))
                % Send N4SID Feature Extraction & Prediction to Background Pool
                status_lbl.Text = 'STATUS: GATE OPEN (EVALUATING...)';
                status_lbl.FontColor = [0.8, 0.5, 0.1];
                
                pred_future = parfeval(backgroundPool, @predict_fall_wrapper, 1, eeg_norm, sys_order, rf_model);
            elseif ~gate_passed && (isempty(pred_future) || strcmp(pred_future.State, 'finished'))
                % Reset text slowly back to monitoring if blindfold is over
                status_lbl.Text = 'STATUS: MONITORING (SAFE)';
                status_lbl.FontColor = [0.1, 0.6, 0.1];
            end
        end
        
        % Check if a background prediction finished
        if ~isempty(pred_future) && strcmp(pred_future.State, 'finished')
            try
                pred_label = fetchOutputs(pred_future);
                if strcmp(pred_label, '1')
                    status_lbl.Text = 'STATUS: FALL DETECTED!';
                    status_lbl.FontColor = [0.8, 0.1, 0.1];
                    ms_since_last_detection = 0; % Activate Blindfold
                else
                    status_lbl.Text = 'STATUS: GATE OPEN (SAFE)';
                    status_lbl.FontColor = [0.8, 0.5, 0.1];
                end
            catch ME
                disp(['Background Prediction Error: ', ME.message]);
            end
            pred_future = []; % Reset future
        end
        
        pause(0.01);
        
    catch ME
        % Disconnection Handle
        status_lbl.Text = 'STATUS: CONNECTION LOST';
        status_lbl.FontColor = [0.5, 0.5, 0.5];
        pause(2);
        try
            t = tcpclient(TCP_HOST, TCP_PORT);
            status_lbl.Text = 'STATUS: RECONNECTED';
        catch
        end
    end
end
disp('System Terminated.');
