function run_offline_simulation_gui()
% run_offline_simulation_gui
% Simulates the real-time live system using offline EDF data with speed controls.

clear all; close all; clc;

% --- 1. GUI Setup ---
fig = uifigure('Name', 'Offline Live Simulation', 'Position', [100, 100, 1100, 750]);
fig.Color = [0.94 0.94 0.94];

uilabel(fig, 'Position', [250, 700, 400, 30], 'Text', 'Offline Real-Time Simulator', ...
    'FontSize', 20, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% Target Selection (Dynamic)
raw_data_dir = fullfile(pwd, 'Raw_Data');
table_path = fullfile(raw_data_dir, 'All34_table.mat');
if exist(table_path, 'file')
    load(table_path, 'event_table');
    subjects_list = event_table.Subject';
    dropdown_items = arrayfun(@(x) sprintf('Subject %d', x), subjects_list, 'UniformOutput', false);
else
    dropdown_items = {'No Data'};
    subjects_list = [];
end

uilabel(fig, 'Position', [50, 650, 100, 30], 'Text', 'Select Subject:', 'FontSize', 12, 'FontWeight', 'bold');
subj_dropdown = uidropdown(fig, 'Position', [150, 655, 150, 25], 'Items', dropdown_items);

uilabel(fig, 'Position', [330, 650, 100, 30], 'Text', 'Speed:', 'FontSize', 12, 'FontWeight', 'bold');
speed_dropdown = uidropdown(fig, 'Position', [390, 655, 120, 25], ...
    'Items', {'0.25x', '0.5x', '1x (Real-time)', '2x', '5x', 'Max Speed'}, 'Value', '1x (Real-time)');

run_btn = uibutton(fig, 'push', 'Position', [550, 645, 120, 40], 'Text', 'Start Simulation', ...
    'FontWeight', 'bold', 'BackgroundColor', [0.2 0.6 0.8], 'FontColor', 'w', ...
    'ButtonPushedFcn', @(btn,event) run_simulation(fig));

stop_btn = uibutton(fig, 'push', 'Position', [690, 645, 120, 40], 'Text', 'Stop / Reset', ...
    'FontWeight', 'bold', 'BackgroundColor', [0.8 0.3 0.3], 'FontColor', 'w', ...
    'ButtonPushedFcn', @(btn,event) stop_simulation(fig));

% Status Indicator
status_lbl = uilabel(fig, 'Position', [300, 580, 300, 50], 'Text', 'STATUS: IDLE', ...
    'FontSize', 20, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
    'FontColor', [0.5, 0.5, 0.5]);

% EEG Axes
ax_eeg = uiaxes(fig, 'Position', [50, 320, 800, 250]);
title(ax_eeg, 'Simulated EEG Signal (Channel R6)');
xlabel(ax_eeg, 'Samples (256ms Window)');
ylabel(ax_eeg, 'Normalized Amplitude');
grid(ax_eeg, 'on');
hold(ax_eeg, 'on');
eeg_line = plot(ax_eeg, nan(1, 256), 'Color', [0 0.4470 0.7410], 'LineWidth', 1.5);
hold(ax_eeg, 'off');
ax_eeg.YLim = [-2, 2];

% ACC Axes
ax_acc = uiaxes(fig, 'Position', [50, 30, 800, 250]);
title(ax_acc, 'Simulated ACC Magnitude');
xlabel(ax_acc, 'Samples (256ms Window)');
ylabel(ax_acc, 'Magnitude (g)');
grid(ax_acc, 'on');
hold(ax_acc, 'on');
acc_line = plot(ax_acc, nan(1, 256), 'k', 'LineWidth', 1.5);
yline(ax_acc, 1.2, 'r--', 'Heuristic Trigger (1.2g)', 'LineWidth', 2);
hold(ax_acc, 'off');
ax_acc.YLim = [0, 3];

% Listbox for events
uilabel(fig, 'Position', [870, 580, 200, 30], 'Text', 'Event Timestamps:', 'FontSize', 12, 'FontWeight', 'bold');
event_listbox = uilistbox(fig, 'Position', [870, 200, 200, 380], 'Items', {'Loading...'}, ...
    'ValueChangedFcn', @(lb,event) listbox_jump(fig, lb));

% Slider for seeking
time_slider = uislider(fig, 'Position', [100, 620, 700, 3], 'Limits', [0, 100], ...
    'ValueChangedFcn', @(sld,event) slider_jump(fig, sld));
uilabel(fig, 'Position', [50, 610, 50, 30], 'Text', 'Time:');

% Event lines for plots
event_line_eeg = xline(ax_eeg, 1, 'r-', 'Event', 'LineWidth', 2, 'Visible', 'off');
event_line_acc = xline(ax_acc, 1, 'r-', 'Event', 'LineWidth', 2, 'Visible', 'off');

fig.UserData = struct('subj_dropdown', subj_dropdown, 'speed_dropdown', speed_dropdown, ...
    'raw_data_dir', raw_data_dir, 'status_lbl', status_lbl, ...
    'eeg_line', eeg_line, 'acc_line', acc_line, 'is_running', false, 'fig', fig, ...
    'time_slider', time_slider, 'event_listbox', event_listbox, 'target_sample', [], ...
    'event_line_eeg', event_line_eeg, 'event_line_acc', event_line_acc);
end

function slider_jump(fig, sld)
    val = sld.Value; % in seconds
    fig.UserData.target_sample = max(1, round(val * 1000));
end

function listbox_jump(fig, lb)
    val = lb.Value;
    % Format is e.g. "Fall at 15.34 s"
    tokens = regexp(val, '(\d+\.\d+) s', 'tokens');
    if ~isempty(tokens)
        time_s = str2double(tokens{1}{1});
        % Jump to 2 seconds before the event
        fig.UserData.target_sample = max(1, round((time_s - 2) * 1000));
    end
end

function stop_simulation(fig)
    fig.UserData.is_running = false;
    fig.UserData.status_lbl.Text = 'STATUS: STOPPED';
    fig.UserData.status_lbl.FontColor = [0.5, 0.5, 0.5];
end

function run_simulation(fig)
    ud = fig.UserData;
    if ud.is_running
        return; % Already running
    end
    fig.UserData.is_running = true;
    ud = fig.UserData; % refresh
    
    % Get settings
    val = ud.subj_dropdown.Value;
    subj = sscanf(val, 'Subject %d');
    speed_val = ud.speed_dropdown.Value;
    
    % Parse Speed
    is_max_speed = strcmp(speed_val, 'Max Speed');
    delay_s = 0.03; % Base delay for 30ms step
    if strcmp(speed_val, '0.25x')
        delay_s = 0.12;
    elseif strcmp(speed_val, '0.5x')
        delay_s = 0.06;
    elseif strcmp(speed_val, '1x (Real-time)')
        delay_s = 0.03;
    elseif strcmp(speed_val, '2x')
        delay_s = 0.015;
    elseif strcmp(speed_val, '5x')
        delay_s = 0.006;
    elseif is_max_speed
        delay_s = 0;
    end
    
    % Load Models
    ud.status_lbl.Text = 'LOADING MODELS...';
    drawnow;
    try
        load('trained_model.mat', 'mdl_eeg', 'mdl_imu', 'mdl_fusion', 'global_eeg_cap', 'b', 'a');
    catch
        uialert(fig, 'trained_model.mat not found. Run train_model.m first.', 'Error');
        fig.UserData.is_running = false;
        ud.status_lbl.Text = 'STATUS: IDLE';
        return;
    end
    
    % Load Subject Data
    ud.status_lbl.Text = 'LOADING EDF DATA...';
    drawnow;
    filtered_dir = fullfile(ud.raw_data_dir, 'Filtered');
    my_edfs_dir = dir(fullfile(filtered_dir, '*edf'));
    edfs_names = {my_edfs_dir.name};
    
    if subj < 10
        edf_idx = find(contains(edfs_names, ['0', num2str(subj)]));
    else
        edf_idx = find(contains(edfs_names, num2str(subj)));
    end
    
    if isempty(edf_idx)
        uialert(fig, 'EDF file not found for selected subject.', 'Error');
        fig.UserData.is_running = false;
        ud.status_lbl.Text = 'STATUS: IDLE';
        return;
    end
    edf_name = fullfile(filtered_dir, edfs_names{edf_idx(1)});
    
    try
        EEG_whole = edfread(edf_name, 'SelectedSignals', 'R6', 'DataRecordOutputType', 'vector');
        eeg_sig = cat(1, EEG_whole.(1){:})';
        ACC_X = edfread(edf_name, 'SelectedSignals', 'x_dir', 'DataRecordOutputType', 'vector');
        ACC_Y = edfread(edf_name, 'SelectedSignals', 'y_dir', 'DataRecordOutputType', 'vector');
        ACC_Z = edfread(edf_name, 'SelectedSignals', 'z_dir', 'DataRecordOutputType', 'vector');
        acc_sig = [cat(1, ACC_X.(1){:})'; cat(1, ACC_Y.(1){:})'; cat(1, ACC_Z.(1){:})'] ./ 980;
    catch
        uialert(fig, 'Failed to read EDF data.', 'Error');
        fig.UserData.is_running = false;
        ud.status_lbl.Text = 'STATUS: IDLE';
        return;
    end
    
    ud.status_lbl.Text = 'STATUS: MONITORING (SAFE)';
    ud.status_lbl.FontColor = [0.1, 0.6, 0.1];
    
    % Update Listbox and Slider
    load(fullfile(ud.raw_data_dir, 'All34_table.mat'), 'event_table');
    load(fullfile(ud.raw_data_dir, 'Label_Table.mat'), 'label_table');
    
    table_idx = find(event_table.Subject == subj);
    subj_events = event_table{table_idx, 3:end};
    subj_labels = label_table{table_idx, 3:end};
    valid_idx = ~isnan(subj_events);
    subj_events = subj_events(valid_idx);
    subj_labels = subj_labels(valid_idx);
    
    listbox_items = {};
    for idx = 1:length(subj_events)
        time_s = subj_events(idx) / 1000;
        lbl = subj_labels(idx);
        if lbl == 1
            lbl_str = 'Fall';
        else
            lbl_str = 'Non-Fall';
        end
        listbox_items{end+1} = sprintf('%s at %.2f s', lbl_str, time_s);
    end
    if isempty(listbox_items)
        listbox_items = {'No Events'};
    end
    ud.event_listbox.Items = listbox_items;
    
    total_samples = length(eeg_sig);
    ud.time_slider.Limits = [0, total_samples / 1000];
    ud.time_slider.Value = 0;
    
    % Simulation Parameters
    window_size = 256;
    step_size = 30; % 30 samples = 30ms slide at 1000Hz
    sys_order = 3;
    
    blindfold_samples = 1000; 
    ms_since_last_detection = blindfold_samples;
    
    total_samples = length(eeg_sig);
    current_sample = 1;
    
    % Metrics Tracking
    detections = []; % Stores indices where falls were detected
    
    % Simulation Loop
    tic;
    last_ui_update = tic;
    
    while current_sample + window_size - 1 <= total_samples && fig.UserData.is_running
        ud = fig.UserData; % refresh inside loop
        
        if ~isempty(ud.target_sample)
            current_sample = ud.target_sample;
            fig.UserData.target_sample = []; % reset
            ms_since_last_detection = blindfold_samples; % clear blindfold
        end
        
        if ~is_max_speed && mod(current_sample, 1000) < step_size
            fig.UserData.time_slider.Value = current_sample / 1000;
        end
        
        end_sample = current_sample + window_size - 1;
        
        eeg_window = eeg_sig(current_sample:end_sample);
        acc_window = acc_sig(:, current_sample:end_sample);
        
        % Preprocess (exact same as live)
        [eeg_norm, acc_mag] = preprocess_signal(eeg_window', acc_window, b, a, global_eeg_cap);
        
        % Transpose eeg_norm if necessary to match 1x256
        if size(eeg_norm, 1) > size(eeg_norm, 2)
            eeg_norm = eeg_norm';
        end
        
        % Heuristic Gate
        heur_score = calculate_imu_heuristic(acc_mag);
        
        ms_since_last_detection = ms_since_last_detection + step_size;
        gate_passed = heur_score >= 1.2;
        
        status_text = 'STATUS: MONITORING (SAFE)';
        status_color = [0.1, 0.6, 0.1];
        
        if ms_since_last_detection >= blindfold_samples
            if gate_passed
                status_text = 'STATUS: GATE OPEN (EVALUATING...)';
                status_color = [0.8, 0.5, 0.1];
                
                % Synchronous prediction to ensure temporal simulation accuracy
                p_fall = predict_fall_wrapper(eeg_norm, acc_window, heur_score, sys_order, mdl_eeg, mdl_imu, mdl_fusion);
                
                if p_fall >= 0.20
                    status_text = sprintf('STATUS: FALL DETECTED! (%.2f)', p_fall);
                    status_color = [0.8, 0.1, 0.1];
                    ms_since_last_detection = 0; % Activate Blindfold
                    
                    % Record detection (center of window)
                    detections(end+1) = current_sample + round(window_size/2);
                else
                    status_text = sprintf('STATUS: GATE OPEN (SAFE %.2f)', p_fall);
                end
            end
        else
            % Blindfold active
            status_text = 'STATUS: BLINDFOLD ACTIVE';
            status_color = [0.5, 0.5, 0.5];
        end
        
        % Update GUI
        if ~is_max_speed
            ud.eeg_line.YData = eeg_norm;
            ud.acc_line.YData = acc_mag;
            ud.status_lbl.Text = status_text;
            ud.status_lbl.FontColor = status_color;
            
            % Event markers
            events_in_window = subj_events(subj_events >= current_sample & subj_events <= end_sample);
            if ~isempty(events_in_window)
                x_pos = events_in_window(1) - current_sample + 1;
                ud.event_line_eeg.Value = x_pos;
                ud.event_line_eeg.Visible = 'on';
                ud.event_line_acc.Value = x_pos;
                ud.event_line_acc.Visible = 'on';
            else
                ud.event_line_eeg.Visible = 'off';
                ud.event_line_acc.Visible = 'off';
            end
            
            drawnow limitrate;
            pause(delay_s);
        else
            % For Max Speed, only update GUI occasionally to show progress
            if toc(last_ui_update) > 0.5
                ud.status_lbl.Text = sprintf('SIMULATING... (%.1f%%)', (current_sample/total_samples)*100);
                ud.status_lbl.FontColor = [0.2 0.2 0.8];
                drawnow limitrate;
                last_ui_update = tic;
            end
        end
        
        current_sample = current_sample + step_size;
    end
    
    fig.UserData.is_running = false;
    ud.status_lbl.Text = 'STATUS: SIMULATION COMPLETE';
    ud.status_lbl.FontColor = [0.1, 0.6, 0.1];
    
    % Show detailed results
    msg = sprintf('Simulation Finished!\n\nTotal Detections: %d\n', length(detections));
    if ~isempty(detections)
        msg = [msg, 'Detection Timestamps (seconds):\n'];
        for i = 1:min(10, length(detections))
            msg = [msg, sprintf(' - %.2f s\n', detections(i) / 1000)];
        end
        if length(detections) > 10
            msg = [msg, sprintf(' ... and %d more.', length(detections)-10)];
        end
    end
    uialert(fig, msg, 'Simulation Results');
end
