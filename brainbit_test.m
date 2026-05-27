clear;
clc;

addpath(genpath('C:\brainflow\brainflow'));

rehash;

BoardShim.enable_dev_board_logger();

params = BrainFlowInputParams();

params.mac_address = 'C4:29:B1:67:F8:A8';

board_id = int32(BoardIds.BRAINBIT_BOARD);

preset = int32(BrainFlowPresets.DEFAULT_PRESET);

board_shim = [];

stream_started = false;
session_prepared = false;

try

    disp('Creating board...');
    board_shim = BoardShim(board_id, params);

    pause(2);

    disp('Preparing session...');
    board_shim.prepare_session();

    session_prepared = true;

    pause(1);

    disp('Starting stream...');
    board_shim.start_stream(45000, '');

    stream_started = true;

    % Allow BrainBit buffer to fill
    pause(5);

    % Get timestamp channel
    timestamp_channel = ...
        BoardShim.get_timestamp_channel( ...
        board_id, ...
        preset);

    fprintf('\n');
    fprintf('Timestamp Channel (BrainFlow): %d\n', ...
        timestamp_channel);

    fprintf('Using MATLAB Row Index: %d\n\n', ...
        timestamp_channel + 1);

    % ============================================
    % INITIAL CLOCK SYNCHRONIZATION
    % ============================================

    sync_data = board_shim.get_current_board_data( ...
        1, ...
        preset);

    sync_pc_time = posixtime(datetime('now'));

    % IMPORTANT FIX:
    % MATLAB indexing correction
    sync_eeg_time = ...
        sync_data(timestamp_channel + 1, end);

    clock_offset = ...
        sync_pc_time - sync_eeg_time;

    fprintf('====================================\n');
    fprintf('Clock Synchronization Completed\n');
    fprintf('Estimated Clock Offset: %.6f sec\n', ...
        clock_offset);
    fprintf('====================================\n\n');

    disp('Measuring EEG acquisition latency...');
    disp('Collecting ONLY 10 samples...');

    latency_values = [];

    % ============================================
    % ONLY 10 LATENCY MEASUREMENTS
    % ============================================

    for sample_counter = 1:10

        % Current PC timestamp
        pc_time = posixtime(datetime('now','TimeZone','UTC'));

        % Get newest EEG sample
        data = board_shim.get_current_board_data( ...
            1, ...
            preset);

        if isempty(data)
            continue;
        end

        % IMPORTANT FIX:
        % MATLAB indexing correction
        eeg_timestamp = ...
            data(timestamp_channel , end);

        

        acquisition_latency_ms = ...
            (pc_time - eeg_timestamp) * 1000;

        latency_values(end+1) = ...
            acquisition_latency_ms;

        fprintf('\n');
        fprintf('Sample %d\n', sample_counter);
        fprintf('------------------------------------\n');

        fprintf('PC Time          : %.6f\n', ...
            pc_time);

        fprintf('EEG Timestamp    : %.6f\n', ...
            eeg_timestamp);

        fprintf('Latency          : %.2f ms\n', ...
            acquisition_latency_ms);

        fprintf('------------------------------------\n');

        pause(0.1);

    end

    % ============================================
    % FINAL STATISTICS
    % ============================================

    fprintf('\n');
    fprintf('====================================\n');
    fprintf('FINAL LATENCY STATISTICS\n');
    fprintf('====================================\n');

    fprintf('Average Latency : %.2f ms\n', ...
        mean(latency_values));

    fprintf('Minimum Latency : %.2f ms\n', ...
        min(latency_values));

    fprintf('Maximum Latency : %.2f ms\n', ...
        max(latency_values));

    fprintf('Std Deviation   : %.2f ms\n', ...
        std(latency_values));

    fprintf('====================================\n');

catch ME

    disp('ERROR OCCURRED:');
    disp(ME.message);

end

disp('Cleaning up...');

try

    if stream_started

        board_shim.stop_stream();

        pause(2);

    end

catch
end

try

    if session_prepared

        board_shim.release_session();

        pause(2);

    end

catch
end

clear board_shim;

disp('DONE');