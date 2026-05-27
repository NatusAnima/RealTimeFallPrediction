function [eeg_normalized, acc_magnitude] = preprocess_signal(raw_eeg, raw_acc, filter_b, filter_a, eeg_cap)
    % preprocess_signal applies filtering and normalization for the realtime pipeline
    % 
    % Inputs:
    %   raw_eeg - 1D array of raw EEG channel data (e.g., 256 samples)
    %   raw_acc - 3xN array of raw Accelerometer data
    %   filter_b - Numerator coefficients of Butterworth bandpass filter
    %   filter_a - Denominator coefficients of Butterworth bandpass filter
    %   eeg_cap  - Baseline EEG normalization factor
    %
    % Outputs:
    %   eeg_normalized - Filtered and baseline-normalized EEG
    %   acc_magnitude  - Normalized 3D acceleration magnitude
    
    % --- EEG Preprocessing ---
    % Apply bandpass filter
    eeg_filtered = filtfilt(filter_b, filter_a, raw_eeg);
    
    % Baseline shift using first 40 samples (similar to offline pipeline)
    baseline_val = mean(eeg_filtered(1:min(40, length(eeg_filtered))));
    
    % Normalize using cap
    eeg_normalized = (eeg_filtered - baseline_val) ./ eeg_cap;
    
    % --- ACC Preprocessing ---
    % Normalize using Savitzky-Golay filtering and combine axes
    acc_magnitude = normalized_acc(raw_acc);
end
