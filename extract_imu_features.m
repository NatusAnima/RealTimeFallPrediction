function imu_features = extract_imu_features(acc_epoch)
    % extract_imu_features Extracts deterministic kinematic features from ACC
    % 
    % Inputs:
    %   acc_epoch - A 3xN array of acceleration data (X, Y, Z rows)
    %
    % Outputs:
    %   imu_features - A 1x4 array containing:
    %       1. Signal Magnitude Area (SMA)
    %       2. Variance of magnitude
    %       3. Minimum magnitude
    %       4. Peak jerk (max diff of magnitude)
    
    N = size(acc_epoch, 2);
    
    % Smooth the 3 axes
    acc_x = sgolayfilt(acc_epoch(1,:), 3, 21);
    acc_y = sgolayfilt(acc_epoch(2,:), 3, 21);
    acc_z = sgolayfilt(acc_epoch(3,:), 3, 21);
    
    % Combined Magnitude
    acc_magnitude = sqrt(acc_x.^2 + acc_y.^2 + acc_z.^2);
    
    % Max Absolute Amplitude
    max_amp = max(abs(acc_magnitude));
    
    % Peak Jerk
    peak_jerk = max(abs(diff(acc_magnitude)));
    
    % Combine into output
    imu_features = [max_amp, peak_jerk];
end
