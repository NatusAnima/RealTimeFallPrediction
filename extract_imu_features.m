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
    
    % Signal Magnitude Area
    sma = sum(abs(acc_x) + abs(acc_y) + abs(acc_z)) / N;
    
    % Combined Magnitude
    acc_magnitude = sqrt(acc_x.^2 + acc_y.^2 + acc_z.^2);
    
    % Variance
    mag_var = var(acc_magnitude);
    
    % Minimum Magnitude
    min_mag = min(acc_magnitude);
    
    % Peak Jerk
    peak_jerk = max(abs(diff(acc_magnitude)));
    
    % Combine into output
    imu_features = [sma, mag_var, min_mag, peak_jerk];
end
