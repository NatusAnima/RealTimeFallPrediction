function p_fall = predict_fall(n4sid_features, imu_features, heuristic_score, mdl_eeg, mdl_imu, mdl_fusion)
    % predict_fall Uses the trained Late Fusion model to predict a fall
    %
    % Inputs:
    %   n4sid_features  - 1D array of extracted N4SID state-space matrices
    %   imu_features    - 1D array of extracted IMU features
    %   heuristic_score - Heuristic double indicating likelihood of free-fall
    %   mdl_eeg         - A trained fitcensemble classification model for EEG
    %   mdl_imu         - A trained fitcensemble classification model for IMU
    %   mdl_fusion      - A trained fitglm logistic regression fusion model
    %
    % Outputs:
    %   p_fall - The continuous predicted probability of a Fall
    
    % Predict probabilities from Random Forests
    [~, score_eeg] = predict(mdl_eeg, n4sid_features);
    [~, score_imu] = predict(mdl_imu, imu_features);
    
    % Extract positive class probabilities (class '1')
    idx_1_eeg = find(strcmp(mdl_eeg.ClassNames, '1'));
    idx_1_imu = find(strcmp(mdl_imu.ClassNames, '1'));
    
    p_eeg = score_eeg(idx_1_eeg);
    p_imu = score_imu(idx_1_imu);
    
    % Feed into fusion layer
    X_fusion = [p_eeg, p_imu, heuristic_score];
    p_fall = predict(mdl_fusion, X_fusion);
end
