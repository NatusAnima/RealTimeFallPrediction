function p_fall = predict_fall_wrapper(eeg_norm, acc_epoch, heuristic_score, sys_order, mdl_eeg, mdl_imu, mdl_fusion)
    % predict_fall_wrapper A helper function to be run on a background worker.
    % It wraps the feature extraction and prediction into a single call.
    %
    % Inputs:
    %   eeg_norm        - 1D array of normalized EEG data
    %   acc_epoch       - 3xN array of acceleration data
    %   heuristic_score - Float indicating likelihood of free-fall
    %   sys_order       - N4SID system order
    %   mdl_eeg, mdl_imu, mdl_fusion - The trained models
    %
    % Outputs:
    %   p_fall - The continuous predicted probability of a Fall
    
    % 1. Extract Features
    feat_eeg = extract_features(eeg_norm, sys_order);
    feat_imu = extract_imu_features(acc_epoch);
    
    % 2. Predict Fall
    p_fall = predict_fall(feat_eeg, feat_imu, heuristic_score, mdl_eeg, mdl_imu, mdl_fusion);
end
