function pred_label = predict_fall_wrapper(eeg_norm, sys_order, rf_model)
    % predict_fall_wrapper A helper function to be run on a background worker.
    % It wraps the feature extraction and prediction into a single call.
    %
    % Inputs:
    %   eeg_norm   - 1D array of normalized EEG data
    %   sys_order  - N4SID system order
    %   rf_model   - Trained fitcensemble model
    %
    % Outputs:
    %   pred_label - The predicted class ('1' for Fall, '0' for Safe)
    
    % 1. Extract Features
    feat = extract_features(eeg_norm, sys_order);
    
    % 2. Predict Fall
    pred_label = predict_fall(feat, rf_model);
end
