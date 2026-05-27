function prediction_label = predict_fall(n4sid_features, trained_model)
    % predict_fall Uses the trained ensemble model to predict a fall
    %
    % Inputs:
    %   n4sid_features - 1D array of extracted N4SID state-space matrices
    %   trained_model  - A trained fitcensemble classification model
    %
    % Outputs:
    %   prediction_label - The predicted class ('1' for Fall, '0' for Safe)
    
    % MATLAB's predict function returns a cell array for ensemble categorical models
    pred_cell = predict(trained_model, n4sid_features);
    prediction_label = pred_cell{1};
end
