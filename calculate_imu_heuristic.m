function heuristic_score = calculate_imu_heuristic(acc_magnitude)
    % calculate_imu_heuristic Computes a continuous free-fall heuristic score.
    % 
    % Inputs:
    %   acc_magnitude - 1D array of combined acceleration magnitude
    %
    % Outputs:
    %   heuristic_score - A double indicating likelihood of free-fall (higher = more likely).
    %                     Formula: max(0, 1.0 - min(acc_magnitude))
    
    heuristic_score = max(abs(acc_magnitude));
end
