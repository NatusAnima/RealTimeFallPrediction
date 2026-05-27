function [gate_passed, max_acc] = evaluate_acc_gate(acc_magnitude, acc_threshold)
    % evaluate_acc_gate Checks if the physical movement exceeds the threshold
    %
    % Inputs:
    %   acc_magnitude - 1D array of combined acceleration magnitude
    %   acc_threshold - Double representing the threshold to pass
    %
    % Outputs:
    %   gate_passed - Boolean indicating if the threshold was crossed
    %   max_acc     - The maximum acceleration value observed in the window
    
    max_acc = max(acc_magnitude);
    gate_passed = max_acc >= acc_threshold;
end
