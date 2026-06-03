function my_feature = extract_features(Data, sys_order)
    % EXTRACT_FEATURES Extracts n4sid state-space feature mapping.
    % 
    % Inputs:
    %   Data      - Normalized 1D EEG epoch array.
    %   sys_order - The order of the state-space model to fit (e.g., 3).
    %
    % Outputs:
    %   my_feature - A 1D array containing system matrix A (flattened), C, and K'.
    
    % Ts = 1 ms sampling time
    dy_sys = iddata(Data(:), [], 1, 'TimeUnit', 'milliseconds', 'Tstart', 0);
    sys_ss = n4sid(dy_sys, sys_order, 'Display', 'off');
    
    sysA = [];
    for k = 1:sys_order
        sysA = [sysA, sys_ss.A(k, :)];
    end
    
    % Classic Time-Series Features
    eeg_var = var(Data);
    eeg_energy = sum(Data.^2);
    
    % Hjorth Parameters
    dy = diff(Data);
    ddy = diff(dy);
    
    var_y = var(Data);
    var_dy = var(dy);
    var_ddy = var(ddy);
    
    hjorth_activity = var_y;
    if var_y > 0
        hjorth_mobility = sqrt(var_dy / var_y);
    else
        hjorth_mobility = 0;
    end
    
    if var_dy > 0
        hjorth_complexity = sqrt(var_ddy / var_dy) / (hjorth_mobility + eps);
    else
        hjorth_complexity = 0;
    end
    
    my_feature = [sysA, sys_ss.C, sys_ss.K', eeg_var, eeg_energy, hjorth_activity, hjorth_mobility, hjorth_complexity];
end
