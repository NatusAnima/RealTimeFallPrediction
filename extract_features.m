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
    
    my_feature = [sysA, sys_ss.C, sys_ss.K'];
end
