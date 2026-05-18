function acc_epoch_norm = normalized_acc(acc_epoch)
    % NORMALIZED_ACC Applies Savitzky-Golay filtering and calculates magnitude.
    %
    % Inputs:
    %   acc_epoch - A 3xN matrix containing X, Y, and Z accelerometer signals.
    %
    % Outputs:
    %   acc_epoch_norm - A 1xN array containing the filtered signal magnitude.

    acc_epoch_x = sgolayfilt(acc_epoch(1,:), 3, 21);
    acc_epoch_y = sgolayfilt(acc_epoch(2,:), 3, 21);
    acc_epoch_z = sgolayfilt(acc_epoch(3,:), 3, 21);
    acc_epoch_norm = sqrt(acc_epoch_x.^2 + acc_epoch_y.^2 + acc_epoch_z.^2);
end
