function totalLoss = customLoss(Y_pred, Y_true, lambda)
    % Task Loss (e.g., Mean Squared Error)
    taskLoss = mean((Y_pred - Y_true).^2);

    % Apply Median Filter to the Output
    nfw = 8; % Window size for median filtering
    ifb = 1; % Use padded boundary
    axis = 2; % Apply along the second axis
    Y_filtered = amf_mf(Y_pred, nfw, ifb, axis); % Apply median filter to predictions

    % Median Filter Loss (Difference between original output and filtered output)
    medianFilterLoss = mean((Y_pred - Y_filtered).^2);

    % Total Loss = Task Loss + lambda * Median Filter Loss
    totalLoss = taskLoss + lambda * medianFilterLoss;
end
