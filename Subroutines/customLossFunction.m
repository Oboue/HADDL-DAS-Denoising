function loss = customLossFunction(YTrue, YPred, lambda)
    % Compute the standard loss (e.g., mean squared error)
    standardLoss = mean((YTrue - YPred).^2, 'all');

    % Apply the median filter to the predicted output
    filteredPred = amf_mf(YPred, 7, 1, 2); % Adjust parameters as needed

    % Compute the difference between the filtered and original predictions
    filterLoss = mean((YPred - filteredPred).^2, 'all');

    % Combine standard loss with filter loss using a regularization parameter lambda
    loss = standardLoss + lambda * filterLoss;
end
