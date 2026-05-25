function [loss, grad] = customLossWithMedianFilter(predictions, targets, lambda, nfw, ifb, axis)
    % Compute the standard mean squared error loss
    mseLoss = mean((predictions - targets).^2, 'all');
    
    % Apply median filter to predictions
    filteredPredictions = amf_mf(predictions, nfw, ifb, axis);
    
    % Compute the median filter loss
    medianFilterLoss = mean((predictions - filteredPredictions).^2, 'all');
    
    % Combine the losses with a regularization parameter lambda
    loss = mseLoss + lambda * medianFilterLoss;
    
    % Compute gradients (for simplicity, this is just a placeholder)
    grad = []; % Calculate gradients if needed for your training routine
end
