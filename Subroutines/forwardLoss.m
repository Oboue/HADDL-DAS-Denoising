function loss = forwardLoss(layer, Y, T)
    % Ensure Y and T are in the correct format
    if isnumeric(Y)
        Y = dlarray(Y, 'SSCB'); % Assume spatial, spatial, channel, batch format
    end
    
    if isnumeric(T)
        T = dlarray(T, 'SSCB'); % Same format as Y
    end

    % Calculate the standard loss
    loss = mse(Y, T);

    % Apply L1 regularization
    if layer.L1Factor > 0
        l1Loss = sum(abs(Y(:)));
        loss = loss + layer.L1Factor * l1Loss;
    end

    % Apply L2 regularization
    if layer.L2Factor > 0
        l2Loss = sum(Y(:).^2);
        loss = loss + layer.L2Factor * l2Loss;
    end
end
