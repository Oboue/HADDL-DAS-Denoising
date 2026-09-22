classdef CustomRegressionLayer < nnet.layer.RegressionLayer
    properties
        Lambda1 % L1 regularization parameter
        Lambda2 % L2 regularization parameter
    end
    
    methods
        function layer = CustomRegressionLayer(name, lambda1, lambda2)
            % Set layer name and regularization parameters
            layer.Name = name;
            layer.Lambda1 = lambda1;
            layer.Lambda2 = lambda2;
            layer.Description = 'Custom regression layer with L1 and L2 regularization';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % Check if Y and T are numeric arrays or dlarray objects
            if isa(Y, 'dlarray') || isnumeric(Y)
                Y = dlarray(Y);
                T = dlarray(T);
                
                % Specify data format
                if Y.ndims == 4
                    % For 4D arrays, use 'SSCB' (Spatial, Spatial, Channel, Batch)
                    Y = dlarray(Y, 'SSCB');
                    T = dlarray(T, 'SSCB');
                elseif Y.ndims == 2
                    % For 2D arrays, use 'CB' (Channel, Batch)
                    Y = dlarray(Y, 'CB');
                    T = dlarray(T, 'CB');
                else
                    error('Unsupported data format for Y and T');
                end
            end
            
            % Mean squared error loss
            mseLoss = mse(Y, T);
            
            % L1 regularization term
            l1Reg = layer.Lambda1 * sum(abs(Y(:)));
            
            % L2 regularization term
            l2Reg = layer.Lambda2 * sum(Y(:).^2);
            
            % Final loss with L1 and L2 regularization
            loss = mseLoss + l1Reg + l2Reg;
        end
    end
end
