classdef CustomLossLayer < nnet.layer.ClassificationLayer
    properties
        Lambda
    end

    methods
        function layer = CustomLossLayer(lambda, name)
            % Set the properties and name of the layer
            layer.Name = name;
            layer.Description = 'Custom loss layer';
            layer.Lambda = lambda;
        end

        function loss = forwardLoss(layer, YTrue, YPred)
            % Compute the loss using the custom loss function
            loss = customLossFunction(YTrue, YPred, layer.Lambda);
        end
    end
end
