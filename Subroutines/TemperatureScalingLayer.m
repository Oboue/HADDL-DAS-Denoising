classdef TemperatureScalingLayer < nnet.layer.Layer
    % Temperature scaling for sharpening the softmax output
    
    properties
        temperature
    end
    
    methods
        function layer = TemperatureScalingLayer(temperature, name)
            % Constructor
            layer.Name = name;
            layer.temperature = temperature;  % Set the temperature value
        end
        
        function Z = predict(layer, X)
            % Apply temperature scaling
            Z = X / layer.temperature;  % Scale the logits by temperature
        end
        
        function Z = forward(layer, X)
            % Same as predict for this case, apply temperature scaling
            Z = layer.predict(X);
        end
    end
end
