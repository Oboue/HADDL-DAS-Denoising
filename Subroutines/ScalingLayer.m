classdef ScalingLayer < nnet.layer.Layer
    properties
        ScaleFactor
    end
    
    methods
        function layer = ScalingLayer(scaleFactor, name)
            % Set layer name and scale factor
            layer.Name = name;
            layer.ScaleFactor = scaleFactor;
        end
        
        function Z = predict(layer, X)
            % Multiply input by the scaling factor
            Z = layer.ScaleFactor * X;
        end
    end
end
