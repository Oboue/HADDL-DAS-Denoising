function attentionBlock = createSelfAttentionBlock(name, featureDim, attentionDim)
    % name: Base name for the layers
    % featureDim: Dimension of the input features
    % attentionDim: Dimension used for computing queries and keys

    attentionBlock = [
        fullyConnectedLayer(attentionDim, 'Name', [name '_query_fc'], 'WeightsInitializer', 'he')
        reluLayer('Name', [name '_query_relu'])
        
        fullyConnectedLayer(attentionDim, 'Name', [name '_key_fc'], 'WeightsInitializer', 'he')
        reluLayer('Name', [name '_key_relu'])
        
        fullyConnectedLayer(featureDim, 'Name', [name '_value_fc'], 'WeightsInitializer', 'he')
        
        % Compute attention scores (dot product of query and key)
        multiplicationLayer( [attentionDim, attentionDim], 'Name', [name '_attn_mult'])
        additionLayer(1, 'Name', [name '_attn_add'])
        
        softmaxLayer('Name', [name '_attn_softmax'])
        
        % Multiply attention weights with values
        elementwiseProductLayer('Name', [name '_attn_product'])
    ];
end
