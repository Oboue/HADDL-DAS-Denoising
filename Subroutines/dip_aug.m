clc; clear; close all;

load eq-3.mat

dn = d1;

[n1, n2, n3] = size(dn);
dt = 0.004;
t = [0:n1-1] * dt; 
x = [1:n2];

% Patch parameters
w1 = 32;
w2 = 1;
w3 = 1;
s1z = 1;
s2z = 1;
s3z = 1;
inpsize = w1 * w2 * w3;

% Define layer dimensions
D1 = 40;
D2 = ceil(D1 / 2);
D3 = ceil(D2 / 2);
D4 = ceil(D3 / 2);
D5 = ceil(D4 / 2);
D6 = ceil(D5 / 2);
D7 = ceil(D6 / 2);
D8 = ceil(D7 / 2);
D9 = ceil(D8 / 2);
D10 = ceil(D8 / 2);

% Define dimensions and parameters
D_query = 512;
D_key = 512;
D_value = 512;
numHeads = 8;  % Number of attention heads
D_head = D_query / numHeads;  % Dimension per head (assuming D_query = D_key = D_value)

% inpsize = 128;  % Input size
l1Factor = 0.01;  % L1 regularization factor for sparsity

% Define layers
layers = [
    featureInputLayer(inpsize, 'Name', 'input')
];

% Initialize the layer graph
lgraph = layerGraph(layers);

% First encoder block
r0 = fullyConnectedLayer(D1, 'WeightL2Factor', l1Factor, 'Name', 'r0');
r0_relu = reluLayer('Name', 'r0_relu');
r0_dropout = dropoutLayer(0.01, 'Name', 'r0_dropout');
lgraph = addLayers(lgraph, r0);
lgraph = addLayers(lgraph, r0_relu);
lgraph = addLayers(lgraph, r0_dropout);
lgraph = connectLayers(lgraph, 'input', 'r0');
lgraph = connectLayers(lgraph, 'r0', 'r0_relu');
lgraph = connectLayers(lgraph, 'r0_relu', 'r0_dropout');

% Second encoder block
r1 = fullyConnectedLayer(D2, 'WeightL2Factor', l1Factor, 'Name', 'r1');
r1_relu = reluLayer('Name', 'r1_relu');
r1_dropout = dropoutLayer(0.01, 'Name', 'r1_dropout');
lgraph = addLayers(lgraph, r1);
lgraph = addLayers(lgraph, r1_relu);
lgraph = addLayers(lgraph, r1_dropout);
lgraph = connectLayers(lgraph, 'r0_dropout', 'r1');
lgraph = connectLayers(lgraph, 'r1', 'r1_relu');
lgraph = connectLayers(lgraph, 'r1_relu', 'r1_dropout');

% Multi-head attention mechanism
queryLayer_multi = fullyConnectedLayer(D_query, 'WeightL2Factor', l1Factor, 'Name', 'query_fc_multi');
keyLayer_multi = fullyConnectedLayer(D_key, 'WeightL2Factor', l1Factor, 'Name', 'key_fc_multi');
valueLayer_multi = fullyConnectedLayer(D_value, 'WeightL2Factor', l1Factor, 'Name', 'value_fc_multi');
lgraph = addLayers(lgraph, queryLayer_multi);
lgraph = addLayers(lgraph, keyLayer_multi);
lgraph = addLayers(lgraph, valueLayer_multi);

% Connect query, key, and value to the multi-head layers
lgraph = connectLayers(lgraph, 'r1_dropout', 'query_fc_multi');
lgraph = connectLayers(lgraph, 'r1_dropout', 'key_fc_multi');
lgraph = connectLayers(lgraph, 'r1_dropout', 'value_fc_multi');

% For each attention head
for headIdx = 1:numHeads
    headName = num2str(headIdx);
    
    % Create per-head layers for query, key, and value
    query_fc_head = fullyConnectedLayer(D_head, 'WeightL2Factor', l1Factor, 'Name', ['query_fc_head_', headName]);
    key_fc_head = fullyConnectedLayer(D_head, 'WeightL2Factor', l1Factor, 'Name', ['key_fc_head_', headName]);
    value_fc_head = fullyConnectedLayer(D_head, 'WeightL2Factor', l1Factor, 'Name', ['value_fc_head_', headName]);
    lgraph = addLayers(lgraph, query_fc_head);
    lgraph = addLayers(lgraph, key_fc_head);
    lgraph = addLayers(lgraph, value_fc_head);
    
    % Connect multi-head layers to per-head layers
    lgraph = connectLayers(lgraph, 'query_fc_multi', ['query_fc_head_', headName]);
    lgraph = connectLayers(lgraph, 'key_fc_multi', ['key_fc_head_', headName]);
    lgraph = connectLayers(lgraph, 'value_fc_multi', ['value_fc_head_', headName]);
    
    % Attention mechanism for each head
    dotProductLayerHead = multiplicationLayer(2, 'Name', ['attention_dot_product_head_', headName]);
    softmaxLayerHead = softmaxLayer('Name', ['attention_softmax_head_', headName]);
    attentionOutputLayerHead = multiplicationLayer(2, 'Name', ['attention_output_head_', headName]);
    lgraph = addLayers(lgraph, dotProductLayerHead);
    lgraph = addLayers(lgraph, softmaxLayerHead);
    lgraph = addLayers(lgraph, attentionOutputLayerHead);
    
    % Connect query and key to dot product
    lgraph = connectLayers(lgraph, ['query_fc_head_', headName], ['attention_dot_product_head_', headName, '/in1']);
    lgraph = connectLayers(lgraph, ['key_fc_head_', headName], ['attention_dot_product_head_', headName, '/in2']);
    
    % Connect dot product to softmax and then to output
    lgraph = connectLayers(lgraph, ['attention_dot_product_head_', headName], ['attention_softmax_head_', headName]);
    lgraph = connectLayers(lgraph, ['attention_softmax_head_', headName], ['attention_output_head_', headName, '/in1']);
    lgraph = connectLayers(lgraph, ['value_fc_head_', headName], ['attention_output_head_', headName, '/in2']);
end

% Concatenate attention heads
concatAttention = concatenationLayer(1, numHeads, 'Name', 'concat_attention');
lgraph = addLayers(lgraph, concatAttention);
for headIdx = 1:numHeads
    headName = num2str(headIdx);
    lgraph = connectLayers(lgraph, ['attention_output_head_', headName], ['concat_attention/in', headName]);
end

% Softmax sharpening: Apply temperature scaling
temperature = 0.01;  % Lower temperature sharpens the softmax distribution
tempScaleLayer = TemperatureScalingLayer(temperature, 'temp_scaling');
lgraph = addLayers(lgraph, tempScaleLayer);
lgraph = connectLayers(lgraph, 'concat_attention', 'temp_scaling');

% Fully connected layers (Decoder)

% Define fc1 block
fc1 = fullyConnectedLayer(D3, 'WeightL2Factor', l1Factor, 'Name', 'fc1');
fc1_relu = reluLayer('Name', 'fc1_relu');
fc1_dropout = dropoutLayer(0.01, 'Name', 'fc1_dropout');
lgraph = addLayers(lgraph, fc1);
lgraph = addLayers(lgraph, fc1_relu);
lgraph = addLayers(lgraph, fc1_dropout);
lgraph = connectLayers(lgraph, 'temp_scaling', 'fc1');
lgraph = connectLayers(lgraph, 'fc1', 'fc1_relu');
lgraph = connectLayers(lgraph, 'fc1_relu', 'fc1_dropout');

% Define fc2 block
fc2 = fullyConnectedLayer(D4, 'WeightL2Factor', l1Factor, 'Name', 'fc2');
fc2_relu = reluLayer('Name', 'fc2_relu');
fc2_dropout = dropoutLayer(0.01, 'Name', 'fc2_dropout');
lgraph = addLayers(lgraph, fc2);
lgraph = addLayers(lgraph, fc2_relu);
lgraph = addLayers(lgraph, fc2_dropout);
lgraph = connectLayers(lgraph, 'fc1_dropout', 'fc2');
lgraph = connectLayers(lgraph, 'fc2', 'fc2_relu');
lgraph = connectLayers(lgraph, 'fc2_relu', 'fc2_dropout');

% Define subsequent fully connected, relu, and dropout layers
fc3 = fullyConnectedLayer(D5, 'WeightL2Factor', l1Factor, 'Name', 'fc3');
fc3_relu = reluLayer('Name', 'fc3_relu');
fc3_dropout = dropoutLayer(0.01, 'Name', 'fc3_dropout');
lgraph = addLayers(lgraph, fc3);
lgraph = addLayers(lgraph, fc3_relu);
lgraph = addLayers(lgraph, fc3_dropout);
lgraph = connectLayers(lgraph, 'fc2_dropout', 'fc3');
lgraph = connectLayers(lgraph, 'fc3', 'fc3_relu');
lgraph = connectLayers(lgraph, 'fc3_relu', 'fc3_dropout');

fc4 = fullyConnectedLayer(D6, 'WeightL2Factor', l1Factor, 'Name', 'fc4');
fc4_relu = reluLayer('Name', 'fc4_relu');
fc4_dropout = dropoutLayer(0.01, 'Name', 'fc4_dropout');
lgraph = addLayers(lgraph, fc4);
lgraph = addLayers(lgraph, fc4_relu);
lgraph = addLayers(lgraph, fc4_dropout);
lgraph = connectLayers(lgraph, 'fc3_dropout', 'fc4');
lgraph = connectLayers(lgraph, 'fc4', 'fc4_relu');
lgraph = connectLayers(lgraph, 'fc4_relu', 'fc4_dropout');

fc5 = fullyConnectedLayer(D7, 'WeightL2Factor', l1Factor, 'Name', 'fc5');
fc5_relu = reluLayer('Name', 'fc5_relu');
fc5_dropout = dropoutLayer(0.01, 'Name', 'fc5_dropout');
lgraph = addLayers(lgraph, fc5);
lgraph = addLayers(lgraph, fc5_relu);
lgraph = addLayers(lgraph, fc5_dropout);
lgraph = connectLayers(lgraph, 'fc4_dropout', 'fc5');
lgraph = connectLayers(lgraph, 'fc5', 'fc5_relu');
lgraph = connectLayers(lgraph, 'fc5_relu', 'fc5_dropout');

fc6 = fullyConnectedLayer(D8, 'WeightL2Factor', l1Factor, 'Name', 'fc6');
fc6_relu = reluLayer('Name', 'fc6_relu');
fc6_dropout = dropoutLayer(0.01, 'Name', 'fc6_dropout');
lgraph = addLayers(lgraph, fc6);
lgraph = addLayers(lgraph, fc6_relu);
lgraph = addLayers(lgraph, fc6_dropout);
lgraph = connectLayers(lgraph, 'fc5_dropout', 'fc6');
lgraph = connectLayers(lgraph, 'fc6', 'fc6_relu');
lgraph = connectLayers(lgraph, 'fc6_relu', 'fc6_dropout');

% Add the missing fc7 layer and its associated dropout and relu layers

% Define fc7 block
% First, define and add the missing layers
% Define fc7 block
fc7 = fullyConnectedLayer(D9, 'WeightL2Factor', l1Factor, 'Name', 'fc7');
fc7_relu = reluLayer('Name', 'fc7_relu');
fc7_dropout = dropoutLayer(0.01, 'Name', 'fc7_dropout');
lgraph = addLayers(lgraph, fc7);
lgraph = addLayers(lgraph, fc7_relu);
lgraph = addLayers(lgraph, fc7_dropout);
lgraph = connectLayers(lgraph, 'fc6_dropout', 'fc7');
lgraph = connectLayers(lgraph, 'fc7', 'fc7_relu');
lgraph = connectLayers(lgraph, 'fc7_relu', 'fc7_dropout');

% Now, define the concatenation layer
concatLayer = concatenationLayer(1, 2, 'Name', 'concat_output');  % 2 inputs for concatenation
lgraph = addLayers(lgraph, concatLayer);

% Connect the layers to be concatenated
% Now, define the concatenation layer
concatLayer = concatenationLayer(1, 2, 'Name', 'concat_output');  % 2 inputs for concatenation
% lgraph = addLayers(lgraph, concatLayer);

% Connect the layers to be concatenated
lgraph = connectLayers(lgraph, 'r0_dropout', 'concat_output/in1');  % Connect r0_dropout (earlier encoder) to the first input
lgraph = connectLayers(lgraph, 'fc7_dropout', 'concat_output/in2');  % Connect fc7_dropout (later fully connected layer) to the second input

% Final output layer
output = fullyConnectedLayer(inpsize, 'WeightL2Factor', l1Factor, 'Name', 'output');  % Output layer with inpsize
lgraph = addLayers(lgraph, output);
lgraph = connectLayers(lgraph, 'concat_output', 'output');  % Connect concatenated output to the fully connected output layer

% Now, add the custom regression layer to apply L1 and L2 regularization
lambda1 = 0.0000001; % L1 regularization
lambda2 = 0.0000001; % L2 regularization

% Assuming your custom regression layer is implemented as 'CustomRegressionLayer'
customRegressionLayer = CustomRegressionLayer('custom_regression', lambda1, lambda2);
lgraph = addLayers(lgraph, customRegressionLayer);

% Connect the output layer to the regression layer
lgraph = connectLayers(lgraph, 'output', 'custom_regression');

% Visualize the final network structure
figure;
plot(lgraph);
analyzeNetwork(lgraph);

% === Denoising Autoencoder (DAE) Pre-training with Multiple Noise Augmentation ===
%%
% Set up the augmentation parameters
noiseLevel = 0.1;  % Base level for Gaussian noise
numAugmentations = 5;  % Number of noisy augmentations for data diversity

% Arrays to store augmented data patches
X_noisy_augmented = [];
X_clean_augmented = [];

% Generate augmented versions of data with multiple noise types
for i = 1:numAugmentations
    % 1. Add Gaussian noise
    noisy_gaussian = dn + noiseLevel * randn(size(dn));
    
    % 2. Add high-frequency noise
    high_freq_noise = noiseLevel * (sin(10 * (1:n1)') * rand(1, n2)); % Example high-freq noise
    noisy_high_freq = dn + high_freq_noise;

    % 3. Add directional (horizontal) noise
    directional_noise = noiseLevel * repmat(randn(1, n2), n1, 1); % Horizontal band noise
    noisy_directional = dn + directional_noise;

    % Extract patches from each noisy version
    X_noisy_gaussian = yc_patch3d(noisy_gaussian, 1, w1, w2, w3, s1z, s2z, s3z)';
    X_noisy_high_freq = yc_patch3d(noisy_high_freq, 1, w1, w2, w3, s1z, s2z, s3z)';
    X_noisy_directional = yc_patch3d(noisy_directional, 1, w1, w2, w3, s1z, s2z, s3z)';
    
    % Clean patches remain the same
    X_clean = yc_patch3d(dn, 1, w1, w2, w3, s1z, s2z, s3z)';
    
    % Accumulate augmented patches for training
    X_noisy_augmented = [X_noisy_augmented; X_noisy_gaussian; X_noisy_high_freq; X_noisy_directional];
    X_clean_augmented = [X_clean_augmented; X_clean; X_clean; X_clean];
end

% Reshape the augmented data for DAE training
X_noisy_reshaped = reshape(X_noisy_augmented, [], size(X_noisy_augmented, 2));
X_clean_reshaped = reshape(X_clean_augmented, [], size(X_clean_augmented, 2));

% Define training options for DAE
batchsize = 2050;
optionsDAE = trainingOptions('adam', ...
    'MaxEpochs', 1, ...
    'InitialLearnRate', 1e-3, ...
    'MiniBatchSize', batchsize, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the DAE using augmented noisy data
netDAE = trainNetwork(X_noisy_reshaped, X_clean_reshaped, lgraph, optionsDAE);

% Predict Denoised output and unpatch
outDAE = DL_Predict(netDAE, X_noisy_reshaped, length(X_noisy_reshaped), batchsize, 1);
dn_denoised = yc_patch3d_inv(outDAE', 1, n1, n2, n3, w1, w2, w3, s1z, s2z, s3z);

% === Fine-tuning for Final Task ===
% Patching for final training
X_final = yc_patch3d(dn_denoised, 1, w1, w2, w3, s1z, s2z, s3z);
X_final = X_final';

% Define the training options for the final task fine-tuning
optionsFineTune = trainingOptions('adam', ...
    'MaxEpochs',1, ...              % Number of epochs
    'InitialLearnRate', 1e-3, ...     % Lower learning rate for fine-tuning
    'MiniBatchSize', batchsize, ...   % Mini-batch size
    'Shuffle', 'every-epoch', ...     % Shuffle the data every epoch
    'Verbose', false, ...             % Disable verbose output
    'Plots', 'training-progress');    % Show training progress

% Train the network for th e final task
netFinal = trainNetwork(X_final, X_clean_reshaped, lgraph, optionsFineTune);

% Predict the final denoised output
outFinal = DL_Predict(netFinal, X_final, length(X_final), batchsize, 1);

% Unpatching to reconstruct the final output data
d_final = yc_patch3d_inv(outFinal', 1, n1, n2, n3, w1, w2, w3, s1z, s2z, s3z);
%%
% Display result
figure('units', 'normalized', 'Position', [0.0 0.0 0.5, 1]);
subplot(2, 2, 1);
imagesc(x, t, d1); colormap(amf_seis); caxis([-1 1] * 0.3); xlabel('x(m)'); ylabel('t(s)');
caxis([-25 25]);
colormap(seis);
title('Clean data');

subplot(2, 2, 2);
imagesc(x, t, dn); colormap(amf_seis); caxis([-1 1] * 0.3); xlabel('x(m)'); ylabel('t(s)');
caxis([-25 25]);
colormap(seis);
title('Noisy data');
 
% subplot(2, 2, 3);
% imagesc(x, t, dn_denoised); colormap(amf_seis); caxis([-1 1] * 0.3); xlabel('x(m)'); ylabel('t(s)');
% caxis([-25 25]);
% colormap(seis);
% title('Denoised data (DAE)');
figure;
subplot(1, 2, 1);
imagesc(x, t, d_final); colormap(amf_seis); caxis([-1 1] * 0.3); xlabel('x(m)'); ylabel('t(s)');
caxis([-25 25]);
colormap(seis);
title('Final denoised data');
% figure('units', 'normalized', 'Position', [0.0 0.0 0.5, 1]);
subplot(1, 2, 2);
imagesc(x, t, dn-d_final); colormap(amf_seis); caxis([-1 1] * 0.3); xlabel('x(m)'); ylabel('t(s)');
caxis([-25 25]);
colormap(seis);
title('Clean data');


% Define the encoder-decoder network (same as the original code)

% Training loop with data augmentation
% niter = 1; 
% for i = 1:niter
%     if i == 1
%         d_prev = dn;
%     else
%         d_prev = d1_denoised; 
%     end
%     %% Patch Generation with Augmentation
%     X = yc_patch3d(dn, 1, w1, w2, w3, s1z, s2z, s3z);
%     X = X';
%     le = length(X);
% 
%     % Apply data augmentation
%     augmentedPatches = [];
%     for j = 1:le
%         % Select the original patch
%         patch = X(j, :);
% 
%         % Augment the patch
%         augmentedPatches = [augmentedPatches; augmentData(patch)];
%     end
% 
%     % Flatten the augmented patches into a single array
%     augmentedPatches = cell2mat(augmentedPatches);
% 
%     %% Selecting patches based on variance (same as the original code)
%     v = var(augmentedPatches');
%     [ord, indx] = sort(v);
%     lex = round(length(indx) * 0.25);
% 
%     % Train DenseNet with augmented patches
%     net = trainNetwork(augmentedPatches(indx(lex:end), :), augmentedPatches(indx(lex:end), :), lgraph, options);
% 
%     % Predict DenseNet output
%     outDN = DL_Predict(net, augmentedPatches, le, batchsize, cc);
% 
%     % UnPatching
%     d1_denoised = yc_patch3d_inv(X', 1, n1, n2, n3, w1, w2, w3, s1z, s2z, s3z); 
% 
%     % Plotting results
%     figure;
%     imagesc(d1_denoised);
%     title(sprintf('Denoised Data at Iteration %d', i));
%     ylabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
%     xlabel('Panel', 'FontSize', 12, 'FontWeight', 'bold');
%     colormap(seis);
%     caxis([-25 25]);
%     set(gca, 'LineWidth', 2, 'FontSize', 12, 'FontWeight', 'bold');
% 
%     figure;
%     imagesc(dn - d1_denoised);
%     title(sprintf('Removed Noise at Iteration %d', i));
%     ylabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
%     xlabel('Panel', 'FontSize', 12, 'FontWeight', 'bold');
%     colormap(seis);
%     caxis([-25 25]);
%     set(gca, 'LineWidth', 2, 'FontSize', 12, 'FontWeight', 'bold');   
% 
%     fprintf('Completed iteration %d\n', i);
% end
