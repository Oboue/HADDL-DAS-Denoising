clc; clear; close all;

load eq-3.mat
dn = d1; % Clean data

% Get data dimensions
[n1, n2, n3] = size(dn);
dt = 0.004;
t = [0:n1-1] * dt;
x = [1:n2];

% Patch parameters
w1 = 32; % Patch height
w2 = 1;  % Patch width
w3 = 1;  % Patch depth (since this is 1D, keep it simple)
s1z = 1; 
s2z = 1;
s3z = 1;

% Define input size for the network
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

% Define the layers
layers = [
    featureInputLayer(inpsize, 'Name', 'input')
    fullyConnectedLayer(D1, 'Name', 'r0', 'WeightsInitializer', 'he')
    batchNormalizationLayer('Name', 'r0_bn')
    reluLayer('Name', 'r0_relu')
    
    fullyConnectedLayer(D2, 'Name', 'r1', 'WeightsInitializer', 'he')
    batchNormalizationLayer('Name', 'r1_bn')
    reluLayer('Name', 'r1_relu')
];

lgraph = layerGraph(layers);

% First Attention Mechanism
attentionLayers1 = [
    fullyConnectedLayer(D2, 'Name', 'attention_fc1', 'WeightsInitializer', 'he')
    reluLayer('Name', 'attention_relu1')
    fullyConnectedLayer(D2, 'Name', 'attention_fc2', 'WeightsInitializer', 'he')
    softmaxLayer('Name', 'attention_softmax1')
];
lgraph = addLayers(lgraph, attentionLayers1);
lgraph = connectLayers(lgraph, 'r1_relu', 'attention_fc1');

% Apply Attention to r1_relu output (Multiply by softmax1 output)
lgraph = addLayers(lgraph, multiplicationLayer(2, 'Name', 'apply_attention1'));
lgraph = connectLayers(lgraph, 'r1_relu', 'apply_attention1/in1');
lgraph = connectLayers(lgraph, 'attention_softmax1', 'apply_attention1/in2');

% Disconnect relu_1 and apply_attention_1
lgraph = disconnectLayers(lgraph, 'r1_relu', 'apply_attention1/in1'); % Disconnect

% Fully Connected Layers with Batch Normalization, ReLU, and Dropout
a0 = [
    fullyConnectedLayer(D3, 'Name', 'a0', 'WeightsInitializer', 'he')
    batchNormalizationLayer('Name', 'a0_bn')
    reluLayer('Name', 'a0_relu')
    dropoutLayer(0.9, 'Name', 'a0_dropout')
];

lgraph = addLayers(lgraph, a0);
lgraph = connectLayers(lgraph, 'apply_attention1', 'a0');  % Connect attention-modified output

% Second Attention Mechanism (Middle Layer)
attentionLayers2 = [
    fullyConnectedLayer(D3, 'Name', 'attention_fc3', 'WeightsInitializer', 'he')
    reluLayer('Name', 'attention_relu2')
    fullyConnectedLayer(D3, 'Name', 'attention_fc4', 'WeightsInitializer', 'he')
    softmaxLayer('Name', 'attention_softmax2')
];
lgraph = addLayers(lgraph, attentionLayers2);
lgraph = connectLayers(lgraph, 'a0_dropout', 'attention_fc3');  % Connect after first fully connected layer

% Apply Attention to a0_dropout output (Multiply by softmax2 output)
lgraph = addLayers(lgraph, multiplicationLayer(2, 'Name', 'apply_attention2'));
lgraph = connectLayers(lgraph, 'attention_softmax2', 'apply_attention2/in2');
lgraph = connectLayers(lgraph, 'a0_dropout', 'apply_attention2/in1');

% Disconnect a0_dropout and apply_attention2
lgraph = disconnectLayers(lgraph, 'a0_dropout', 'apply_attention2/in1');  % Disconnect

% Third Attention Mechanism (Near Output Layer)
attentionLayers3 = [
    fullyConnectedLayer(D4, 'Name', 'attention_fc5', 'WeightsInitializer', 'he')
    reluLayer('Name', 'attention_relu3')
    fullyConnectedLayer(D4, 'Name', 'attention_fc6', 'WeightsInitializer', 'he')
    softmaxLayer('Name', 'attention_softmax3')
];
lgraph = addLayers(lgraph, attentionLayers3);
lgraph = connectLayers(lgraph, 'a0_dropout', 'attention_fc5');  % Connect near output layer

% Apply Attention to a0_dropout output (Multiply by softmax3 output)
lgraph = addLayers(lgraph, multiplicationLayer(2, 'Name', 'apply_attention3'));


% **Here is the disconnection as per your request:**
lgraph = disconnectLayers(lgraph, 'r1_relu', 'apply_attention3/in1');  % Disconnect

% **Connect attention_softmax3 to apply_attention3**
lgraph = connectLayers(lgraph, 'attention_softmax3', 'apply_attention3/in2'); % Connect softmax3 to apply_attention3

% Concatenate Layer (Combine Attention Output with Original)
concat_1 = concatenationLayer(1, 2, 'Name', 'concat_1');
lgraph = addLayers(lgraph, concat_1);
lgraph = connectLayers(lgraph, 'r0_relu', 'concat_1/in1');
% lgraph = connectLayers(lgraph, 'apply_attention3', 'concat_1/in1');
lgraph = connectLayers(lgraph, 'apply_attention3', 'concat_1/in2');
% lgraph = connectLayers(lgraph, 'apply_attention3', 'concat_1');  % Connect directly after apply_attention_2

% Disconnect a0_dropout and concat_1
lgraph = disconnectLayers(lgraph, 'a0_dropout', 'concat_1/in2');  % Disconnect

% **Add New Fully Connected Layers with Batch Normalization, ReLU, and Dropout after apply_attention_2**
new_fc_layers = [
    fullyConnectedLayer(D5, 'Name', 'new_fc1', 'WeightsInitializer', 'he')
    batchNormalizationLayer('Name', 'new_fc1_bn')
    reluLayer('Name', 'new_fc1_relu')
    dropoutLayer(0.9, 'Name', 'new_fc1_dropout')
];

lgraph = addLayers(lgraph, new_fc_layers);

lgraph = connectLayers(lgraph, 'apply_attention2', 'new_fc1');  % Connect directly after apply_attention_2

outx = [
    fullyConnectedLayer(2 * inpsize, 'Name', 'outx')
    batchNormalizationLayer('Name', 'outx_bn')
    reluLayer('Name', 'outx_relu')
    dropoutLayer(0.05, 'Name', 'outx_drop')
    ];
lgraph = addLayers(lgraph, outx);
lgraph = connectLayers(lgraph, 'concat_1', 'outx');  % Connect concatenated output to CustomRegres

% Output Layer - Custom Regression Layer (Final Layer)
lambda1 = 0.0000001; % L1 regularization
lambda2 = 0.0000001; % L2 regularization
customLayer = CustomRegressionLayer('custom_regression', lambda1, lambda2);
lgraph = addLayers(lgraph, customLayer);
lgraph = connectLayers(lgraph, 'outx_drop', 'custom_regression');  % Connect concatenated output to CustomRegres
% lgraph = connectLayers(lgraph, 'concat_1', 'custom_regression');  % Connect concatenated output to CustomRegres

% **Disconnect the current connection from a0_dropout to attention_fc5**
lgraph = disconnectLayers(lgraph, 'a0_dropout', 'attention_fc5');

% **Connect new_fc1_dropout to attention_fc5**
lgraph = connectLayers(lgraph, 'new_fc1_dropout', 'attention_fc5');

% Plot and analyze the network
figure;
plot(lgraph);
analyzeNetwork(lgraph);

%%
% Define parameters
noiseLevel = 0.1;  % Adjust the noise level as necessary
w1 = 32; w2 = 1; w3 = 1; % Patch dimensions (w1 = 32, w2 = 1, w3 = 1)
s1z = 1; s2z = 1; s3z = 1; % Stride for patches

% Add Gaussian noise to the original data
noisy_d1 = dn + noiseLevel * randn(size(dn));

% Add high-frequency noise (sinusoidal + random)
highFreqNoise = noiseLevel * (sin(10 * (1:size(dn, 1))') * rand(1, size(dn, 2)));
noisy_highFreq = dn + highFreqNoise;

% Add horizontal (directional) noise
directionalNoise = noiseLevel * repmat(randn(1, size(dn, 2)), size(dn, 1), 1);
noisy_directional = dn + directionalNoise;

% Patching for DAE pre-training
X_noisy_gaussian = yc_patch3d(noisy_d1, 1, w1, w2, w3, s1z, s2z, s3z)';
X_noisy_highFreq = yc_patch3d(noisy_highFreq, 1, w1, w2, w3, s1z, s2z, s3z)';
X_noisy_directional = yc_patch3d(noisy_directional, 1, w1, w2, w3, s1z, s2z, s3z)';

% Clean patches remain the same (no noise)
X_clean = yc_patch3d(dn, 1, w1, w2, w3, s1z, s2z, s3z)';

% Combine all noisy patches
X_noisy = [X_noisy_gaussian; X_noisy_highFreq; X_noisy_directional];
X_clean_repeated = repmat(X_clean, 3, 1);  % Repeat clean patches for comparison

% Reshape the augmented data for DAE training
X_noisy_reshaped = reshape(X_noisy, [], size(X_noisy, 2));
X_clean_reshaped = reshape(X_clean_repeated, [], size(X_clean_repeated, 2));

% Normalize the augmented data
X_noisy_reshaped = X_noisy_reshaped / max(abs(X_noisy_reshaped(:)));
X_clean_reshaped = X_clean_reshaped / max(abs(X_clean_reshaped(:)));
%%
% Define training options for DAE
batchsize = 50000;
optionsDAE = trainingOptions('adam', ...
    'MaxEpochs', 1, ...  % Increase epochs for better training
    'InitialLearnRate', 1e-3, ...
    'MiniBatchSize', batchsize, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the DAE using augmented noisy data
netDAE = trainNetwork(X_noisy_reshaped, X_clean_reshaped, layers, optionsDAE);

% Predict Denoised output and unpatch
outDAE = DL_Predict(netDAE, X_noisy_reshaped, length(X_noisy_reshaped), batchsize, 1);
dn_denoised = yc_patch3d_inv(outDAE', 1, n1, n2, n3, w1, w2, w3, s1z, s2z, s3z);
%%
% === Fine-tuning for Final Task ===
% Patching for final training
X_final = yc_patch3d(dn_denoised, 1, w1, w2, w3, s1z, s2z, s3z);
X_final = X_final';

% Define the training options for the final task fine-tuning
optionsFineTune = trainingOptions('adam', ...
    'MaxEpochs', 1, ...              % Number of epochs
    'InitialLearnRate', 1e-3, ...     % Lower learning rate for fine-tuning
    'MiniBatchSize', batchsize, ...   % Mini-batch size
    'Shuffle', 'every-epoch', ...     % Shuffle the data every epoch
    'Verbose', false, ...             % Disable verbose output
    'Plots', 'training-progress');    % Show training progress

% Train the network for the final task
netFinal = trainNetwork(X_final, X_clean_reshaped, layers, optionsFineTune);

% Predict the final denoised output
outFinal = DL_Predict(netFinal, X_final, length(X_final), batchsize, 1);

% Unpatching to reconstruct the final output data
d_final = yc_patch3d_inv(outFinal', 1, n1, n2, n3, w1, w2, w3, s1z, s2z, s3z);
