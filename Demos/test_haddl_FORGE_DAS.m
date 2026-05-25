% =========================================================================
% SCRIPT: test_haddl_FORGE_DAS.m
% -------------------------------------------------------------------------
% Author:      Oboué et al.
% Date:        2026
% Affiliation: Zhejiang University 
% Reference:   Oboué et al., "A Hybrid Attention-Driven Deep Learning 
%              Framework for Denoising DAS Data", Geophysics, 2026.
% -------------------------------------------------------------------------
% Description:
%   This script is dedicated to processing the FORGE (Frontier Observatory 
%   for Research in Geothermal Energy) DAS dataset as presented in the 
%   manuscript. 
%
%   It implements the HADDL (Hybrid Attention-Driven Deep Learning) 
%   framework to handle complex geothermal noise environments and recover 
%   low-amplitude microseismic events.
%
% Copyright (c) 2026. All rights reserved.
% =========================================================================

clc; clear; close all;

%% SECTION 1: PATH MANAGEMENT & ENVIRONMENT SETUP
% -------------------------------------------------------------------------
fprintf('--- Initializing FORGE Dataset Processing ---\n');

addpath(genpath('../subroutines')); % Helper functions
addpath(genpath('../Field_data'));   

% Ensures reproducibility for research validation and peer review
rng(42, 'twister'); 

%% SECTION 2: DATA LOADING & DOMAIN SPECIFICATION
% -------------------------------------------------------------------------
% Loading the FORGE project DAS seismic data wavefield.
% -------------------------------------------------------------------------
fprintf('--- Loading FORGE Seismic Data ---\n');

if exist('forgeDAS_data.mat', 'file')
    load('forgeDAS_data.mat');
    % The raw input wavefield is stored in variable 'd1'
    dn = d1; 
    fprintf('-> Success: FORGE data loaded.\n');
else
    error('File forgeDAS_data.mat not found. Please check your path.');
end

% 3. Dimensional Analysis and Physical Sampling
[n1, n2, n3] = size(dn);
dt = 0.004;         % Time sampling interval (4ms)
t = (0:n1-1) * dt;  % Temporal axis vector (s)
x = (1:n2);         % Spatial axis (Trace/Channel index)

fprintf('-> Input dimensions: %d time samples x %d channels\n', n1, n2);
fprintf('-> Ready for HADDL iterative denoising.\n');

%% --- 3. GEOMETRIC PATCHING PARAMETERS ---
% Defining 3D patch dimensions for local feature extraction
w1 = 5;             % Temporal window
w2 = 5;             % Spatial window
w3 = 1;             % Cross-line window (if applicable)

% Stride/Step parameters for overlapping patches
s1z = 1; 
s2z = 1;
s3z = 1;

% Flattened input size for the network
inpsize = w1 * w2 * w3;
fprintf('System ready. Architecture defined for input size: %d\n', inpsize);
%% --- 4. NETWORK ARCHITECTURE DIMENSIONS (BOTTLENECK) ---
% Hierarchical layer sizing for the residual framework
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
numHeads = 64;  % Number of attention heads
D_head = D_query / numHeads;  % Dimension per head (assuming D_query = D_key = D_value)

% inpsize = 128;  % Input size
l1Factor = 0.00001;  % L1 regularization factor for sparsity

% Define layers
layers = [
    featureInputLayer(inpsize, 'Name', 'input')
];

% Initialize the layer graph
lgraph = layerGraph(layers);

% First encoder block
r0 = fullyConnectedLayer(D1, 'WeightL2Factor', l1Factor, 'Name', 'r0');
r0_relu = reluLayer('Name', 'r0_relu');
r0_dropout = dropoutLayer(0.00001, 'Name', 'r0_dropout');
lgraph = addLayers(lgraph, r0);
lgraph = addLayers(lgraph, r0_relu);
lgraph = addLayers(lgraph, r0_dropout);
lgraph = connectLayers(lgraph, 'input', 'r0');
lgraph = connectLayers(lgraph, 'r0', 'r0_relu');
lgraph = connectLayers(lgraph, 'r0_relu', 'r0_dropout');

% Second encoder block
r1 = fullyConnectedLayer(D2, 'WeightL2Factor', l1Factor, 'Name', 'r1');
r1_relu = reluLayer('Name', 'r1_relu');
r1_dropout = dropoutLayer(0.00001, 'Name', 'r1_dropout');
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
temperature = 0.0001;  % Lower temperature sharpens the softmax distribution
tempScaleLayer = TemperatureScalingLayer(temperature, 'temp_scaling');
lgraph = addLayers(lgraph, tempScaleLayer);
lgraph = connectLayers(lgraph, 'concat_attention', 'temp_scaling');

% Fully connected layers (Decoder)

% Define fc1 block
fc1 = fullyConnectedLayer(D3, 'WeightL2Factor', l1Factor, 'Name', 'fc1');
fc1_relu = reluLayer('Name', 'fc1_relu');
fc1_dropout = dropoutLayer(0.00001, 'Name', 'fc1_dropout');
lgraph = addLayers(lgraph, fc1);
lgraph = addLayers(lgraph, fc1_relu);
lgraph = addLayers(lgraph, fc1_dropout);
lgraph = connectLayers(lgraph, 'temp_scaling', 'fc1');
lgraph = connectLayers(lgraph, 'fc1', 'fc1_relu');
lgraph = connectLayers(lgraph, 'fc1_relu', 'fc1_dropout');

% Define fc2 block
fc2 = fullyConnectedLayer(D4, 'WeightL2Factor', l1Factor, 'Name', 'fc2');
fc2_relu = reluLayer('Name', 'fc2_relu');
fc2_dropout = dropoutLayer(0.00001, 'Name', 'fc2_dropout');
lgraph = addLayers(lgraph, fc2);
lgraph = addLayers(lgraph, fc2_relu);
lgraph = addLayers(lgraph, fc2_dropout);
lgraph = connectLayers(lgraph, 'fc1_dropout', 'fc2');
lgraph = connectLayers(lgraph, 'fc2', 'fc2_relu');
lgraph = connectLayers(lgraph, 'fc2_relu', 'fc2_dropout');

% Define subsequent fully connected, relu, and dropout layers
fc3 = fullyConnectedLayer(D5, 'WeightL2Factor', l1Factor, 'Name', 'fc3');
fc3_relu = reluLayer('Name', 'fc3_relu');
fc3_dropout = dropoutLayer(0.00001, 'Name', 'fc3_dropout');
lgraph = addLayers(lgraph, fc3);
lgraph = addLayers(lgraph, fc3_relu);
lgraph = addLayers(lgraph, fc3_dropout);
lgraph = connectLayers(lgraph, 'fc2_dropout', 'fc3');
lgraph = connectLayers(lgraph, 'fc3', 'fc3_relu');
lgraph = connectLayers(lgraph, 'fc3_relu', 'fc3_dropout');

fc4 = fullyConnectedLayer(D6, 'WeightL2Factor', l1Factor, 'Name', 'fc4');
fc4_relu = reluLayer('Name', 'fc4_relu');
fc4_dropout = dropoutLayer(0.00001, 'Name', 'fc4_dropout');
lgraph = addLayers(lgraph, fc4);
lgraph = addLayers(lgraph, fc4_relu);
lgraph = addLayers(lgraph, fc4_dropout);
lgraph = connectLayers(lgraph, 'fc3_dropout', 'fc4');
lgraph = connectLayers(lgraph, 'fc4', 'fc4_relu');
lgraph = connectLayers(lgraph, 'fc4_relu', 'fc4_dropout');

fc5 = fullyConnectedLayer(D7, 'WeightL2Factor', l1Factor, 'Name', 'fc5');
fc5_relu = reluLayer('Name', 'fc5_relu');
fc5_dropout = dropoutLayer(0.00001, 'Name', 'fc5_dropout');
lgraph = addLayers(lgraph, fc5);
lgraph = addLayers(lgraph, fc5_relu);
lgraph = addLayers(lgraph, fc5_dropout);
lgraph = connectLayers(lgraph, 'fc4_dropout', 'fc5');
lgraph = connectLayers(lgraph, 'fc5', 'fc5_relu');
lgraph = connectLayers(lgraph, 'fc5_relu', 'fc5_dropout');

fc6 = fullyConnectedLayer(D8, 'WeightL2Factor', l1Factor, 'Name', 'fc6');
fc6_relu = reluLayer('Name', 'fc6_relu');
fc6_dropout = dropoutLayer(0.00001, 'Name', 'fc6_dropout');
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
fc7_dropout = dropoutLayer(0.00001, 'Name', 'fc7_dropout');
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
lgraph = connectLayers(lgraph, 'r0_dropout', 'concat_output/in1');  % Connect r0_dropout (earlier encoder) to the first input
lgraph = connectLayers(lgraph, 'fc7_dropout', 'concat_output/in2');  % Connect fc7_dropout (later fully connected layer) to the second input

% Now connect the concatenated output to the final output layer
output = fullyConnectedLayer(inpsize, 'WeightL2Factor', l1Factor, 'Name', 'output');  % Output layer with 16 units
lgraph = addLayers(lgraph, output);
lgraph = connectLayers(lgraph, 'concat_output', 'output');  % Connect concatenated output to the fully connected output layer

% Output layer for regression
regressionLayer = regressionLayer('Name', 'regression_output');
lgraph = addLayers(lgraph, regressionLayer);

% Connect the output to the regression layer
lgraph = connectLayers(lgraph, 'output', 'regression_output');

% Finalize the network structure
figure;
plot(lgraph);
analyzeNetwork(lgraph);
%%
rng(42, 'twister'); % ensures repeatability of random operations%% numHeads = 2;  % Number of attention heads

%% --- 5. OUTPUT DIRECTORY SETUP ---
% Automatic folder creation for high-fidelity result storage
resultsFolder = 'Output_Field_FORGE_DAS_HADDL';
if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

%%
niter = 1; 
for i = 1:niter
    % Determine previous data for the current iteration
    if i == 1
        d_prev = dn; % Use the original data for the first itceration
    else
        d_prev = d1_denoised; % Use the denoised output from the previous iteration
    end
    %% Patch Generation
    X = haddl_patch3d(d_prev, 1, w1, w2, w3, s1z, s2z, s3z);
    X = double(X');  % Convert to double
    %% Apply Non-Local Means filtering only during the first iteration
    if i == 1
        minPatchSize = 21;  % Minimum size for imnlmfilt
        filterStrength = 0.01;  % Degree of smoothing
        for j = 1:size(X, 1)
            patch = reshape(X(j, :), [w1, w2, w3]);
            if all(size(patch) >= minPatchSize)
                patch_denoised = imnlmfilt(patch, 'DegreeOfSmoothing', filterStrength);
            else
                patch_denoised = mean(patch(:)) * ones(size(patch));  
            end    
            X(j, :) = patch_denoised(:);  
        end
     end
    %% Variance-Based Patch Selection
    v = var(X');  
    [ord, indx] = sort(v);  
    lex = round(length(indx) * 0.25);          
    train_ratio = 0.90;  
    num_train = round(train_ratio * (length(indx) - lex));  
    X_train = X(indx(lex:end), :);  
    Y_train = X(indx(lex:end), :);   
    X_val = X(indx(1:num_train), :);  
    Y_val = X(indx(1:num_train), :);
    % =========================================================================
% SCRIPT : NETWORK TRAINING, PREDICTION & QUALITY CONTROL (QC)
% -------------------------------------------------------------------------
% Methodology: Deep Learning Reconstruction & Residual Filtering
% Processing Stage: Iterative Denoising & Local Similarity Analysis
% Application: Distributed Acoustic Sensing (DAS) Data
% =========================================================================

%% --- 1. DEEP LEARNING TRAINING CONFIGURATION ---
% Optimization parameters for the Adam solver

 batchsize = 50000;  
    lam       = 1e-1;   
    options   = trainingOptions('adam', ...
        'MaxEpochs', 25, ...
        'Verbose', true, ...
        'MiniBatchSize', batchsize, ...
        'InitialLearnRate', 0.001, ...
        'L2Regularization', lam, ...
        'Plots', 'training-progress'); 

%% --- 2. NETWORK TRAINING & PREDICTION ---
% Ensures repeatability of the stochastic training process
rng(42, 'twister'); 

% Training the Multi-Head Attention Residual Network (MHA-RN)
net = trainNetwork(X_train, Y_train, lgraph, options);   

% Inference/Prediction on the full dataset
fprintf('Predicting denoised wavefield at iteration %d...\n', i);
outDN = haddl_DL_Predict(net, X, length(X), batchsize, 1);   

%% --- 3. WAVEFIELD RECONSTRUCTION ---
% Inverse patching to reconstruct the full 3D/2D seismic volume
% Variables n1, n2, n3 and w, s represent data and patch dimensions
d1_denoised = haddl_patch3d_inv(outDN', 1, n1, n2, n3, w1, w2, w3, s1z, s2z, s3z); 

%% --- 4. CONDITIONAL RESIDUAL FILTERING (FK-DIP) ---
% Applying FK-Dip filtering only during the initial iteration to 
% stabilize the signal-to-noise ratio (SNR).
if i == 1
    % Adaptive weight calculation
    w_fk = max(0.0001, 0.02 - 0.01 * i);
    
    % Residual noise suppression via FK-Dip
    d1_denoised = d1_denoised - haddl_fk_dip(d1_denoised, w_fk);
    fprintf('Initial FK-Dip filter applied (w = %.4f)\n', w_fk);
end

% Update the input for the next iteration if applicable
d_prev = d1_denoised;

d_haddl_f=d1_denoised;
%% --- 5. VISUALIZATION OF RESULTS ---
% Denoised Wavefield Display
figure('Name', sprintf('Denoised Data Iteration %d', i));
haddl_imagesc(d_haddl_f);
title(sprintf('Denoised Data: Iteration %d', i));
ylabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Panel', 'FontSize', 12, 'FontWeight', 'bold');
colormap(seis);
caxis([-100 100]);
set(gca, 'LineWidth', 2, 'FontSize', 12, 'FontWeight', 'bold');

% Residual Noise Display (Signal Leakage Check)
figure('Name', sprintf('Removed Noise Iteration %d', i));
haddl_imagesc(dn - d_haddl_f);
title(sprintf('Removed Noise: Iteration %d', i));
ylabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Panel', 'FontSize', 12, 'FontWeight', 'bold');
colormap(seis);
caxis([-100 100]);
set(gca, 'LineWidth', 2, 'FontSize', 12, 'FontWeight', 'bold');   

fprintf('Completed processing for iteration %d\n', i);

%% --- 6. DATA EXPORT ---
% Saving the high-fidelity results for subsequent research steps
filename = fullfile(resultsFolder, sprintf('d_haddl_f%d.mat', i));
save(filename, 'd_haddl_f');

%% --- 7. QUANTITATIVE QC: LOCAL SIMILARITY ANALYSIS ---
% Evaluating the similarity between the denoised signal and the removed noise
% to ensure no primary seismic events are lost.
rect = [20, 20, 1]; 
niter_sim = 20; 
eps_sim = 0; 
verb_sim = 0;

[simi1] = haddl_localsimi(dn - d_haddl_f, d_haddl_f, rect, niter_sim, eps_sim, verb_sim);

% Plotting Local Similarity Map
figure('Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8], 'Color', 'w');
imagesc(x, t, simi1);
colormap(jet);
c = colorbar;
c.Label.String = 'Local Similarity Index';
c.Label.FontSize = 20;
caxis([0, 1]);
ylabel('Time (s)', 'FontSize', 20, 'FontWeight', 'bold');
xlabel('Trace', 'FontSize', 20, 'FontWeight', 'bold');
title('Local Similarity Analysis (QC)', 'FontSize', 22);
set(gca, 'LineWidth', 2, 'FontSize', 20, 'FontWeight', 'bold');

% Optional: Export figure for the recruitment dossier
% print(gcf, '-depsc', '-r300', sprintf('local_similarity_iter_%d.eps', i));
end


