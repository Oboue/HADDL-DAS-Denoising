% =========================================================================
% SCRIPT: test_mharn_FORGE_DAS.m
% -------------------------------------------------------------------------
% Article Title: "A Hybrid Attention-Driven Deep Learning Framework for 
%                Denoising DAS Data"
% -------------------------------------------------------------------------
% Authors:      Oboué, Y. A. S. I., Chen, Y., & Chen, Y.
% Date:         May 2026
% Affiliations: Zhejiang University | University of Texas at Austin
% Journal:      Geophysics 
% -------------------------------------------------------------------------
% Description:
%   This script serves as a baseline benchmark using the MHA-RN (Multi-head 
%   Attention Residual Network) to process the FORGE DAS dataset.
%
%   The results from this script are used to evaluate and contrast the 
%   performance of the proposed HADDL framework against standard 
%   multi-head attention architectures in geothermal environments.
%
% Copyright (c) 2026. All rights reserved.
% =========================================================================

clc; clear; close all;

%% SECTION 1: PATH MANAGEMENT & ENVIRONMENT SETUP
% -------------------------------------------------------------------------
fprintf('--- Initializing FORGE Dataset: MHA-RN Benchmark ---\n');
addpath(genpath('../Field_data'));   
addpath(genpath('../subroutines')); % Necessary for MHA-RN operators

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

fprintf('-> Input dimensions: %d samples x %d channels\n', n1, n2);
fprintf('-> Ready for MHA-RN baseline execution.\n');
%% --- 2. HYPERPARAMETER DEFINITION ---
% Patch extraction parameters (3D Context)
w1 = 8; w2 = 8; w3 = 1;
s1z = 1; s2z = 1; s3z = 1;
inpsize = w1 * w2 * w3; % Input dimension for the neural network

% Hidden layer dimensions (Deep Bottleneck Architecture)
D1 = 40; D2 = ceil(D1 / 2); D3 = ceil(D2 / 2);
D4 = ceil(D3 / 2); D5 = ceil(D4 / 2); D6 = ceil(D5 / 2);
D7 = ceil(D6 / 2); D8 = ceil(D7 / 2); D9 = ceil(D8 / 2); D10 = ceil(D8 / 2);

% Multi-Head Attention (MHA) Parameters
D_query = 512; D_key = 512; D_value = 512;
numHeads = 8;
D_head = D_query / numHeads;
l1Factor = 0.01; % L1 Regularization for sparsity control

%% --- 3. NEURAL NETWORK ARCHITECTURE CONSTRUCTION ---
% Initial Input Layer
layers = [featureInputLayer(inpsize, 'Name', 'input')];
lgraph = layerGraph(layers);

% First Encoder Block (Fully Connected + ReLU + Dropout)
r0 = fullyConnectedLayer(D1, 'WeightL2Factor', l1Factor, 'Name', 'r0');
r0_relu = reluLayer('Name', 'r0_relu');
r0_dropout = dropoutLayer(0.35, 'Name', 'r0_dropout');
lgraph = addLayers(lgraph, r0); lgraph = addLayers(lgraph, r0_relu); lgraph = addLayers(lgraph, r0_dropout);
lgraph = connectLayers(lgraph, 'input', 'r0');
lgraph = connectLayers(lgraph, 'r0', 'r0_relu');
lgraph = connectLayers(lgraph, 'r0_relu', 'r0_dropout');

% Second Encoder Block
r1 = fullyConnectedLayer(D2, 'WeightL2Factor', l1Factor, 'Name', 'r1');
r1_relu = reluLayer('Name', 'r1_relu');
r1_dropout = dropoutLayer(0.35, 'Name', 'r1_dropout');
lgraph = addLayers(lgraph, r1); lgraph = addLayers(lgraph, r1_relu); lgraph = addLayers(lgraph, r1_dropout);
lgraph = connectLayers(lgraph, 'r0_dropout', 'r1');
lgraph = connectLayers(lgraph, 'r1', 'r1_relu');
lgraph = connectLayers(lgraph, 'r1_relu', 'r1_dropout');

% --- Multi-Head Attention Mechanism Integration ---
queryLayer_multi = fullyConnectedLayer(D_query, 'WeightL2Factor', l1Factor, 'Name', 'query_fc_multi');
keyLayer_multi = fullyConnectedLayer(D_key, 'WeightL2Factor', l1Factor, 'Name', 'key_fc_multi');
valueLayer_multi = fullyConnectedLayer(D_value, 'WeightL2Factor', l1Factor, 'Name', 'value_fc_multi');
lgraph = addLayers(lgraph, queryLayer_multi); lgraph = addLayers(lgraph, keyLayer_multi); lgraph = addLayers(lgraph, valueLayer_multi);
lgraph = connectLayers(lgraph, 'r1_dropout', 'query_fc_multi');
lgraph = connectLayers(lgraph, 'r1_dropout', 'key_fc_multi');
lgraph = connectLayers(lgraph, 'r1_dropout', 'value_fc_multi');

% Loop to create and connect individual Attention Heads
for headIdx = 1:numHeads
    headName = num2str(headIdx);
    query_fc_head = fullyConnectedLayer(D_head, 'WeightL2Factor', l1Factor, 'Name', ['query_fc_head_', headName]);
    key_fc_head = fullyConnectedLayer(D_head, 'WeightL2Factor', l1Factor, 'Name', ['key_fc_head_', headName]);
    value_fc_head = fullyConnectedLayer(D_head, 'WeightL2Factor', l1Factor, 'Name', ['value_fc_head_', headName]);
    lgraph = addLayers(lgraph, query_fc_head); lgraph = addLayers(lgraph, key_fc_head); lgraph = addLayers(lgraph, value_fc_head);
    lgraph = connectLayers(lgraph, 'query_fc_multi', ['query_fc_head_', headName]);
    lgraph = connectLayers(lgraph, 'key_fc_multi', ['key_fc_head_', headName]);
    lgraph = connectLayers(lgraph, 'value_fc_multi', ['value_fc_head_', headName]);
    
    % Per-head Attention logic (Dot-product -> Softmax -> Weighting)
    dotProductLayerHead = multiplicationLayer(2, 'Name', ['attention_dot_product_head_', headName]);
    softmaxLayerHead = softmaxLayer('Name', ['attention_softmax_head_', headName]);
    attentionOutputLayerHead = multiplicationLayer(2, 'Name', ['attention_output_head_', headName]);
    lgraph = addLayers(lgraph, dotProductLayerHead); lgraph = addLayers(lgraph, softmaxLayerHead); lgraph = addLayers(lgraph, attentionOutputLayerHead);
    lgraph = connectLayers(lgraph, ['query_fc_head_', headName], ['attention_dot_product_head_', headName, '/in1']);
    lgraph = connectLayers(lgraph, ['key_fc_head_', headName], ['attention_dot_product_head_', headName, '/in2']);
    lgraph = connectLayers(lgraph, ['attention_dot_product_head_', headName], ['attention_softmax_head_', headName]);
    lgraph = connectLayers(lgraph, ['attention_softmax_head_', headName], ['attention_output_head_', headName, '/in1']);
    lgraph = connectLayers(lgraph, ['value_fc_head_', headName], ['attention_output_head_', headName, '/in2']);
end

% Concatenate heads and apply Temperature Scaling
concatAttention = concatenationLayer(1, numHeads, 'Name', 'concat_attention');
lgraph = addLayers(lgraph, concatAttention);
for headIdx = 1:numHeads
    headName = num2str(headIdx);
    lgraph = connectLayers(lgraph, ['attention_output_head_', headName], ['concat_attention/in', headName]);
end
temperature = 0.001; 
tempScaleLayer = TemperatureScalingLayer(temperature, 'temp_scaling');
lgraph = addLayers(lgraph, tempScaleLayer);
lgraph = connectLayers(lgraph, 'concat_attention', 'temp_scaling');

% --- Decoder Block (Post-Attention Layers) ---
fc1 = fullyConnectedLayer(D3, 'WeightL2Factor', l1Factor, 'Name', 'fc1');
fc1_relu = reluLayer('Name', 'fc1_relu');
fc1_dropout = dropoutLayer(0.35, 'Name', 'fc1_dropout');
lgraph = addLayers(lgraph, fc1); lgraph = addLayers(lgraph, fc1_relu); lgraph = addLayers(lgraph, fc1_dropout);
lgraph = connectLayers(lgraph, 'temp_scaling', 'fc1');
lgraph = connectLayers(lgraph, 'fc1', 'fc1_relu');
lgraph = connectLayers(lgraph, 'fc1_relu', 'fc1_dropout');

% Sequential layers fc2 to fc6
fc2 = fullyConnectedLayer(D4, 'WeightL2Factor', l1Factor, 'Name', 'fc2'); fc2_relu = reluLayer('Name', 'fc2_relu'); fc2_dropout = dropoutLayer(0.35, 'Name', 'fc2_dropout');
lgraph = addLayers(lgraph, fc2); lgraph = addLayers(lgraph, fc2_relu); lgraph = addLayers(lgraph, fc2_dropout);
lgraph = connectLayers(lgraph, 'fc1_dropout', 'fc2'); lgraph = connectLayers(lgraph, 'fc2', 'fc2_relu'); lgraph = connectLayers(lgraph, 'fc2_relu', 'fc2_dropout');

fc3 = fullyConnectedLayer(D5, 'WeightL2Factor', l1Factor, 'Name', 'fc3'); fc3_relu = reluLayer('Name', 'fc3_relu'); fc3_dropout = dropoutLayer(0.35, 'Name', 'fc3_dropout');
lgraph = addLayers(lgraph, fc3); lgraph = addLayers(lgraph, fc3_relu); lgraph = addLayers(lgraph, fc3_dropout);
lgraph = connectLayers(lgraph, 'fc2_dropout', 'fc3'); lgraph = connectLayers(lgraph, 'fc3', 'fc3_relu'); lgraph = connectLayers(lgraph, 'fc3_relu', 'fc3_dropout');

fc4 = fullyConnectedLayer(D6, 'WeightL2Factor', l1Factor, 'Name', 'fc4'); fc4_relu = reluLayer('Name', 'fc4_relu'); fc4_dropout = dropoutLayer(0.35, 'Name', 'fc4_dropout');
lgraph = addLayers(lgraph, fc4); lgraph = addLayers(lgraph, fc4_relu); lgraph = addLayers(lgraph, fc4_dropout);
lgraph = connectLayers(lgraph, 'fc3_dropout', 'fc4'); lgraph = connectLayers(lgraph, 'fc4', 'fc4_relu'); lgraph = connectLayers(lgraph, 'fc4_relu', 'fc4_dropout');

fc5 = fullyConnectedLayer(D7, 'WeightL2Factor', l1Factor, 'Name', 'fc5'); fc5_relu = reluLayer('Name', 'fc5_relu'); fc5_dropout = dropoutLayer(0.35, 'Name', 'fc5_dropout');
lgraph = addLayers(lgraph, fc5); lgraph = addLayers(lgraph, fc5_relu); lgraph = addLayers(lgraph, fc5_dropout);
lgraph = connectLayers(lgraph, 'fc4_dropout', 'fc5'); lgraph = connectLayers(lgraph, 'fc5', 'fc5_relu'); lgraph = connectLayers(lgraph, 'fc5_relu', 'fc5_dropout');

fc6 = fullyConnectedLayer(D8, 'WeightL2Factor', l1Factor, 'Name', 'fc6'); fc6_relu = reluLayer('Name', 'fc6_relu'); fc6_dropout = dropoutLayer(0.35, 'Name', 'fc6_dropout');
lgraph = addLayers(lgraph, fc6); lgraph = addLayers(lgraph, fc6_relu); lgraph = addLayers(lgraph, fc6_dropout);
lgraph = connectLayers(lgraph, 'fc5_dropout', 'fc6'); lgraph = connectLayers(lgraph, 'fc6', 'fc6_relu'); lgraph = connectLayers(lgraph, 'fc6_relu', 'fc6_dropout');

% fc7 block and Output Concatenation (Residual connection from r0_dropout)
fc7 = fullyConnectedLayer(D9, 'WeightL2Factor', l1Factor, 'Name', 'fc7'); fc7_relu = reluLayer('Name', 'fc7_relu'); fc7_dropout = dropoutLayer(0.35, 'Name', 'fc7_dropout');
lgraph = addLayers(lgraph, fc7); lgraph = addLayers(lgraph, fc7_relu); lgraph = addLayers(lgraph, fc7_dropout);
lgraph = connectLayers(lgraph, 'fc6_dropout', 'fc7'); lgraph = connectLayers(lgraph, 'fc7', 'fc7_relu'); lgraph = connectLayers(lgraph, 'fc7_relu', 'fc7_dropout');

concatLayer = concatenationLayer(1, 2, 'Name', 'concat_output');
lgraph = addLayers(lgraph, concatLayer);
lgraph = connectLayers(lgraph, 'r0_dropout', 'concat_output/in1');
lgraph = connectLayers(lgraph, 'fc7_dropout', 'concat_output/in2');

output = fullyConnectedLayer(inpsize, 'WeightL2Factor', l1Factor, 'Name', 'output');
regressionLayer = regressionLayer('Name', 'regression_output');
lgraph = addLayers(lgraph, output); lgraph = addLayers(lgraph, regressionLayer);
lgraph = connectLayers(lgraph, 'concat_output', 'output');
lgraph = connectLayers(lgraph, 'output', 'regression_output');

% Visualize Network Structure
figure; plot(lgraph); analyzeNetwork(lgraph);

%% --- 4. OUTPUT DIRECTORY SETUP ---
outputFolder = 'Output_Field_FORGE_DAS_MHARN';
if ~exist(outputFolder, 'dir'), mkdir(outputFolder); end

%% --- 5. ITERATIVE DENOISING PROCESS ---
niter = 1; 
for i = 1:niter
    % Set data for the current iteration
    if i == 1
        d_prev = dn; 
    else
        d_prev = d1_denoised; 
    end
    
    % Patch Generation (Vectorization of seismic data)
    X = haddl_patch3d(d_prev, 1, w1, w2, w3, s1z, s2z, s3z);
    X = double(X');  

    % Variance-Based Selection for Training/Validation Sets
    v = var(X');  
    [ord, indx] = sort(v);  
    lex = round(length(indx) * 0.25);          
    train_ratio = 0.90;  
    num_train = round(train_ratio * (length(indx) - lex));  
    X_train = X(indx(lex:end), :); Y_train = X(indx(lex:end), :);   
    % X_val = X(indx(1:num_train), :); Y_val = X(indx(1:num_train), :);

    % DL Training Options

     batchsize = 50000;  
    lam       = 1e-1;   
    options   = trainingOptions('adam', ...
        'MaxEpochs', 25, ...
        'Verbose', true, ...
        'MiniBatchSize', batchsize, ...
        'InitialLearnRate', 0.001, ...
        'L2Regularization', lam, ...
        'Plots', 'training-progress'); 
    
    % Train Network and Predict Denoised Patches
    net = trainNetwork(X_train, Y_train, lgraph, options);   
    outDN = haddl_DL_Predict(net, X, length(X), batchsize, 1);   

    % Reconstruct Full 2D/3D Seismic Data from Patches (Inverse Mapping)
    d1_denoised = haddl_patch3d_inv(outDN', 1, n1, n2, n3, w1, w2, w3, s1z, s2z, s3z); 
    d_prev = d1_denoised;
d_mharn_f=d1_denoised;
    %% --- 6. VISUALIZATION OF RESULTS ---
    % Denoised Wavefield
    figure; 
    haddl_imagesc(d_mharn_f);
    title(sprintf('Denoised Data at Iteration %d', i));
    ylabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold'); xlabel('Panel', 'FontSize', 12, 'FontWeight', 'bold');
    colormap(seis); caxis([-100 100]); set(gca, 'LineWidth', 2, 'FontSize', 12, 'FontWeight', 'bold');

    % Residual Noise (Noisy - Denoised)
    figure; 
    haddl_imagesc(dn - d_mharn_f);
    title(sprintf('Removed Noise at Iteration %d', i));
    ylabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold'); xlabel('Panel', 'FontSize', 12, 'FontWeight', 'bold');
    colormap(seis); caxis([-100 100]); set(gca, 'LineWidth', 2, 'FontSize', 12, 'FontWeight', 'bold');   
    
    fprintf('Completed iteration %d\n', i);

    %% --- 7. DATA EXPORT ---
    filename = fullfile(outputFolder, sprintf('d_mharn_f%d.mat', i));
    save(filename, 'd_mharn_f');

    %% --- 8. QUALITY CONTROL: LOCAL SIMILARITY ANALYSIS ---
    rect = [20, 20, 1]; niter_sim = 20; eps = 0; verb = 0;
    [simi1] = haddl_localsimi(dn-d_mharn_f, d_mharn_f, rect, niter_sim, eps, verb);
    
    figure('units', 'normalized', 'Position', [0.0 0.0 1, 1], 'color', 'w');
    imagesc(x, t, simi1); colormap(jet);
    c = colorbar; c.Label.String = 'Local similarity'; c.Label.FontSize = 30;
    caxis([0, 1]); ylabel('Time (s)', 'Fontsize', 30); xlabel('Trace', 'Fontsize', 30);
    set(gca, 'Linewidth', 2, 'Fontsize', 30);
end