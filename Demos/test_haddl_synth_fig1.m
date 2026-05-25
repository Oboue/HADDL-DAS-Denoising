% =========================================================================
% SCRIPT: test_haddl_synth_fig1.m
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
%   This script reproduces Figure 1 of the manuscript using SYNTHETIC DATA.
%   It illustrates the comparative performance and structural limitations of 
%   MHA-RN-based DIP, DP-NLM, and FK-dip filtering across three highly complex 
%   noise environments:
%
%   - Clean reference data (Ground Truth)
%   - Case 1: Pure Random Noise (SNR = -4.59 dB)
%   - Case 2: Mixed Random + Erratic Noise (SNR = -11.75 dB)
%   - Case 3: Dominant Horizontal Structural Noise (SNR = -15.97 dB)
% -------------------------------------------------------------------------
% Primary References:   
%   [1] Oboué, Y. A. S. I., Chen, Y., & Chen, Y. (2026). A Hybrid 
%       Attention-Driven Deep Learning Framework for Denoising DAS Data. 
%       Geophysics.
%   [2] Oboué, Y. A. S. I., Ying, H., Ma, S., Zuo, P., & Chen, Y. (2026). 
%       Deep Learning-Based Recovery of Weak PcP Phases from Noisy Seismic 
%       Records. Big Data and Earth System, 100041.
%   [3] Oboué, Y. A. S. I., Chen, Y., Guo, Z., & Chen, Y. (2025). Leveraging 
%       overfitting for good—A two-step deep image prior model for seismic 
%       denoising. Geophysics, 90(3), V205-V221.
%   [4] Saad, O. M., Oboue, Y. A. S. I., Bai, M., Samy, L., Yang, L., & 
%       Chen, Y. (2021). Self-attention deep image prior network for 
%       unsupervised 3-D seismic data enhancement. IEEE TGRS, 60, 1-14.
% -------------------------------------------------------------------------
% GNU General Public License Notice:
%   This program is free software: you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.
% =========================================================================
% Copyright (c) 2026 Oboué, Y. A. S. I., Chen, Y., & Chen, Y.
% =========================================================================
clc; clear; close all;
% =========================================================================
%% SECTION 1: SYNTHETIC REFERENCE DATA LOADING AND GLOBAL PARAMETERS
% -------------------------------------------------------------------------
% This script is located in the /Demos/ folder. 
% We use '..' to move up one level to reach the /Synth_data/ directory.
% -------------------------------------------------------------------------

% 1. Manage Paths (Move up from Demos to access other folders)
addpath(genpath(fullfile('..', 'Synth_data')));
addpath(genpath(fullfile('..', 'Subroutines')));

% 2. Load synthetic reference data (Ground Truth)
% MATLAB can now "see" this file thanks to the relative path added above
if exist('micro_sf_3001_334_3.mat', 'file')
    load('micro_sf_3001_334_3.mat'); 
    fprintf('-> Success: Synthetic data loaded correctly.\n');
else
    error('Data file not found. Check if /Synth_data/ is in the parent directory.');
end

% 3. Scaling and Dimensions
% Note: Using data(:,:,1) to extract the first slice of the synthetic cube
d = haddl_scale(data(:,:,1), 2);
[n1, n2, n3] = size(d);

% 4. Time and Space Axes
dt = 0.004;         % 4ms sampling
t  = (0:n1-1) * dt; 
x  = 1:n2;

fprintf('-> Grid ready: [%d samples x %d channels]\n', n1, n2);
%% The MHA-RN-based DIP 
%% PATCH CONFIGURATION
% -------------------------------------------------------------------------
w1 = 20; 
w2 = 20;
w3 = 1;
s1z = 1;
s2z = 1;
s3z = 1;
inpsize = w1 * w2 * w3; 

%% NEURAL NETWORK ARCHITECTURE DIMENSIONS
% -------------------------------------------------------------------------
% Fully connected layer dimensions
D1  = 40;
D2  = ceil(D1 / 2);
D3  = ceil(D2 / 2);
D4  = ceil(D3 / 2);
D5  = ceil(D4 / 2);
D6  = ceil(D5 / 2);
D7  = ceil(D6 / 2);
D8  = ceil(D7 / 2);
D9  = ceil(D8 / 2);
D10 = ceil(D8 / 2);

% Attention mechanism parameters
D_query  = 512;
D_key    = 512;
D_value  = 512;
numHeads = 8;
D_head   = D_query / numHeads;

% Regularization
l1Factor = 0.001;

%% SECTION 2: NETWORK DEFINITION AND ANALYSIS (LGRAPH)
% -------------------------------------------------------------------------
% Initialize Layer Graph
layers = [featureInputLayer(inpsize, 'Name', 'input')];
lgraph = layerGraph(layers);

%% ENCODER STAGE: INITIAL FULLY CONNECTED LAYERS
% -------------------------------------------------------------------------
% Layer r0
r0         = fullyConnectedLayer(D1, 'WeightL2Factor', l1Factor, 'Name', 'r0');
r0_relu    = reluLayer('Name', 'r0_relu');
r0_dropout = dropoutLayer(0.001, 'Name', 'r0_dropout');

lgraph = addLayers(lgraph, r0);
lgraph = addLayers(lgraph, r0_relu);
lgraph = addLayers(lgraph, r0_dropout);

lgraph = connectLayers(lgraph, 'input', 'r0');
lgraph = connectLayers(lgraph, 'r0', 'r0_relu');
lgraph = connectLayers(lgraph, 'r0_relu', 'r0_dropout');

% Layer r1
r1         = fullyConnectedLayer(D2, 'WeightL2Factor', l1Factor, 'Name', 'r1');
r1_relu    = reluLayer('Name', 'r1_relu');
r1_dropout = dropoutLayer(0.001, 'Name', 'r1_dropout');

lgraph = addLayers(lgraph, r1);
lgraph = addLayers(lgraph, r1_relu);
lgraph = addLayers(lgraph, r1_dropout);

lgraph = connectLayers(lgraph, 'r0_dropout', 'r1');
lgraph = connectLayers(lgraph, 'r1', 'r1_relu');
lgraph = connectLayers(lgraph, 'r1_relu', 'r1_dropout');

%% HYBRID ATTENTION MECHANISM (MULTI-HEAD)
% -------------------------------------------------------------------------
% Projection layers for Query, Key, and Value
queryLayer_multi = fullyConnectedLayer(D_query, 'WeightL2Factor', l1Factor, 'Name', 'query_fc_multi');
keyLayer_multi   = fullyConnectedLayer(D_key,   'WeightL2Factor', l1Factor, 'Name', 'key_fc_multi');
valueLayer_multi = fullyConnectedLayer(D_value, 'WeightL2Factor', l1Factor, 'Name', 'value_fc_multi');

lgraph = addLayers(lgraph, queryLayer_multi);
lgraph = addLayers(lgraph, keyLayer_multi);
lgraph = addLayers(lgraph, valueLayer_multi);

lgraph = connectLayers(lgraph, 'r1_dropout', 'query_fc_multi');
lgraph = connectLayers(lgraph, 'r1_dropout', 'key_fc_multi');
lgraph = connectLayers(lgraph, 'r1_dropout', 'value_fc_multi');

% Multi-Head Attention Loop
for headIdx = 1:numHeads
    headName = num2str(headIdx);
    
    % Head-specific projections
    query_fc_head = fullyConnectedLayer(D_head, 'WeightL2Factor', l1Factor, 'Name', ['query_fc_head_', headName]);
    key_fc_head   = fullyConnectedLayer(D_head, 'WeightL2Factor', l1Factor, 'Name', ['key_fc_head_', headName]);
    value_fc_head = fullyConnectedLayer(D_head, 'WeightL2Factor', l1Factor, 'Name', ['value_fc_head_', headName]);
    
    lgraph = addLayers(lgraph, query_fc_head);
    lgraph = addLayers(lgraph, key_fc_head);
    lgraph = addLayers(lgraph, value_fc_head);
    
    lgraph = connectLayers(lgraph, 'query_fc_multi', ['query_fc_head_', headName]);
    lgraph = connectLayers(lgraph, 'key_fc_multi',   ['key_fc_head_', headName]);
    lgraph = connectLayers(lgraph, 'value_fc_multi', ['value_fc_head_', headName]);
    
    % Attention operations (Dot product, Softmax, Weighting)
    dotProductLayerHead      = multiplicationLayer(2, 'Name', ['attention_dot_product_head_', headName]);
    softmaxLayerHead         = softmaxLayer('Name', ['attention_softmax_head_', headName]);
    attentionOutputLayerHead = multiplicationLayer(2, 'Name', ['attention_output_head_', headName]);
    
    lgraph = addLayers(lgraph, dotProductLayerHead);
    lgraph = addLayers(lgraph, softmaxLayerHead);
    lgraph = addLayers(lgraph, attentionOutputLayerHead);
    
    lgraph = connectLayers(lgraph, ['query_fc_head_', headName], ['attention_dot_product_head_', headName, '/in1']);
    lgraph = connectLayers(lgraph, ['key_fc_head_', headName],   ['attention_dot_product_head_', headName, '/in2']);
    lgraph = connectLayers(lgraph, ['attention_dot_product_head_', headName], ['attention_softmax_head_', headName]);
    lgraph = connectLayers(lgraph, ['attention_softmax_head_', headName],     ['attention_output_head_', headName, '/in1']);
    lgraph = connectLayers(lgraph, ['value_fc_head_', headName],              ['attention_output_head_', headName, '/in2']);
end

% Concatenation and Temperature Scaling
concatAttention = concatenationLayer(1, numHeads, 'Name', 'concat_attention');
lgraph = addLayers(lgraph, concatAttention);

for headIdx = 1:numHeads
    headName = num2str(headIdx);
    lgraph = connectLayers(lgraph, ['attention_output_head_', headName], ['concat_attention/in', headName]);
end

temperature    = 0.001; 
tempScaleLayer = TemperatureScalingLayer(temperature, 'temp_scaling');
lgraph = addLayers(lgraph, tempScaleLayer);
lgraph = connectLayers(lgraph, 'concat_attention', 'temp_scaling');

%% DECODER STAGE: DEEP REFINEMENT LAYERS (fc1 - fc7)
% -------------------------------------------------------------------------
% Constructing Sequential FC Blocks
fc_layers = {'fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7'};
fc_dims   = {D3, D4, D5, D6, D7, D8, D9};
prev_layer = 'temp_scaling';

for i = 1:length(fc_layers)
    name = fc_layers{i};
    dim  = fc_dims{i};
    
    layer_fc      = fullyConnectedLayer(dim, 'WeightL2Factor', l1Factor, 'Name', name);
    layer_relu    = reluLayer('Name', [name, '_relu']);
    layer_dropout = dropoutLayer(0.001, 'Name', [name, '_dropout']);
    
    lgraph = addLayers(lgraph, layer_fc);
    lgraph = addLayers(lgraph, layer_relu);
    lgraph = addLayers(lgraph, layer_dropout);
    
    lgraph = connectLayers(lgraph, prev_layer, name);
    lgraph = connectLayers(lgraph, name, [name, '_relu']);
    lgraph = connectLayers(lgraph, [name, '_relu'], [name, '_dropout']);
    
    prev_layer = [name, '_dropout'];
end

%% OUTPUT STAGE: SKIP CONNECTION AND REGRESSION
% -------------------------------------------------------------------------
% Concatenate initial encoder features with refined decoder features
concatLayer = concatenationLayer(1, 2, 'Name', 'concat_output'); 
lgraph = addLayers(lgraph, concatLayer);
lgraph = connectLayers(lgraph, 'r0_dropout', 'concat_output/in1'); 
lgraph = connectLayers(lgraph, 'fc7_dropout', 'concat_output/in2'); 

% Final output and regression
output = fullyConnectedLayer(inpsize, 'WeightL2Factor', l1Factor, 'Name', 'output'); 
lgraph = addLayers(lgraph, output);
lgraph = connectLayers(lgraph, 'concat_output', 'output'); 

regressionLayer = regressionLayer('Name', 'regression_output');
lgraph = addLayers(lgraph, regressionLayer);
lgraph = connectLayers(lgraph, 'output', 'regression_output');

% Visualize and Analyze
figure; plot(lgraph); analyzeNetwork(lgraph);

% Workspace setup
outputFolder = 'DenoisedSynthFORGEDASData_DIPMHAR_06032025';
if ~exist(outputFolder, 'dir'), mkdir(outputFolder); end

%% SECTION 3: COMPUTATION FOR DATASET 1 (dsynthDAS4.mat)
% -------------------------------------------------------------------------
disp('--- PROCESSING DATA 1: dsynthDAS4 ---');
load dsynthDAS4.mat
% Ensure dn contains the correct data
psnr_initial = haddl_snr(d, dn, 2);

%% ITERATIVE REFINEMENT STRATEGY
% -------------------------------------------------------------------------
niter = 1; 
for i = 1:niter
    % Set input for current iteration
    if i == 1
        d_prev = dn;
    else
        d_prev = d1_denoised;
    end
    
    % Patch extraction and variance-based sorting
    X = haddl_patch3d(d_prev, 1, w1, w2, w3, s1z, s2z, s3z);
    X = double(X');
    v = var(X');  
    [ord, indx] = sort(v);  
    
    % Data partitioning for training and validation
    lex         = round(length(indx) * 0.25);          
    train_ratio = 0.90;  
    num_train   = round(train_ratio * (length(indx) - lex));  
    
    X_train = X(indx(lex:end), :);  
    Y_train = X(indx(lex:end), :);   
    X_val   = X(indx(1:num_train), :);  
    Y_val   = X(indx(1:num_train), :);
    
    %% NETWORK TRAINING CONFIGURATION
    % ---------------------------------------------------------------------
    lam       = 0.1;   
    batchsize = 50000;  
    options   = trainingOptions('adam', ...
        'MaxEpochs', 5, ...
        'Verbose', true, ...
        'MiniBatchSize', batchsize, ...
        'InitialLearnRate', 0.01, ...
        'L2Regularization', lam, ...
        'Plots', 'training-progress'); 
    
    % Execute training
    net = trainNetwork(X_train, Y_train, lgraph, options);   
   
    %% PREDICTION AND RECONSTRUCTION
    % ---------------------------------------------------------------------
    outDN       = haddl_DL_Predict(net, X, length(X), batchsize, 1);   
    d1_denoised = haddl_patch3d_inv(outDN', 1, n1, n2, n3, w1, w2, w3, s1z, s2z, s3z); 
    
    % Store and evaluate results
    d1_mharn      = d1_denoised;
    psnr_d1_mharn = haddl_snr(d, d1_mharn, 2);
    
    fprintf('DATA 1 - SNR after iteration %d: %.2f dB\n', i, psnr_d1_mharn);
end

%% SECTION 4: COMPUTATION FOR DATASET 2 (dsynthDAS3.mat)
% -------------------------------------------------------------------------
disp('--- PROCESSING DATA 2: dsynthDAS3 ---');
load dsynthDAS3.mat 
dn = dsynthDAS3; % Update dn for the second dataset
psnr_initial2 = haddl_snr(d, dn, 2);

%% ITERATIVE REFINEMENT STRATEGY (DATASET 2)
% -------------------------------------------------------------------------
for i = 1:niter
    % Set input for current iteration
    if i == 1
        d_prev = dn;
    else
        d_prev = d1_denoised;
    end
    
    % Patch extraction and variance-based sorting
    X = haddl_patch3d(d_prev, 1, w1, w2, w3, s1z, s2z, s3z);
    X = double(X');
    v = var(X');  
    [ord, indx] = sort(v);  
    
    % Data partitioning
    lex         = round(length(indx) * 0.25);          
    train_ratio = 0.90;  
    num_train   = round(train_ratio * (length(indx) - lex));  
    
    X_train = X(indx(lex:end), :);  
    Y_train = X(indx(lex:end), :);   
    X_val   = X(indx(1:num_train), :);  
    Y_val   = X(indx(1:num_train), :);
    
    %% NETWORK TRAINING CONFIGURATION (SPECIFIC LAMBDA)
    % ---------------------------------------------------------------------
    lam = 0.25; % Specific lambda parameter for Dataset 2
    options = trainingOptions('adam', ...
        'MaxEpochs', 5, ...
        'Verbose', true, ...
        'MiniBatchSize', batchsize, ...
        'InitialLearnRate', 0.01, ...
        'L2Regularization', lam, ...
        'Plots', 'training-progress'); 
    
    % Execute training
    net = trainNetwork(X_train, Y_train, lgraph, options);   
    
    %% PREDICTION AND RECONSTRUCTION
    % ---------------------------------------------------------------------
    outDN       = haddl_DL_Predict(net, X, length(X), batchsize, 1);   
    d1_denoised = haddl_patch3d_inv(outDN', 1, n1, n2, n3, w1, w2, w3, s1z, s2z, s3z); 
    
    % Store and evaluate results
    d2_mharn      = d1_denoised;
    psnr_d2_mharn = haddl_snr(d, d2_mharn, 2);
    
    fprintf('DATA 2 - SNR after iteration %d: %.2f dB\n', i, psnr_d2_mharn);
end

%% SECTION 5: DP-NLM METHOD - PARAMETERS AND PROCESSING
% -------------------------------------------------------------------------
% Define common parameters for the Non-Local Means (NLM) filter
minPatchSize   = 21;  
filterStrength = 0.001;

%% DP-NLM: PROCESSING SYNTHETIC DATASET 1 (dsynthDAS4)
% -------------------------------------------------------------------------
fprintf('--- DP-NLM: Processing dsynthDAS4 ---\n');
load dsynthDAS4.mat
[n1, n2, n3] = size(dn);

% Set time and spatial axes
t = [0:n1-1] * dt; 
x = [1:n2];

% Patch Generation Parameters for DP-NLM
w1 = 3; w2 = 3; w3 = 1;
s1z = 1; s2z = 1; s3z = 1;

% Extract patches from the noisy synthetic data
X1 = haddl_patch3d(dn, 1, w1, w2, w3, s1z, s2z, s3z);
X1 = double(X1');

%% DP-NLM FILTERING CORE
% -------------------------------------------------------------------------
% Iteratively apply the NLM filter to each extracted patch
for j = 1:size(X1, 1)
    patch = reshape(X1(j, :), [w1, w2, w3]);
    if all(size(patch) >= minPatchSize)
        patch_denoised = imnlmfilt(patch, 'DegreeOfSmoothing', filterStrength);
    else
        % Fallback to mean value if patch size is smaller than minPatchSize
        patch_denoised = mean(patch(:)) * ones(size(patch));  
    end    
    X1(j, :) = patch_denoised(:);  
end

% Reconstruct the 3D data from processed patches
d1_denoised_4 = haddl_patch3d_inv(X1', 1, n1, n2, n3, w1, w2, w3, s1z, s2z, s3z); 
dpnlm_1       = d1_denoised_4;

% Performance evaluation using Signal-to-Noise Ratio (SNR)
psnr_noisy4    = haddl_snr(d, dn, 2);
psnr_denoised4 = haddl_snr(d, dpnlm_1, 2);

%% SECTION 6: DP-NLM METHOD - PROCESSING SYNTHETIC DATASET 2 (dsynthDAS3)
% -------------------------------------------------------------------------
fprintf('--- DP-NLM: Processing dsynthDAS3 ---\n');
load dsynthDAS3.mat 
dn = dsynthDAS3; 
[n1, n2, n3] = size(dn);

% Adjust patch dimensions for the second dataset
w1 = 4; w2 = 4; w3 = 1; 

% Patch Generation for Dataset 2
X2 = haddl_patch3d(dn, 1, w1, w2, w3, s1z, s2z, s3z);
X2 = double(X2');

%% DP-NLM FILTERING CORE
% -------------------------------------------------------------------------
% Apply NLM filtering loop for the second synthetic dataset
for j = 1:size(X2, 1)
    patch = reshape(X2(j, :), [w1, w2, w3]);
    if all(size(patch) >= minPatchSize)
        patch_denoised = imnlmfilt(patch, 'DegreeOfSmoothing', filterStrength);
    else
        patch_denoised = mean(patch(:)) * ones(size(patch));  
    end    
    X2(j, :) = patch_denoised(:);  
end

% Reconstruction and SNR evaluation for Dataset 2
d1_denoised_3 = haddl_patch3d_inv(X2', 1, n1, n2, n3, w1, w2, w3, s1z, s2z, s3z); 
dpnlm_2       = d1_denoised_3;
psnr_noisy3    = haddl_snr(d, dn, 2);
psnr_denoised3 = haddl_snr(d, dpnlm_2, 2);

%% SECTION 7: FK-DIP FILTERING - SINGLE DATASET (dnhoriz)
% -------------------------------------------------------------------------
fprintf('--- FK Filtering: Processing dnhoriz ---\n');
load dnhoriz.mat 
w_fk = 0.153;

% Apply the FK-dip filter to mitigate horizontal noise
d1_denoised_fk = dnhoriz - haddl_fk_dip(dnhoriz, w_fk);

% Evaluate performance for the FK-dip case
psnr_noisy_fk    = haddl_snr(d, dnhoriz, 2);
psnr_denoised_fk = haddl_snr(d, d1_denoised_fk, 2);

%% SECTION 8: FIGURE 1 GENERATION - COMPARATIVE ANALYSIS
% -------------------------------------------------------------------------
% Create the figure window for comprehensive visualization
figure('Name', 'Figure 1: Method Limitations', 'units', 'normalized', ...
       'Position', [0.1 0.1 0.8 0.8], 'color', 'w');

% Define common color axis limits for visual consistency
c_lim = [-0.5 0.5];

%% ROW 1: CASE 1 - RANDOM NOISE (-4.59 dB)
% -------------------------------------------------------------------------
subplot(3,4,1);
    haddl_imagesc(d, 95, 2, x, t); title('Clean');
    ylabel('Time (s)'); xlabel('Channel');
    text(-0.3, 1.1, '(a)', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold');
subplot(3,4,2);
    haddl_imagesc(dn, 95, 2, x, t); title('Noisy; S/N = -4.59 dB');
    ylabel('Time (s)'); xlabel('Channel');
    text(-0.3, 1.1, '(b)', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold');
subplot(3,4,3);
    haddl_imagesc(d1_mharn, 95, 2, x, t); title('MHA-RN');
    ylabel('Time (s)'); xlabel('Channel');
    text(-0.3, 1.1, '(c)', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold');
subplot(3,4,4);
    haddl_imagesc(dpnlm_1, 95, 2, x, t); title('DP-NLM');
    ylabel('Time (s)'); xlabel('Channel');
    text(-0.3, 1.1, '(d)', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold');

%% ROW 2: CASE 2 - RANDOM + ERRATIC NOISE (-11.75 dB)
% -------------------------------------------------------------------------
subplot(3,4,6);
    haddl_imagesc(dsynthDAS3, 95, 2, x, t); title('Noisy; S/N = -11.75 dB');
    ylabel('Time (s)'); xlabel('Channel');
    text(-0.3, 1.1, '(e)', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold');
subplot(3,4,7);
    haddl_imagesc(d2_mharn, 95, 2, x, t); title('MHA-RN');
    ylabel('Time (s)'); xlabel('Channel');
    text(-0.3, 1.1, '(f)', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold');
subplot(3,4,8);
    haddl_imagesc(dpnlm_2, 95, 2, x, t); title('DP-NLM');
    ylabel('Time (s)'); xlabel('Channel');
    text(-0.3, 1.1, '(g)', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold');

%% ROW 3: CASE 3 - HORIZONTAL NOISE (-15.97 dB)
% -------------------------------------------------------------------------
subplot(3,4,11);
    haddl_imagesc(dnhoriz, 95, 2, x, t); title('Noisy; S/N = -15.97 dB');
    ylabel('Time (s)'); xlabel('Channel');
    text(-0.3, 1.1, '(h)', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold');
subplot(3,4,12);
    haddl_imagesc(d1_denoised_fk, 95, 2, x, t); title('FK-dip');
    ylabel('Time (s)'); xlabel('Channel');
    text(-0.3, 1.1, '(i)', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold');

% Apply final visual styling to all axes in the figure
colormap(seis);
set(findobj(gcf, 'type', 'axes'), 'LineWidth', 2, 'FontSize', 12, 'CLim', c_lim);