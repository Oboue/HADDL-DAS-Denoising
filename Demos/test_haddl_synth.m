% =========================================================================
% SCRIPT: test_haddl_synth.m
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
%   This script is designed to process synthetic seismic data and reproduce 
%   Figures 5 through 8 of the manuscript using the proposed HADDL method.
%
%   The analysis demonstrates the iterative refinement capabilities and 
%   signal reconstruction performance under controlled noise conditions:
%% Primary References:   
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

%% SECTION 1: PATH MANAGEMENT & REPRODUCIBILITY
% -------------------------------------------------------------------------
fprintf('--- Initializing Synthetic Environment ---\n');

% Manage Paths (Cross-platform compatible)
addpath(genpath(fullfile('..', 'Synth_data')));
addpath(genpath(fullfile('..', 'Subroutines')));

% Standard seed for Deep Learning weight initialization consistency
rng(42, 'twister'); 

%% SECTION 2: DATA LOADING AND DYNAMIC SCALING
% -------------------------------------------------------------------------
fprintf('--- Loading Synthetic Seismic Datasets ---\n');

% 1. Load Ground Truth (Clean Data)
if exist('micro_sf_3001_334_3.mat', 'file')
    load('micro_sf_3001_334_3.mat'); 
    % Scaling to range [-1, 1] or [0, 1] for network convergence
    d = haddl_scale(data(:,:,1), 2); 
    [n1, n2, n3] = size(d); % n1: Time, n2: Channels
else
    error('Ground Truth file not found.');
end

% 2. Load Noisy Benchmark Input
if exist('dnoiseSynthDAS.mat', 'file')
    load('dnoiseSynthDAS.mat'); 
    dn = dnoiseSynthDAS; 
    
    % Dimensions consistency check
    if any(size(dn) ~= size(d))
        fprintf('Warning: Dimensions mismatch. Resizing noisy data...\n');
        dn = dn(1:n1, 1:n2);
    end
else
    error('Noisy Synthetic file not found.');
end
%%
% 5. Metrics & Grid
psnr_Noisy = haddl_snr(d, dn, 2) 
dt = 0.004; 
t  = (0:n1-1) * dt; 
x  = 1:n2;
%% SECTION 3: QUANTITATIVE QUALITY CONTROL (QC)
% -------------------------------------------------------------------------
% SNR / PSNR estimation before processing
snr_initial = haddl_snr(d, dn, 2); 
fprintf('-> Dataset Dimensions: [%d x %d]\n', n1, n2);
fprintf('-> Baseline SNR (Input): %.2f dB\n', snr_initial);

% Physical Grid Definition
dt = 0.004;             % Sampling interval (s)
time_axis = (0:n1-1)*dt;
chan_axis = 1:n2;

fprintf('-> Success: Synthetic environment ready for HADDL processing.\n');

%% SECTION 4: PATCH AND HYPERPARAMETERS
% -------------------------------------------------------------------------
% Extraction parameters
w1 = 5; 
w2 = 5; 
w3 = 1; 
s1z = 1;
s2z = 1;
s3z = 1;
inpsize = w1 * w2 * w3; 

% Layer Dimensions (Encoder/Decoder)
D1 = 40;
D2 = ceil(D1 / 2);
D3 = ceil(D2 / 2);
D4 = ceil(D3 / 2);
D5 = ceil(D4 / 2);
D6 = ceil(D5 / 2);
D7 = ceil(D6 / 2);
D8 = ceil(D7 / 2);
D9 = ceil(D8 / 2);
D10= ceil(D8 / 2);

% Attention Mechanism dimensions
D_query  = 512;
D_key    = 512;
D_value  = 512;
numHeads = 8; 
D_head   = D_query / numHeads; 
l1Factor = 0.00001; % Sparsity regularization

%% SECTION 3: NETWORK ARCHITECTURE (LAYER GRAPH)
% -------------------------------------------------------------------------
layers = [featureInputLayer(inpsize, 'Name', 'input')];
lgraph = layerGraph(layers);

%% ENCODER STAGE
% -------------------------------------------------------------------------
% Encoder Block 0
r0         = fullyConnectedLayer(D1, 'WeightL2Factor', l1Factor, 'Name', 'r0');
r0_relu    = reluLayer('Name', 'r0_relu');
r0_dropout = dropoutLayer(0.00001, 'Name', 'r0_dropout'); 

lgraph = addLayers(lgraph, r0);
lgraph = addLayers(lgraph, r0_relu);
lgraph = addLayers(lgraph, r0_dropout);

lgraph = connectLayers(lgraph, 'input', 'r0');
lgraph = connectLayers(lgraph, 'r0', 'r0_relu');
lgraph = connectLayers(lgraph, 'r0_relu', 'r0_dropout');

% Encoder Block 1
r1         = fullyConnectedLayer(D2, 'WeightL2Factor', l1Factor, 'Name', 'r1');
r1_relu    = reluLayer('Name', 'r1_relu');
r1_dropout = dropoutLayer(0.00001, 'Name', 'r1_dropout');

lgraph = addLayers(lgraph, r1);
lgraph = addLayers(lgraph, r1_relu);
lgraph = addLayers(lgraph, r1_dropout);

lgraph = connectLayers(lgraph, 'r0_dropout', 'r1');
lgraph = connectLayers(lgraph, 'r1', 'r1_relu');
lgraph = connectLayers(lgraph, 'r1_relu', 'r1_dropout');

%% MULTI-HEAD ATTENTION STAGE
% -------------------------------------------------------------------------
% Projection Layers
queryLayer_multi = fullyConnectedLayer(D_query, 'WeightL2Factor', l1Factor, 'Name', 'query_fc_multi');
keyLayer_multi   = fullyConnectedLayer(D_key,   'WeightL2Factor', l1Factor, 'Name', 'key_fc_multi');
valueLayer_multi = fullyConnectedLayer(D_value, 'WeightL2Factor', l1Factor, 'Name', 'value_fc_multi');

lgraph = addLayers(lgraph, queryLayer_multi);
lgraph = addLayers(lgraph, keyLayer_multi);
lgraph = addLayers(lgraph, valueLayer_multi);

lgraph = connectLayers(lgraph, 'r1_dropout', 'query_fc_multi');
lgraph = connectLayers(lgraph, 'r1_dropout', 'key_fc_multi');
lgraph = connectLayers(lgraph, 'r1_dropout', 'value_fc_multi');

% Parallel Head Generation
for headIdx = 1:numHeads
    headName = num2str(headIdx);
    
    % Per-head projections
    query_fc_head = fullyConnectedLayer(D_head, 'WeightL2Factor', l1Factor, 'Name', ['query_fc_head_', headName]);
    key_fc_head   = fullyConnectedLayer(D_head, 'WeightL2Factor', l1Factor, 'Name', ['key_fc_head_', headName]);
    value_fc_head = fullyConnectedLayer(D_head, 'WeightL2Factor', l1Factor, 'Name', ['value_fc_head_', headName]);
    
    lgraph = addLayers(lgraph, query_fc_head);
    lgraph = addLayers(lgraph, key_fc_head);
    lgraph = addLayers(lgraph, value_fc_head);
    
    lgraph = connectLayers(lgraph, 'query_fc_multi', ['query_fc_head_', headName]);
    lgraph = connectLayers(lgraph, 'key_fc_multi',   ['key_fc_head_', headName]);
    lgraph = connectLayers(lgraph, 'value_fc_multi', ['value_fc_head_', headName]);
    
    % Core Attention Ops
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

% Aggregation and Softmax Sharpening
concatAttention = concatenationLayer(1, numHeads, 'Name', 'concat_attention');
lgraph = addLayers(lgraph, concatAttention);

for headIdx = 1:numHeads
    headName = num2str(headIdx);
    lgraph = connectLayers(lgraph, ['attention_output_head_', headName], ['concat_attention/in', headName]);
end

temperature    = 0.0001; 
tempScaleLayer = TemperatureScalingLayer(temperature, 'temp_scaling');
lgraph = addLayers(lgraph, tempScaleLayer);
lgraph = connectLayers(lgraph, 'concat_attention', 'temp_scaling');

%% DECODER STAGE (FC1 - FC7)
% -------------------------------------------------------------------------
% Sequential Block Generation
fc_names = {'fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7'};
fc_dims  = {D3, D4, D5, D6, D7, D8, D9};
prev_in  = 'temp_scaling';

for i = 1:length(fc_names)
    f_name = fc_names{i};
    f_dim  = fc_dims{i};
    
    fc_l      = fullyConnectedLayer(f_dim, 'WeightL2Factor', l1Factor, 'Name', f_name);
    fc_relu   = reluLayer('Name', [f_name, '_relu']);
    fc_drop   = dropoutLayer(0.00001, 'Name', [f_name, '_dropout']);
    
    lgraph = addLayers(lgraph, fc_l);
    lgraph = addLayers(lgraph, fc_relu);
    lgraph = addLayers(lgraph, fc_drop);
    
    lgraph = connectLayers(lgraph, prev_in, f_name);
    lgraph = connectLayers(lgraph, f_name, [f_name, '_relu']);
    lgraph = connectLayers(lgraph, [f_name, '_relu'], [f_name, '_dropout']);
    
    prev_in = [f_name, '_dropout'];
end

%% OUTPUT AND SKIP CONNECTION
% -------------------------------------------------------------------------
concatLayer = concatenationLayer(1, 2, 'Name', 'concat_output'); 
lgraph = addLayers(lgraph, concatLayer);

lgraph = connectLayers(lgraph, 'r0_dropout',  'concat_output/in1'); 
lgraph = connectLayers(lgraph, 'fc7_dropout', 'concat_output/in2'); 

output = fullyConnectedLayer(inpsize, 'WeightL2Factor', l1Factor, 'Name', 'output'); 
lgraph = addLayers(lgraph, output);
lgraph = connectLayers(lgraph, 'concat_output', 'output'); 

regressionLayer = regressionLayer('Name', 'regression_output');
lgraph = addLayers(lgraph, regressionLayer);
lgraph = connectLayers(lgraph, 'output', 'regression_output');

%% NETWORK VISUALIZATION
% -------------------------------------------------------------------------
figure; plot(lgraph);
analyzeNetwork(lgraph);
%%
%% SECTION 4: ITERATIVE REFINEMENT AND TRAINING LOOP
% -------------------------------------------------------------------------
% Setup output directory
outputFolder = 'Output_Synth_HADDL_Proposed';
if ~exist(outputFolder, 'dir'), mkdir(outputFolder); end

% Loop Parameters
niter = 2; 
SNR_values = zeros(1, niter); % Pre-allocate for automation

for i = 1:niter
    % Determine input data for the current iteration
    if i == 1
        d_prev = dn;           % Start with original noisy data
    else
        d_prev = d1_denoised;  % Use output from previous iteration
    end

    %% PATCH GENERATION
    % ---------------------------------------------------------------------
    X = haddl_patch3d(d_prev, 1, w1, w2, w3, s1z, s2z, s3z);
    X = double(X'); 

    %% CONDITIONAL DP-NLM FILTERING (Iteration 1 Only)
    % ---------------------------------------------------------------------
    if i == 1
        minPatchSize   = 21;  
        filterStrength = 0.001; 
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

    %% DATA PARTITIONING & VARIANCE-BASED SELECTION
    % ---------------------------------------------------------------------
    v = var(X');  
    [ord, indx] = sort(v);  
    lex         = round(length(indx) * 0.25);                     
    train_ratio = 0.9;  
    num_train   = round(train_ratio * (length(indx) - lex));  
    
    X_train = X(indx(lex:end), :);  
    Y_train = X(indx(lex:end), :);   
    X_val   = X(indx(1:num_train), :);  
    Y_val   = X(indx(1:num_train), :);

    %% TRAINING CONFIGURATION
    % ---------------------------------------------------------------------
    batchsize = 50000;  
    lam       = 1e-1;   
    options   = trainingOptions('adam', ...
        'MaxEpochs', 5, ...
        'Verbose', true, ...
        'MiniBatchSize', batchsize, ...
        'InitialLearnRate', 0.01, ...
        'L2Regularization', lam, ...
        'Plots', 'training-progress'); 

    % Train Network
    net = trainNetwork(X_train, Y_train, lgraph, options);   

    %% PREDICTION AND SIGNAL RECONSTRUCTION
    rng(42, 'twister'); % Ensures repeatability of random operations
    % ---------------------------------------------------------------------    
    outDN       = haddl_DL_Predict(net, X, length(X), batchsize, 1);   
    d1_denoised = haddl_patch3d_inv(outDN', 1, n1, n2, n3, w1, w2, w3, s1z, s2z, s3z); 

     %% CONDITIONAL FK-DIP FILTERING (Iteration 1 Only)
    % ---------------------------------------------------------------------
    if i == 1
        w_fk = 0.000001;  
        % Removal of dipping noise artifacts via FK-filter
        d1_denoised = d1_denoised - haddl_fk_dip(d1_denoised, w_fk);
    end
    
    % Update for next iteration
    d_prev = d1_denoised;
    
    %% PERFORMANCE EVALUATION & AUTOMATED DATA CAPTURE
    % ---------------------------------------------------------------------
    % Calculate SNR for current iteration (MANDATORY before assignment)
    rng(42, 'twister'); % Maintain repeatability
    psnr_Noisy    = haddl_snr(d, dn, 2);
    psnr_denoised = haddl_snr(d, d1_denoised, 2);
    
    SNR_values(i) = psnr_denoised; % Capture SNR for plotting
    
    % Store specific iterations for Figures 3 and 4
    % if i == 1, d1_iter1 = d1_denoised; end
    if i == 2, d1_iter2 = d1_denoised; end
    % if i == 5, d1_iter5 = d1_denoised; end
    
    % Assign result for final save (Works for any niter)
    d_synth_haddl = d1_denoised; 

    fprintf('Iteration %d Summary:\n', i);
    fprintf('  - Input SNR: %.2f dB\n', psnr_Noisy);
    fprintf('  - Denoised SNR: %.2f dB\n', psnr_denoised);
end

%% FINAL ARCHIVING (Outside the loop)
% -------------------------------------------------------------------------
save(fullfile(outputFolder, 'd_synth_haddl.mat'), 'd_synth_haddl', 'SNR_values');
fprintf('Success: Data saved to %s\n', outputFolder);
toc;

%% SECTION 6: FIGURE 4 - COMPREHENSIVE ANALYSIS (CLEAN, NOISY, HADDL, RESIDUAL, SIMI)
% -------------------------------------------------------------------------
% 1. Data Preparation
noise_haddl = dn - d_synth_haddl; 

% 2. Local Similarity Analysis
rect = [20 20 1]; nsim_iter = 20; eps_val = 0; verb = 0;
simi1 = haddl_localsimi(noise_haddl, d_synth_haddl, rect, nsim_iter, eps_val, verb); 

% 3. Figure Initialization
figure('Name', 'Figure 4: HADDL Performance Analysis', ...
       'units', 'normalized', 'Position', [0.1 0.1 0.8 0.8], 'color', 'w');
t_layout = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% --- ROW 1: WAVEFORM COMPARISON ---

% (a) Clean Data
nexttile;
    haddl_imagesc(d, 100, 2, x, t); title('Clean (d)');
    ylabel('Time (s)'); xlabel('Channel');
    text(-0.2, 1.1, '(a)', 'Units', 'normalized', 'FontSize', 22, 'FontWeight', 'bold');

% (b) Noisy Input
nexttile;
    haddl_imagesc(dn, 100, 2, x, t); title('Noisy (dn)');
    ylabel('Time (s)'); xlabel('Channel');
    text(-0.2, 1.1, '(b)', 'Units', 'normalized', 'FontSize', 22, 'FontWeight', 'bold');

% (c) HADDL Result
nexttile;
    haddl_imagesc(d_synth_haddl, 100, 2, x, t); title('HADDL Denoised');
    ylabel('Time (s)'); xlabel('Channel');
    text(-0.2, 1.1, '(c)', 'Units', 'normalized', 'FontSize', 22, 'FontWeight', 'bold');

% --- ROW 2: ERROR ANALYSIS ---

% (d) Difference Section (Residuals)
nexttile;
    haddl_imagesc(noise_haddl, 100, 2, x, t); title('Difference (dn - HADDL)');
    ylabel('Time (s)'); xlabel('Channel');
    text(-0.2, 1.1, '(d)', 'Units', 'normalized', 'FontSize', 22, 'FontWeight', 'bold');
    cb1 = colorbar; cb1.Label.String = 'Amplitude';

% (e) Local Similarity Map
nexttile;
    haddl_imagesc(simi1, 100, 2, x, t); title('Local Similarity');
    ylabel('Time (s)'); xlabel('Channel');
    text(-0.2, 1.1, '(e)', 'Units', 'normalized', 'FontSize', 22, 'FontWeight', 'bold');
    cb2 = colorbar; cb2.Label.String = 'Similarity Index';

% --- FORMATTING ---

% Apply 'seis' colormap and CLim to the first 4 tiles (Waveforms & Residual)
for k = 1:4
    ax = nexttile(k);
    colormap(ax, seis);
    set(ax, 'CLim', [-0.5 0.5], 'LineWidth', 1.5, 'FontSize', 14);
end

% Apply 'jet' colormap and CLim to the Similarity tile (Tile 5)
ax_sim = nexttile(5);
colormap(ax_sim, jet);
set(ax_sim, 'CLim', [0 1], 'LineWidth', 1.5, 'FontSize', 14);

% Final layout adjustment
title(t_layout, 'HADDL Framework Evaluation: Residual and Similarity Analysis', ...
      'FontSize', 24, 'FontWeight', 'bold');

%%