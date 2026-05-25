% =========================================================================
% SCRIPT: test_mharn_synth.m
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
%
%   Primary References:   
%   [1] Oboué, Y. A. S. I., Chen, Y., & Chen, Y. (2026). A Hybrid 
%       Attention-Driven Deep Learning Framework for Denoising DAS Data. 
%       Geophysics.
%   [2] Oboué, Y. A. S. I., Ying, H., Ma, S., Zuo, P., & Chen, Y. (2026). 
%       Deep Learning-Based Recovery of Weak PcP Phases from Noisy Seismic 
%       Records. Big Data and Earth System, 100041.
%   [3] Oboué, Y. A. S. I., Chen, Y., Guo, Z., & Chen, Y. (2025). Leveraging 
%       overfitting for good—A two-step deep image prior model for seismic 
%       denoising. Geophysics, 90(3), V205-V221.
%   [4] Saad, O. M., M. Ravasi, and T. Alkhalifah, 2024, Noise attenuation 
%       in distributed acoustic sensing data using a guided unsupervised deep 
%       learning network: Geophysics, 89, V573–V587.
%   [5] Saad, O. M., Oboue, Y. A. S. I., Bai, M., Samy, L., Yang, L., & 
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
addpath(genpath('./Synth_data'));   
addpath(genpath('./subroutines')); 

% Standard seed for Deep Learning weight initialization consistency
rng(42, 'twister'); 

%% SECTION 2: DATA LOADING AND DYNAMIC SCALING
% -------------------------------------------------------------------------
fprintf('--- Loading Synthetic Seismic Datasets ---\n');

% 1. Load Ground Truth (Clean Data)
if exist('micro_sf_3001_334_3.mat', 'file')
    load('micro_sf_3001_334_3.mat'); 
    % Scaling to range [-1, 1] or [0, 1] for network convergence
    d = LO_scale(data(:,:,1), 2); 
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

%% SECTION 2: PATCH AND HYPERPARAMETERS
% -------------------------------------------------------------------------
% Extraction parameters (Large scale baseline)
w1 = 40; w2 = 40; w3 = 1;
s1z = 1; s2z = 1; s3z = 1;
inpsize = w1 * w2 * w3; 

% Layer Dimensions (Encoder/Decoder)
D1 = 650;
D2 = ceil(D1 / 2);
D3 = ceil(D2 / 2); D4 = ceil(D3 / 2); D5 = ceil(D4 / 2);
D6 = ceil(D5 / 2); D7 = ceil(D6 / 2); D8 = ceil(D7 / 2); D9 = ceil(D8 / 2);

% Attention Mechanism dimensions
D_query  = 512; D_key = 512; D_value = 512;
numHeads = 8; 
D_head   = D_query / numHeads; 
l1Factor = 0.00001; % Sparsity regularization

%% SECTION 3: NETWORK ARCHITECTURE (LAYER GRAPH)
% -------------------------------------------------------------------------
layers = [featureInputLayer(inpsize, 'Name', 'input')];
lgraph = layerGraph(layers);

% --- ENCODER STAGE ---
% Block 0
lgraph = addLayers(lgraph, [ ...
    fullyConnectedLayer(D1, 'WeightL2Factor', l1Factor, 'Name', 'r0'), ...
    reluLayer('Name', 'r0_relu'), ...
    dropoutLayer(0.00001, 'Name', 'r0_dropout')]);
lgraph = connectLayers(lgraph, 'input', 'r0');

% Block 1
lgraph = addLayers(lgraph, [ ...
    fullyConnectedLayer(D2, 'WeightL2Factor', l1Factor, 'Name', 'r1'), ...
    reluLayer('Name', 'r1_relu'), ...
    dropoutLayer(0.00001, 'Name', 'r1_dropout')]);
lgraph = connectLayers(lgraph, 'r0_dropout', 'r1');

% --- MULTI-HEAD ATTENTION STAGE ---
lgraph = addLayers(lgraph, fullyConnectedLayer(D_query, 'WeightL2Factor', l1Factor, 'Name', 'query_fc_multi'));
lgraph = addLayers(lgraph, fullyConnectedLayer(D_query, 'WeightL2Factor', l1Factor, 'Name', 'key_fc_multi'));
lgraph = addLayers(lgraph, fullyConnectedLayer(D_query, 'WeightL2Factor', l1Factor, 'Name', 'value_fc_multi'));
lgraph = connectLayers(lgraph, 'r1_dropout', 'query_fc_multi');
lgraph = connectLayers(lgraph, 'r1_dropout', 'key_fc_multi');
lgraph = connectLayers(lgraph, 'r1_dropout', 'value_fc_multi');

for headIdx = 1:numHeads
    hName = num2str(headIdx);
    lgraph = addLayers(lgraph, fullyConnectedLayer(D_head, 'WeightL2Factor', l1Factor, 'Name', ['q_head_', hName]));
    lgraph = addLayers(lgraph, fullyConnectedLayer(D_head, 'WeightL2Factor', l1Factor, 'Name', ['k_head_', hName]));
    lgraph = addLayers(lgraph, fullyConnectedLayer(D_head, 'WeightL2Factor', l1Factor, 'Name', ['v_head_', hName]));
    
    lgraph = connectLayers(lgraph, 'query_fc_multi', ['q_head_', hName]);
    lgraph = connectLayers(lgraph, 'key_fc_multi',   ['k_head_', hName]);
    lgraph = connectLayers(lgraph, 'value_fc_multi', ['v_head_', hName]);
    
    lgraph = addLayers(lgraph, multiplicationLayer(2, 'Name', ['dot_', hName]));
    lgraph = addLayers(lgraph, softmaxLayer('Name', ['soft_', hName]));
    lgraph = addLayers(lgraph, multiplicationLayer(2, 'Name', ['att_out_', hName]));
    
    lgraph = connectLayers(lgraph, ['q_head_', hName], ['dot_', hName, '/in1']);
    lgraph = connectLayers(lgraph, ['k_head_', hName], ['dot_', hName, '/in2']);
    lgraph = connectLayers(lgraph, ['dot_', hName], ['soft_', hName]);
    lgraph = connectLayers(lgraph, ['soft_', hName], ['att_out_', hName, '/in1']);
    lgraph = connectLayers(lgraph, ['v_head_', hName], ['att_out_', hName, '/in2']);
end

lgraph = addLayers(lgraph, concatenationLayer(1, numHeads, 'Name', 'concat_attention'));
for headIdx = 1:numHeads
    lgraph = connectLayers(lgraph, ['att_out_', num2str(headIdx)], ['concat_attention/in', num2str(headIdx)]);
end
lgraph = addLayers(lgraph, TemperatureScalingLayer(0.0001, 'temp_scaling'));
lgraph = connectLayers(lgraph, 'concat_attention', 'temp_scaling');

% --- DECODER STAGE ---
fc_names = {'fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7'};
fc_dims  = {D3, D4, D5, D6, D7, D8, D9};
prev_in  = 'temp_scaling';
for i = 1:7
    lgraph = addLayers(lgraph, [ ...
        fullyConnectedLayer(fc_dims{i}, 'WeightL2Factor', l1Factor, 'Name', fc_names{i}), ...
        reluLayer('Name', [fc_names{i}, '_relu']), ...
        dropoutLayer(0.00001, 'Name', [fc_names{i}, '_dropout'])]);
    lgraph = connectLayers(lgraph, prev_in, fc_names{i});
    prev_in = [fc_names{i}, '_dropout'];
end

% --- OUTPUT AND SKIP CONNECTION ---
lgraph = addLayers(lgraph, concatenationLayer(1, 2, 'Name', 'concat_output'));
lgraph = connectLayers(lgraph, 'r0_dropout',  'concat_output/in1'); 
lgraph = connectLayers(lgraph, 'fc7_dropout', 'concat_output/in2'); 
lgraph = addLayers(lgraph, fullyConnectedLayer(inpsize, 'WeightL2Factor', l1Factor, 'Name', 'output')); 
lgraph = connectLayers(lgraph, 'concat_output', 'output'); 
lgraph = addLayers(lgraph, regressionLayer('Name', 'regression_output'));
lgraph = connectLayers(lgraph, 'output', 'regression_output');
%%
% Setup output directory
outputFolder = 'Output_Synth_MHA-RN';
if ~exist(outputFolder, 'dir'), mkdir(outputFolder); end

%% SECTION 4: ITERATIVE REFINEMENT AND TRAINING LOOP
% -------------------------------------------------------------------------
% Setup output directory
outputFolder = 'Output_Synth_MHA-RN';
if ~exist(outputFolder, 'dir'), mkdir(outputFolder); end

% Loop Parameters
niter = 1; 
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
    X = yc_patch3d(d_prev, 1, w1, w2, w3, s1z, s2z, s3z);
    X = double(X'); 

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
    outDN       = DL_Predict(net, X, length(X), batchsize, 1);   
    d1_denoised = yc_patch3d_inv(outDN', 1, n1, n2, n3, w1, w2, w3, s1z, s2z, s3z); 

    % Update for next iteration
    d_prev = d1_denoised;
    
    %% PERFORMANCE EVALUATION & AUTOMATED DATA CAPTURE
    % ---------------------------------------------------------------------
    % Calculate SNR for current iteration (MANDATORY before assignment)
    rng(42, 'twister'); % Maintain repeatability
    psnr_Noisy    = yc_snr(d, dn, 2);
    psnr_denoised = yc_snr(d, d1_denoised, 2);
    
    SNR_values(i) = psnr_denoised; % Capture SNR for plotting
   
    
    % Assign result for final save (Works for any niter)
    d_mharn = d1_denoised; 

    fprintf('Iteration %d Summary:\n', i);
    fprintf('  - Input SNR: %.2f dB\n', psnr_Noisy);
    fprintf('  - Denoised SNR: %.2f dB\n', psnr_denoised);
end
    d_mharn = d1_denoised;
%% FINAL ARCHIVING (Outside the loop)
% -------------------------------------------------------------------------
save(fullfile(outputFolder, 'd_mharn.mat'), 'd_mharn', 'SNR_values');
fprintf('Success: Data saved to %s\n', outputFolder);
toc;

%% SECTION 6: FIGURE 4 - COMPREHENSIVE ANALYSIS (CLEAN, NOISY, HADDL, RESIDUAL, SIMI)
% -------------------------------------------------------------------------
% 1. Data Preparation
noise_mharn = dn - d_mharn; 

% 2. Local Similarity Analysis
rect = [20 20 1]; nsim_iter = 20; eps_val = 0; verb = 0;
simi1 = localsimi(noise_mharn, d_mharn, rect, nsim_iter, eps_val, verb); 

% 3. Figure Initialization
figure('Name', 'Figure 4: MHA-RN Performance Analysis', ...
       'units', 'normalized', 'Position', [0.1 0.1 0.8 0.8], 'color', 'w');
t_layout = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% --- ROW 1: WAVEFORM COMPARISON ---

% (a) Clean Data
nexttile;
    LO_imagesc(d, 100, 2, x, t); title('Clean (d)');
    ylabel('Time (s)'); xlabel('Channel');
    text(-0.2, 1.1, '(a)', 'Units', 'normalized', 'FontSize', 22, 'FontWeight', 'bold');

% (b) Noisy Input
nexttile;
    LO_imagesc(dn, 100, 2, x, t); title('Noisy (dn)');
    ylabel('Time (s)'); xlabel('Channel');
    text(-0.2, 1.1, '(b)', 'Units', 'normalized', 'FontSize', 22, 'FontWeight', 'bold');

% (c) HADDL Result
nexttile;
    LO_imagesc(d_mharn, 100, 2, x, t); title('HADDL Denoised');
    ylabel('Time (s)'); xlabel('Channel');
    text(-0.2, 1.1, '(c)', 'Units', 'normalized', 'FontSize', 22, 'FontWeight', 'bold');

% --- ROW 2: ERROR ANALYSIS ---

% (d) Difference Section (Residuals)
nexttile;
    LO_imagesc(noise_mharn, 100, 2, x, t); title('Difference (dn - MHA-RN)');
    ylabel('Time (s)'); xlabel('Channel');
    text(-0.2, 1.1, '(d)', 'Units', 'normalized', 'FontSize', 22, 'FontWeight', 'bold');
    cb1 = colorbar; cb1.Label.String = 'Amplitude';

% (e) Local Similarity Map
nexttile;
    LO_imagesc(simi1, 100, 2, x, t); title('Local Similarity');
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
title(t_layout, 'MHA-RN Framework Evaluation: Residual and Similarity Analysis', ...
      'FontSize', 24, 'FontWeight', 'bold');

%%

