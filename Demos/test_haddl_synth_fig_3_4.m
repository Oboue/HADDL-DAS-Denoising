% =========================================================================
% SCRIPT: test_haddl_synth_fig_3_4.m
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
%   This script reproduces Figures 3 and 4 of the manuscript using 
%   SYNTHETIC DATA, demonstrating the iterative refinement capabilities 
%   of the proposed HADDL framework for DAS denoising.
%
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
% REPRODUCIBILITY AND INITIALIZATION
% -------------------------------------------------------------------------
rng(42, 'twister'); % Ensures repeatability of random operations

%% SECTION 1: SYNTHETIC DATA LOADING AND GLOBAL PARAMETERS
% -------------------------------------------------------------------------
% Path: /haddl/Demos/test_haddl_synth.m
% -------------------------------------------------------------------------

% 1. Manage Paths (Cross-platform compatible)
addpath(genpath(fullfile('..', 'Synth_data')));
addpath(genpath(fullfile('..', 'Subroutines')));

% 2. Load Ground Truth Data
if exist('micro_sf_3001_334_3.mat', 'file')
    load('micro_sf_3001_334_3.mat'); % Variable 'data' enters workspace
    fprintf('-> Success: Ground truth loaded.\n');
else
    error('Ground Truth file missing in /Synth_data/');
end

% 3. Load Noisy Data
if exist('dnoiseSynthDAS.mat', 'file')
    load('dnoiseSynthDAS.mat'); % Variable 'dnoiseSynthDAS' enters workspace
    fprintf('-> Success: Noisy data loaded.\n');
else
    error('Noisy data file missing in /Synth_data/');
end

% 4. Scaling and Dimension Alignment
d = haddl_scale(data(:,:,1), 2);
[n1, n2, n3] = size(d); 

dn = dnoiseSynthDAS;
% Safety: match dimensions to avoid yc_snr errors
if size(dn,1) ~= n1 || size(dn,2) ~= n2
    dn = dn(1:n1, 1:n2);
end

% 5. Metrics & Grid
psnr_Noisy = haddl_snr(d, dn, 2); 
dt = 0.004; 
t  = (0:n1-1) * dt; 
x  = 1:n2;

fprintf('-> Initial SNR: %.2f dB\n', psnr_Noisy);
fprintf('-> Grid ready: [%d samples x %d channels]\n', n1, n2);
%% SECTION 2: PATCH AND HYPERPARAMETERS
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
outputFolder = 'DenoisedSynth2FORGEDASData_reviewed_02052026';
if ~exist(outputFolder, 'dir'), mkdir(outputFolder); end

% Loop Parameters
niter = 10; 
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
    % ---------------------------------------------------------------------
    outDN       = haddl_DL_Predict(net, X, length(X), batchsize, 1);   
    d1_denoised = haddl_patch3d_inv(outDN', 1, n1, n2, n3, w1, w2, w3, s1z, s2z, s3z); 

    %% CONDITIONAL FK-DIP FILTERING (Iteration 1 Only)
    % ---------------------------------------------------------------------
    if i == 1
        w = 0.000001;  
        d1_denoised = d1_denoised - haddl_fk_dip(d1_denoised, w);
    end
    
    d_prev = d1_denoised;

    %% PERFORMANCE EVALUATION & AUTOMATED DATA CAPTURE
    % ---------------------------------------------------------------------
    rng(42, 'twister'); % Maintain repeatability
    psnr_Noisy    = haddl_snr(d, dn, 2);
    psnr_denoised = haddl_snr(d, d1_denoised, 2);
    
    SNR_values(i) = psnr_denoised; % Capture SNR for plotting
    
    % Store specific iterations for Figures 3 and 4
    if i == 1, d1_iter1 = d1_denoised; end
    if i == 2, d1_iter2 = d1_denoised; end
    if i == 5, d1_iter5 = d1_denoised; end
    
    fprintf('Iteration %d Summary:\n', i);
    fprintf('  - Input SNR: %.2f dB\n', psnr_Noisy);
    fprintf('  - Denoised SNR: %.2f dB\n', psnr_denoised);
end

%% SECTION 5: FIGURE 3 - WAVEFORM COMPARISON AND SNR CONVERGENCE
% -------------------------------------------------------------------------
iterations = 1:10;
best_iter = 2; 

figure('Name', 'Figure 3: Iterative Refinement Analysis', ...
       'units', 'normalized', 'Position', [0.1 0.1 0.8 0.6], 'color', 'w');
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% (a) Clean Reference Waveform
nexttile(1);
    haddl_imagesc(d, 100, 2, x, t); title('Clean');
    ylabel('Time (s)'); xlabel('Channel');
    text(-0.15, 1, '(a)', 'Units', 'normalized', 'FontSize', 20, 'FontWeight', 'bold');

% (b) Noisy Input Waveform
nexttile(2);
    haddl_imagesc(dn, 100, 2, x, t); title('Noisy');
    ylabel('Time (s)'); xlabel('Channel');
    text(-0.15, 1, '(b)', 'Units', 'normalized', 'FontSize', 20, 'FontWeight', 'bold');

% (c) HADDL Result: Iteration 1
nexttile(3);
    haddl_imagesc(d1_iter1, 100, 2, x, t); 
    title(sprintf('HADDL Iter 1 (SNR = %.2f dB)', SNR_values(1)));
    ylabel('Time (s)'); xlabel('Channel');
    text(-0.15, 1, '(c)', 'Units', 'normalized', 'FontSize', 20, 'FontWeight', 'bold');

% (d) HADDL Result: Iteration 2 (Optimal)
nexttile(4);
    haddl_imagesc(d1_iter2, 100, 2, x, t); 
    title(sprintf('HADDL Iter 2 (SNR = %.2f dB)', SNR_values(2)));
    ylabel('Time (s)'); xlabel('Channel');
    text(-0.15, 1, '(d)', 'Units', 'normalized', 'FontSize', 20, 'FontWeight', 'bold');

% (e) HADDL Result: Iteration 5
nexttile(5);
    haddl_imagesc(d1_iter5, 100, 2, x, t); 
    title(sprintf('HADDL Iter 5 (SNR = %.2f dB)', SNR_values(5)));
    ylabel('Time (s)'); xlabel('Channel');
    text(-0.15, 1, '(e)', 'Units', 'normalized', 'FontSize', 20, 'FontWeight', 'bold');

% (f) SNR Evolution (Full 10 Iterations)
nexttile(6);
    plot(iterations, SNR_values, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'Color', [0 0.4470 0.7410]); 
    hold on;
    plot(best_iter, SNR_values(best_iter), 'rp', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
    xlabel('Iteration Number'); ylabel('SNR (dB)');
    title('SNR Evolution'); grid on;
    xlim([1 10]); ylim([min(SNR_values)-1 max(SNR_values)+1]);
    text(best_iter + 0.2, SNR_values(best_iter) + 0.2, sprintf('Chosen Iter %d', best_iter), 'Color', 'r');
    text(-0.15, 1, '(f)', 'Units', 'normalized', 'FontSize', 20, 'FontWeight', 'bold');

% Global Formatting for Figure 3
colormap(seis);
set(findobj(gcf, 'type', 'axes'), 'LineWidth', 2, 'FontSize', 16, 'CLim', [-0.5 0.5]);

%% SECTION 6: FIGURE 4 - RESIDUAL NOISE AND LOCAL SIMILARITY
% -------------------------------------------------------------------------
% Residual Computation
noise1 = dn - d1_iter1; 
noise2 = dn - d1_iter2; 
noise3 = dn - d1_iter5; 

% Local Similarity Analysis
rect = [20 20 1]; nsim_iter = 20; eps_val = 0; verb = 0;
simi1 = haddl_localsimi(noise1, d1_iter1, rect, nsim_iter, eps_val, verb); 
simi2 = haddl_localsimi(noise2, d1_iter2, rect, nsim_iter, eps_val, verb); 
simi3 = haddl_localsimi(noise3, d1_iter5, rect, nsim_iter, eps_val, verb); 

figure('Name', 'Figure 4: Residual and Similarity Mapping', ...
       'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.7], 'Color', 'w');

% Create a 2x3 layout for a clean, two-row comparison matrix
t_layout2 = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

%% --- ROW 1: DIFFERENCE SECTIONS (NOISY - DENOISED) ---
nexttile(1);
haddl_imagesc(noise1, 100, 2, x, t); title('Difference (Iter 1)');
ylabel('Time (s)'); xlabel('Channel');
text(-0.15, 1.05, '(a)', 'Units', 'normalized', 'FontSize', 20, 'FontWeight', 'bold');

nexttile(2);
haddl_imagesc(noise2, 100, 2, x, t); title('Difference (Iter 2)');
ylabel('Time (s)'); xlabel('Channel');
text(-0.15, 1.05, '(b)', 'Units', 'normalized', 'FontSize', 20, 'FontWeight', 'bold');

nexttile(3);
haddl_imagesc(noise3, 100, 2, x, t); title('Difference (Iter 5)');
ylabel('Time (s)'); xlabel('Channel');
text(-0.15, 1.05, '(c)', 'Units', 'normalized', 'FontSize', 20, 'FontWeight', 'bold');


%% --- ROW 2: LOCAL SIMILARITY MAPS ---
nexttile(4);
haddl_imagesc(simi1, 100, 2, x, t); title('Local Similarity (Iter 1)');
ylabel('Time (s)'); xlabel('Channel'); 
text(-0.15, 1.05, '(d)', 'Units', 'normalized', 'FontSize', 20, 'FontWeight', 'bold');

nexttile(5);
haddl_imagesc(simi2, 100, 2, x, t); title('Local Similarity (Iter 2)');
ylabel('Time (s)'); xlabel('Channel');
text(-0.15, 1.05, '(e)', 'Units', 'normalized', 'FontSize', 20, 'FontWeight', 'bold');

nexttile(6);
haddl_imagesc(simi3, 100, 2, x, t); title('Local Similarity (Iter 5)');
ylabel('Time (s)'); xlabel('Channel');
text(-0.15, 1.05, '(f)', 'Units', 'normalized', 'FontSize', 20, 'FontWeight', 'bold');


%% --- FINAL FORMATTING AND COLORBAR INTEGRATION ---
% -------------------------------------------------------------------------

% 1. Format Row 1 (Difference/Residual plots)
for k = 1:3
    ax1 = nexttile(k);
    set(ax1, 'Colormap', seis, 'CLim', [-0.5 0.5], 'LineWidth', 2, 'FontSize', 16, 'FontWeight', 'bold');
    
    % Add an amplitude colorbar to the final panel of the first row
    if k == 3
        cb1 = colorbar(ax1);
        ylabel(cb1, 'Residual Amplitude', 'FontSize', 12, 'FontWeight', 'bold');
    end
end

% 2. Format Row 2 (Local Similarity maps)
for k = 4:6
    ax2 = nexttile(k);
    set(ax2, 'Colormap', jet, 'CLim', [0 1], 'LineWidth', 2, 'FontSize', 16, 'FontWeight', 'bold');
    
    % Add the requested "Local Similarity" colorbar to the final panel (Tile 6)
    if k == 6
        cb2 = colorbar(ax2);
        ylabel(cb2, 'Local Similarity', 'FontSize', 12, 'FontWeight', 'bold');
    end
end

fprintf('-> Figure 4 paths and colorbars successfully formatted.\n');