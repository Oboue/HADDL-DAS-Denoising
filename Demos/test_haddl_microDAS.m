% =========================================================================
% SCRIPT: test_haddl_microDAS.m
% -------------------------------------------------------------------------
% Author:      Oboué et al.
% Date:        2026
% Affiliation: Zhejiang University 
% Reference:   Oboué et al., "A Hybrid Attention-Driven Deep Learning 
%              Framework for Denoising DAS Data", Geophysics, 2026.
% -------------------------------------------------------------------------
% Description:
%   This script prepares and visualizes the microDAS dataset for processing 
%   with the proposed HADDL (Hybrid Attention-Driven Deep Learning) method.
%
%   Key steps:
%   1. Data loading and channel selection (175-550).
%   2. Temporal focusing (t >= 2.5 s) to capture microseismic events.
%   3. Amplitude normalization and average trace analysis for P-wave 
%      arrival identification.
%
% Copyright (c) 2026. All rights reserved.
% =========================================================================

clc; clear; close all;

%% SECTION 1: PATH MANAGEMENT & INITIALIZATION
% -------------------------------------------------------------------------
fprintf('--- Initializing microDAS Dataset Preparation ---\n');
addpath(genpath('../Field_data'));   
addpath(genpath('../subroutines')); % Necessary for MHA-RN functions   

% Ensures repeatability for research validation
rng(42, 'twister'); 

%% SECTION 2: DATA LOADING & SUBSET SELECTION
% -------------------------------------------------------------------------
fprintf('--- Loading microDAS Raw Data ---\n');

if exist('microDAS_data.mat', 'file')
    S = load('microDAS_data.mat');
    data_raw = S.data_raw;
    dt = S.dt;
    fprintf('-> Success: microDAS data loaded.\n');
else
    error('File microDAS_data.mat not found. Please check your path.');
end

% 1. Spatial Selection (Channel focusing)
chan_start = 175;
chan_end   = 550;
data_subset = data_raw(chan_start:chan_end, :);
[n_chan, n_time] = size(data_subset);

% 2. Temporal Windowing (Focusing on t >= 2.5 s)
time_window = (0:n_time-1) * dt;
time_mask = time_window >= 2.5; 

data_subset_focused = data_subset(:, time_mask);
time_window_focused = time_window(time_mask);

%% SECTION 3: NORMALIZATION & QC PREPARATION
% -------------------------------------------------------------------------
% Normalizing data for deep learning input consistency
data_norm = data_subset_focused / max(abs(data_subset_focused(:)));
dn = data_norm;

% Single-trace average amplitude for P-wave identification
avg_trace = mean(dn, 1);
p_wave_time = 3.35; % Identified P-wave arrival

%% SECTION 4: VISUALIZATION (DATA INSPECTION)
% -------------------------------------------------------------------------
figure('Name', 'QC: microDAS Input Analysis', 'Position', [100 100 1200 900], 'Color', 'w');

% Plot 1: DAS Waveform (Space-Time Section)
subplot(2,1,1)
imagesc(chan_start:chan_end, time_window_focused, dn'); 
colormap(seis);
caxis([-1 1]);
colorbar;
xlabel('Channel'); ylabel('Time (s)');
title('DAS Input (Normalized, Channels 175-550, Time >= 2.5 s)');
set(gca, 'YDir', 'reverse', 'LineWidth', 1.5, 'FontSize', 12, 'FontWeight', 'bold');
ylim([min(time_window_focused) max(time_window_focused)]);

% Plot 2: Average Trace (Temporal Profile)
subplot(2,1,2)
plot(time_window_focused, avg_trace, 'k', 'LineWidth', 1.5, 'DisplayName', 'Average trace'); 
hold on;
xline(p_wave_time, '--r', 'LineWidth', 2, 'Label', 'P-wave arrival', ...
      'LabelOrientation', 'horizontal', 'FontSize', 12, 'FontWeight', 'bold');

xlabel('Time (s)'); ylabel('Normalized Amplitude');
title('Average Trace Analysis (P-wave Detection)');
set(gca, 'LineWidth', 1.5, 'FontSize', 12, 'FontWeight', 'bold');
xlim([2.5 5.12]); 
ylim([-1 1]);
legend('Location', 'northeast');
hold off;

fprintf('-> microDAS preprocessing complete. Ready for HADDL.\n');

%% SECTION 5: HADDL HYBRID ARCHITECTURE & ATTENTION PARAMETERS
% -------------------------------------------------------------------------
% This section defines the structural parameters for the multi-scale 
% dictionary and the Multi-Head Attention (MHA) mechanism.
% -------------------------------------------------------------------------

% 1. Input Dimensions
[n2, n1, n3] = size(dn); 

% 2. Patch Window & Stride (Spatial & Temporal)
% Using a 4x4 window to capture spatial-temporal seismic wave patterns
w1 = 4; 
w2 = 4;
w3 = 1;
s1z = 1;
s2z = 1;
s3z = 1;
inpsize = w1 * w2 * w3; 

%% 3. Hierarchical Dictionary Scaling (Levels 1 to 10)
% Progressive reduction of atoms to enforce sparse feature abstraction
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

%% 4. Multi-Head Attention (MHA) Specifications
% -------------------------------------------------------------------------
% These parameters govern the high-dimensional projection for global 
% dependency modeling within the DAS records.
% -------------------------------------------------------------------------
D_query = 512;          % Query projection dimension
D_key   = 512;          % Key projection dimension
D_value = 512;          % Value projection dimension
numHeads = 8;           % Parallel attention heads for diversity of features

% Dimension per individual head
D_head = D_query / numHeads; 

% Regularization for sparse weight distribution
l1Factor = 1e-8;  % L1 regularization factor for sparsity control

fprintf('-> HADDL config: %d heads, %d-layer dictionary, %d atoms/patch.\n', ...
        numHeads, 10, inpsize);

% Define layers
layers = [
    featureInputLayer(inpsize, 'Name', 'input')
];

% Initialize the layer graph
lgraph = layerGraph(layers);

% First encoder block
r0 = fullyConnectedLayer(D1, 'WeightL2Factor', l1Factor, 'Name', 'r0');
r0_relu = reluLayer('Name', 'r0_relu');
r0_dropout = dropoutLayer(0.0001, 'Name', 'r0_dropout'); 
lgraph = addLayers(lgraph, r0);
lgraph = addLayers(lgraph, r0_relu);
lgraph = addLayers(lgraph, r0_dropout);
lgraph = connectLayers(lgraph, 'input', 'r0');
lgraph = connectLayers(lgraph, 'r0', 'r0_relu');
lgraph = connectLayers(lgraph, 'r0_relu', 'r0_dropout');

% Second encoder block
r1 = fullyConnectedLayer(D2, 'WeightL2Factor', l1Factor, 'Name', 'r1');
r1_relu = reluLayer('Name', 'r1_relu');
r1_dropout = dropoutLayer(0.0001, 'Name', 'r1_dropout');
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
fc1_dropout = dropoutLayer(0.0001, 'Name', 'fc1_dropout');
lgraph = addLayers(lgraph, fc1);
lgraph = addLayers(lgraph, fc1_relu);
lgraph = addLayers(lgraph, fc1_dropout);
lgraph = connectLayers(lgraph, 'temp_scaling', 'fc1');
lgraph = connectLayers(lgraph, 'fc1', 'fc1_relu');
lgraph = connectLayers(lgraph, 'fc1_relu', 'fc1_dropout');

% Define fc2 block
fc2 = fullyConnectedLayer(D4, 'WeightL2Factor', l1Factor, 'Name', 'fc2');
fc2_relu = reluLayer('Name', 'fc2_relu');
fc2_dropout = dropoutLayer(0.0001, 'Name', 'fc2_dropout');
lgraph = addLayers(lgraph, fc2);
lgraph = addLayers(lgraph, fc2_relu);
lgraph = addLayers(lgraph, fc2_dropout);
lgraph = connectLayers(lgraph, 'fc1_dropout', 'fc2');
lgraph = connectLayers(lgraph, 'fc2', 'fc2_relu');
lgraph = connectLayers(lgraph, 'fc2_relu', 'fc2_dropout');

% Define subsequent fully connected, relu, and dropout layers
fc3 = fullyConnectedLayer(D5, 'WeightL2Factor', l1Factor, 'Name', 'fc3');
fc3_relu = reluLayer('Name', 'fc3_relu');
fc3_dropout = dropoutLayer(0.0001, 'Name', 'fc3_dropout');
lgraph = addLayers(lgraph, fc3);
lgraph = addLayers(lgraph, fc3_relu);
lgraph = addLayers(lgraph, fc3_dropout);
lgraph = connectLayers(lgraph, 'fc2_dropout', 'fc3');
lgraph = connectLayers(lgraph, 'fc3', 'fc3_relu');
lgraph = connectLayers(lgraph, 'fc3_relu', 'fc3_dropout');

fc4 = fullyConnectedLayer(D6, 'WeightL2Factor', l1Factor, 'Name', 'fc4');
fc4_relu = reluLayer('Name', 'fc4_relu');
fc4_dropout = dropoutLayer(0.0001, 'Name', 'fc4_dropout');
lgraph = addLayers(lgraph, fc4);
lgraph = addLayers(lgraph, fc4_relu);
lgraph = addLayers(lgraph, fc4_dropout);
lgraph = connectLayers(lgraph, 'fc3_dropout', 'fc4');
lgraph = connectLayers(lgraph, 'fc4', 'fc4_relu');
lgraph = connectLayers(lgraph, 'fc4_relu', 'fc4_dropout');

fc5 = fullyConnectedLayer(D7, 'WeightL2Factor', l1Factor, 'Name', 'fc5');
fc5_relu = reluLayer('Name', 'fc5_relu');
fc5_dropout = dropoutLayer(0.0001, 'Name', 'fc5_dropout');
lgraph = addLayers(lgraph, fc5);
lgraph = addLayers(lgraph, fc5_relu);
lgraph = addLayers(lgraph, fc5_dropout);
lgraph = connectLayers(lgraph, 'fc4_dropout', 'fc5');
lgraph = connectLayers(lgraph, 'fc5', 'fc5_relu');
lgraph = connectLayers(lgraph, 'fc5_relu', 'fc5_dropout');

fc6 = fullyConnectedLayer(D8, 'WeightL2Factor', l1Factor, 'Name', 'fc6');
fc6_relu = reluLayer('Name', 'fc6_relu');
fc6_dropout = dropoutLayer(0.0001, 'Name', 'fc6_dropout');
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
fc7_dropout = dropoutLayer(0.0001, 'Name', 'fc7_dropout');
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

% Connect the output t3o the regression layer
lgraph = connectLayers(lgraph, 'output', 'regression_output');

% Finalize the network structure
figure;
plot(lgraph);
analyzeNetwork(lgraph);
%%
outputFolder = 'Output_Field_microDAS_HADDL';
if ~exist(outputFolder, 'dir')  
    mkdir(outputFolder);
end
rng(42, 'twister'); % ensures repeatability of random operations
niter = 1; 
for i = 1:niter
    % Determine previous data for the current iteration
    if i == 1
        d_prev = dn; % Use the original data for the first iteration
    else
        d_prev = d1_denoised; % Use the denoised output from the previous iteration
    end
    %% Patch Generation
    X = haddl_patch3d(d_prev, 1, w1, w2, w3, s1z, s2z, s3z);
    X = double(X');  % Convert to double
    %% Apply Non-Local Means filtering only during the first iteration
    if i == 1
        minPatchSize = 21;  % Minimum size for imnlmfilt
        filterStrength = 0.0000000001;  % Degree of smoothing
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
    train_ratio = 0.9;  
    num_train = round(train_ratio * (length(indx) - lex));  
    X_train = X(indx(lex:end), :);  
    Y_train = X(indx(lex:end), :);   
    X_val = X(indx(1:num_train), :);  
    Y_val = X(indx(1:num_train), :);
    %% Training Options
    % 13.58 dB 
     lam = 1e-1;   % grid (include 0)

    batchsize = 50000;  
    options = trainingOptions('adam', ...
        'MaxEpochs',10, ...
        'Verbose', true, ...
        'MiniBatchSize', batchsize, ...
        'InitialLearnRate', 0.01, ...
        'L2Regularization', lam, ...   % <-- L2 regularization factor
        'Plots', 'training-progress');                    
    %%
    rng(42, 'twister'); % ensures repeatability of random operations
    %% Train Network
    net = trainNetwork(X_train, Y_train, lgraph, options);   
    %% Predict Using the Network
    outDN = haddl_DL_Predict(net, X, length(X), batchsize, 1);   
    %% Unpatch the Result to Get the Full Denoised Signal for the next iterations
    d1_denoised = haddl_patch3d_inv(outDN', 1, n2, n1, n3, w1, w2, w3, s1z, s2z, s3z); 
    %% Apply FK-Dip Filter After Unpatching during the first iteration
    %  if i == 1
    % % w = max(0.05, 0.05 - 0.001 * i) ;
    %                w = 0.03;  
    % % %     d1_denoised = denoise_with_fk_dip(d1_denoised, w, niter, nfw, i, n1, n2);
    % d1_denoised= d1_denoised - amf_fk_dip(d1_denoised,w);
    %%  end
%% SECTION 6: POST-PROCESSING & VISUAL QC (HADDL RESULTS)
% -------------------------------------------------------------------------
% This section evaluates the denoising performance using three metrics:
% 1. Visual inspection of the wavefield.
% 2. Residual analysis (Removed noise).
% 3. Local similarity analysis (Signal leakage check).
% -------------------------------------------------------------------------

% Updating state variables
d_prev = d1_denoised;
haddl_microDAS = d1_denoised;

%% 1. COMPARATIVE WAVEFIELD VISUALIZATION
figure('Name', 'HADDL Performance: Noisy vs Denoised vs Residual', ...
       'Position', [100 100 1200 900], 'Color', 'w');

% Subplot A: Noisy Input
subplot(1,3,1)
imagesc(chan_start:chan_end, time_window_focused, dn'); 
colormap(seis); caxis([-1 1]); colorbar;
xlabel('Channel'); ylabel('Time (s)');
title('Noisy Input', 'FontSize', 14);
set(gca, 'YDir', 'reverse', 'LineWidth', 1.5, 'FontWeight', 'bold');
ylim([min(time_window_focused) max(time_window_focused)]);

% Subplot B: HADDL Denoised Result
subplot(1,3,2)
imagesc(chan_start:chan_end, time_window_focused, d1_denoised'); 
colormap(seis); caxis([-1 1]); colorbar;
xlabel('Channel'); ylabel('Time (s)');
title('HADDL Result', 'FontSize', 14);
set(gca, 'YDir', 'reverse', 'LineWidth', 1.5, 'FontWeight', 'bold');
ylim([min(time_window_focused) max(time_window_focused)]);

% Subplot C: Removed Noise (Residual)
dn1 = dn - d1_denoised; % Calculate residual
subplot(1,3,3)
imagesc(chan_start:chan_end, time_window_focused, dn1'); 
colormap(seis); caxis([-1 1]); colorbar;
xlabel('Channel'); ylabel('Time (s)');
title('Removed Noise', 'FontSize', 14);
set(gca, 'YDir', 'reverse', 'LineWidth', 1.5, 'FontWeight', 'bold');
ylim([min(time_window_focused) max(time_window_focused)]);

%% 2. LOCAL SIMILARITY ANALYSIS (QC)
% Quantifying potential signal leakage into the noise section.
% -------------------------------------------------------------------------
rect = [20,20,1]; niter = 20; eps = 0; verb = 0; 
[simi1] = haddl_localsimi(dn - d1_denoised, d1_denoised, rect, niter, eps, verb);

figure('Name', 'QC: Local Similarity Map', 'units', 'normalized', ...
       'Position', [0.1 0.1 0.5 0.7], 'color', 'w');

imagesc(chan_start:chan_end, time_window_focused, simi1'); 
colormap(jet); % Standard for similarity
c = colorbar; 
c.Label.String = 'Local Similarity Coefficient'; 
c.Label.FontSize = 14;
caxis([0, 1]);

xlabel('Channel'); ylabel('Time (s)');
title('Local Similarity (Signal Integrity Check)', 'FontSize', 15);
set(gca, 'YDir', 'reverse', 'LineWidth', 2, 'FontSize', 14, 'FontWeight', 'bold');
ylim([min(time_window_focused) max(time_window_focused)]);

%% 3. AVERAGE TRACE COMPARISON (FIDELITY CHECK)
figure('Name', 'QC: Average Trace Fidelity', 'Position', [150 150 1000 600], 'Color', 'w');

% Calculate denoised average trace
avg_tracedenoised_norm = mean(d1_denoised, 1);

plot(time_window_focused, avg_trace, 'k', 'LineWidth', 1, 'DisplayName', 'Noisy Trace'); 
hold on;
plot(time_window_focused, avg_tracedenoised_norm, 'r', 'LineWidth', 1.5, 'DisplayName', 'HADDL Denoised');

% Mark P-wave arrival
xline(p_wave_time, '--b', 'LineWidth', 2, 'Label', 'P-wave arrival', ...
      'LabelOrientation', 'horizontal', 'FontSize', 12, 'FontWeight', 'bold');

xlabel('Time (s)'); ylabel('Normalized Amplitude');
title('Average Trace Comparison: HADDL vs Noisy', 'FontSize', 15);
grid on;
xlim([2.5 5.12]); 
ylim([-0.5 0.5]); % Focus on signal amplitude
set(gca, 'LineWidth', 1.5, 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'northeast');
hold off;

%% 4. DATA EXPORT
filename = fullfile(outputFolder, sprintf('haddl_microDAS%.mat', i));
save(filename, 'haddl_microDAS');
fprintf('-> Iteration %d results saved to: %s\n', i, filename);
%%
end

