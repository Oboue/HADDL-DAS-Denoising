% =========================================================================
% SCRIPT: test_haddl_jDAS_processing.m
% -------------------------------------------------------------------------
% Author:      Oboué et al.
% Date:        2026
% Affiliation: Zhejiang University 
% Reference:   Oboué et al., "A Hybrid Attention-Driven Deep Learning 
%              Framework for Denoising DAS Data", Geophysics, 2026.
% -------------------------------------------------------------------------
% Description:
%   This script is dedicated to processing the jDAS field dataset using 
%   the proposed HADDL (Hybrid Attention-Driven Deep Learning) method.
%
%   It handles the loading of large-scale DAS records, physical coordinate 
%   mapping (Time in s, Distance in m), and prepares the data for 
%   high-fidelity denoising.
%
% Copyright (c) 2026. All rights reserved.
% =========================================================================

clc; clear; close all; 

%% SECTION 1: PATH MANAGEMENT & INITIALIZATION
% -------------------------------------------------------------------------
% Toolboxes and Subroutines
addpath(genpath('../subroutines')); 
addpath(genpath('../Field_data'));   

% Ensures repeatability of random operations within the HADDL framework
rng(42, 'twister'); 

%% SECTION 2: DATA LOADING & PHYSICAL DIMENSIONS
% -------------------------------------------------------------------------
fprintf('--- Loading jDAS Field Dataset ---\n');

% 1. Load the .mat file (contains variable "jdas_data")
if exist('jDAS_data.mat', 'file')
    load('jDAS_data.mat');  
    fprintf('-> Success: jDAS data loaded.\n');
else
    error('File jDAS_data.mat not found in the path.');
end

% 2. Prepare Data Dimensions
% Current format: [nt x nx] = (650 x 2048)
[nt, nx] = size(jdas_data);

% 3. Physical Axis Definition
tmax = 42;          % Total duration in seconds
xmax = 12e3;        % Total fiber length in meters (12 km)

time = linspace(0, tmax, nt);
distance = linspace(0, xmax, nx);

%% SECTION 3: INPUT PREPARATION FOR HADDL
% -------------------------------------------------------------------------
% Define the noisy input for the denoising algorithm
dn = jdas_data;
[n1, n2, n3] = size(dn);

% Optional: Working with channel x time transpose if needed for specific kernels
jdas_data_ct = jdas_data';  

fprintf('-> Input dimensions: %d samples x %d channels\n', n1, n2);
fprintf('-> Grid ready: dt = %.3f s, dx = %.1f m\n', time(2)-time(1), distance(2)-distance(1));

%%
%% SECTION 4: HADDL ARCHITECTURE & DICTIONARY PARAMETERS
% -------------------------------------------------------------------------
% This section defines the patch dimensions and the hierarchical structure 
% of the dictionary layers. The multi-scale approach (D1 to D10) allows 
% the HADDL framework to learn both coarse and fine seismic features.
% -------------------------------------------------------------------------

% 1. Patch Window Dimensions (Spatial & Temporal)
w1 = 2; 
w2 = 2;
w3 = 1;

% 2. Stride/Step Size for Patch Extraction
s1z = 1;
s2z = 1;
s3z = 1;

% Total Input Size per Patch
inpsize = w1 * w2 * w3;  

%% 3. Hierarchical Dictionary Size Definition (Multi-Scale)
% -------------------------------------------------------------------------
% The dictionary size decreases progressively to enforce sparse coding 
% and feature abstraction at each level of the HADDL framework.
% -------------------------------------------------------------------------
D1  = 40;                     % Initial dictionary atoms
D2  = ceil(D1 / 2);           % Level 2 abstraction
D3  = ceil(D2 / 2);           % Level 3
D4  = ceil(D3 / 2);           % Level 4
D5  = ceil(D4 / 2);           % Level 5
D6  = ceil(D5 / 2);           % Level 6
D7  = ceil(D6 / 2);           % Level 7
D8  = ceil(D7 / 2);           % Level 8
D9  = ceil(D8 / 2);           % Level 9
D10 = ceil(D8 / 2);           % Level 10 (Refinement stage)

fprintf('-> Architecture defined: %d-layer hierarchical dictionary.\n', 10);
fprintf('-> Patch size: [%d x %d x %d] (Total: %d atoms).\n', w1, w2, w3, inpsize);
%% Define dimensions and parameters
D_query = 512;
D_key = 512;
D_value = 512;
numHeads = 8;  % Number of attention heads
D_head = D_query / numHeads;  % Dimension per head (assuming D_query = D_key = D_value) 
%%
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

% Connect the output t3o the regression layer
lgraph = connectLayers(lgraph, 'output', 'regression_output');

% Finalize the network structure
figure;
plot(lgraph);
analyzeNetwork(lgraph);
%%
outputFolder = 'Output_Field_jDAS_HADDL';
if ~exist(outputFolder, 'dir')  
    mkdir(outputFolder);
end

%% ================== STEP 0: Initialization ==================
rng(42);  % Fix random seed for reproducibility

niter=1;
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

    % --- Variance-Based Patch Selection ---
    v = var(X');  
    [ord, indx] = sort(v);  
    lex = round(length(indx) * 0.25);                     
    train_ratio = 0.9;  
    num_train = round(train_ratio * (length(indx) - lex));  
    X_train = X(indx(lex:end), :);  
    Y_train = X(indx(lex:end), :);   
    X_val = X(indx(1:num_train), :);  
    Y_val = X(indx(1:num_train), :);

    % --- Training Options ---
    lam = 1e-1;   % grid (include 0)
    batchsize = 50000;  
    options = trainingOptions('adam', ...
        'MaxEpochs',100, ...
        'Verbose', true, ...
        'MiniBatchSize', batchsize, ...
        'InitialLearnRate', 0.01, ...
        'L2Regularization', lam, ...   % <-- L2 regularization factor
        'Plots', 'training-progress');    

    %% --- Train Network ---
    net = trainNetwork(X_train, Y_train, lgraph, options);   

    %% --- Predict Using the Network ---
    rng(42, 'twister'); 

    outDN = haddl_DL_Predict(net, X, length(X), batchsize, 1);   
    d1_denoised = haddl_patch3d_inv(outDN', 1, n1, n2, n3, w1, w2, w3, s1z, s2z, s3z); 
    %% --- Update input for next iteration ---
    d_prev = d1_denoised;  

    d_haddl_jDAS=d1_denoised;
%% Plot Figures

figure('Position',[100 100 1200 900]);
subplot(131);
% Plot
imagesc(distance/1000, time, jdas_data); % distance in km
colormap(seis);
colorbar;
xlabel('Distance along cable (km)');
ylabel('Time (s)');
title('Noisy (Time × Channel)');
        caxis([-5 5]);
set(gca, 'YDir', 'reverse'); % optional: flip time downward

subplot(132);
imagesc(distance/1000, time, d_haddl_jDAS); % distance in km
colormap(seis);
colorbar;
xlabel('Distance along cable (km)');
ylabel('Time (s)');
title('Denoised data (Time × Channel)');
        caxis([-5 5]);
set(gca, 'YDir', 'reverse'); % optional: flip time downward

dn1=jdas_data-d_haddl_jDAS;
subplot(133);
% imagesc(1:size_channel, 1:size_time, dn1');  
imagesc(distance/1000, time, dn1); % distance in km
% here transpose again because imagesc expects rows=Y
colormap(seis);   % if not available, use jet or parula
colorbar;
xlabel('Channel index');
ylabel('Time samples');
        caxis([-5 5]);
title('Removed moise (Channel × Time)');
set(gca, 'YDir', 'reverse'); % optional: flip time downward

        %% Save denoised data
        filename = fullfile(outputFolder, sprintf('d_haddl_jDAS%d.mat', i));
        save(filename, 'd_haddl_jDAS');
%%
end
%%
% DAS data: time x channel
[nt, nx] = size(jdas_data);

% Physical time axis
tmax = 45;  % seconds
time = linspace(0, tmax, nt);

% Select a subset of channels to plot
n_traces = 15;
channels_idx = round(linspace(1, nx, n_traces));

% Amplitude scaling factor (shared for fair comparison)
scale = 0.5 * max([abs(jdas_data(:, channels_idx)), abs(d1_denoised(:, channels_idx))], [], 'all');

% Plot input and denoised traces
figure('Position',[100 100 1200 900]); hold on;

for i = 1:length(channels_idx)
    ch = channels_idx(i);

    % Input (black)
    plot(time, jdas_data(:, ch)/scale + i, 'k', 'LineWidth', 1.2);

    % Denoised (red, dashed)
    plot(time, d_haddl_jDAS(:, ch)/scale + i, 'r--', 'LineWidth', 1.2);
end

% Define P-wave arrival time (example)
% p_wave_time = 12.3; % seconds
% xline(p_wave_time, '--b', 'LineWidth', 1.5, 'Label', 'P-wave', ...
%     'LabelOrientation', 'horizontal', 'Color', 'b');

xlabel('Time (s)');
ylabel('Trace index (offset for visualization)');
title(['DAS Input vs Denoised Traces (Channels ', ...
    num2str(channels_idx(1)), '–', num2str(channels_idx(end)), ')']);
legend('Input', 'Denoised', 'P-wave arrival');
grid on;
hold off;
%%
% DAS data: time x channel
[nt, nx] = size(jdas_data);
time = linspace(0, 45, nt);

% Select channels
n_traces = 5;
channels_idx = round(linspace(1, nx, n_traces));

% Common scaling factor (across both datasets)
scale = 0.5 * max([abs(jdas_data(:, channels_idx)), abs(d_haddl_jDAS(:, channels_idx))], [], 'all');

figure('Position',[100 100 1200 900]); hold on;
for i = 1:length(channels_idx)
    ch = channels_idx(i);

    % Input in black
    plot(time, jdas_data(:,ch)/scale + i, 'k', 'LineWidth', 1.2);

    % Denoised in red
    plot(time, d_haddl_jDAS(:,ch)/scale + i, 'r', 'LineWidth', 1.2);
end

xline(12.3, '--b', 'LineWidth', 1.5, 'Label', 'P-wave');
xlabel('Time (s)'); ylabel('Trace index');
title('DAS Wiggle Plot (Input: black, Denoised: red, Same Amplitude Scale)');
grid on; hold off;

%% Plot Figures

figure('Position',[100 100 1200 900]); hold on;

% Dummy handles for legend
h_in = plot(NaN, NaN, 'k', 'LineWidth', 1.2);
h_de = plot(NaN, NaN, 'r', 'LineWidth', 1.2);

for i = 1:length(channels_idx)
    ch = channels_idx(i);

    % Normalize input trace
    in_trace = jdas_data(:,ch) / max(abs(jdas_data(:,ch)));
    % Normalize denoised trace
    de_trace = d_haddl_jDAS(:,ch) / max(abs(d_haddl_jDAS(:,ch)));

    % Input in black
    plot(time, in_trace + i, 'k', 'LineWidth', 1.2);

    % Denoised in red
    plot(time, de_trace + i, 'r', 'LineWidth', 1.2);
end

xline(12.3, '--b', 'LineWidth', 1.5, 'Label', 'P-wave');

xlabel('Time (s)');
ylabel('Trace index');
title('DAS Wiggle Plot (Input vs Denoised, Normalized per Trace)');
grid on;

% Legend
legend([h_in h_de], {'Input (raw)', 'Denoised'}, 'Location', 'best');

hold off;


