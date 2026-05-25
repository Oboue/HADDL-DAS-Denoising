% =========================================================================
% SCRIPT: test_mharn_microDAS_benchmark.m
% -------------------------------------------------------------------------
% Author:      Oboué et al.
% Date:        2026
% Affiliation: Zhejiang University
% Reference:   Oboué et al., "A Hybrid Attention-Driven Deep Learning 
%              Framework for Denoising DAS Data", Geophysics, 2026.
% -------------------------------------------------------------------------
% Description:
%   This script prepares and visualizes the microDAS dataset for processing 
%   with the MHA-RN (Multi-head Attention Regression Network) method.
%
%   The output of this script serves as a baseline benchmark to evaluate 
%   the performance gains of the proposed HADDL framework.
%
% Copyright (c) 2026. All rights reserved.
% =========================================================================

clc; clear; close all;

%% SECTION 1: PATH MANAGEMENT & INITIALIZATION
% -------------------------------------------------------------------------
fprintf('--- Initializing microDAS Dataset: MHA-RN Benchmark ---\n');

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
figure('Name', 'QC: microDAS Input Analysis (MHA-RN)', 'Position', [100 100 1200 900], 'Color', 'w');

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

fprintf('-> microDAS preprocessing complete. Ready for MHA-RN benchmark.\n');
%%
%% SECTION 5: ARCHITECTURAL CONFIGURATION (LARGE-SCALE PATCHES)
% -------------------------------------------------------------------------
% This section defines the hyperparameters for the multi-scale dictionary 
% and the Multi-Head Attention (MHA) mechanism, optimized for large-scale 
% spatial-temporal feature extraction (20x20 patches).
% -------------------------------------------------------------------------

% 1. Input Dimensions extraction
[n2, n1, n3] = size(dn); 

% 2. Patch & Stride Definition (Large-Scale Window)
% Using a 20x20 window to capture broad wavefield morphology
w1 = 20; 
w2 = 20;
w3 = 1;
s1z = 1;
s2z = 1;
s3z = 1;
inpsize = w1 * w2 * w3; % Total elements per patch: 400

%% 3. Hierarchical Dictionary Scaling (10-Level Abstraction)
% -------------------------------------------------------------------------
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

%% 4. Multi-Head Attention (MHA) & Regularization
% -------------------------------------------------------------------------
% High-dimensional projection parameters for global context modeling
D_query  = 512;          % Query projection size
D_key    = 512;          % Key projection size
D_value  = 512;          % Value projection size
numHeads = 8;            % Number of parallel attention heads

% Derived dimension per attention head
D_head = D_query / numHeads; 

% Regularization parameter
% L1 factor set to 0.001 to promote sparsity in high-dimensional feature space
l1Factor = 0.001;  

fprintf('-> Model Configured: %dx%d Patches | %d Heads | L1: %.4f\n', ...
        w1, w2, numHeads, l1Factor);

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
rng(42, 'twister'); % ensures repeatability of random operations%% numHeads = 2;  % Number of attention heads

outputFolder = 'Output_Field_microDAS_MHARN';
if ~exist(outputFolder, 'dir')  
    mkdir(outputFolder);
end
lam = 1e-1;   % grid (include 0)
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
%     %% Apply Non-Local Means filtering only during the first iteration
%     if i == 1
%         minPatchSize = 21;  % Minimum size for imnlmfilt
%         filterStrength = 0.0000000001;  % Degree of smoothing
%         for j = 1:size(X, 1)
%             patch = reshape(X(j, :), [w1, w2, w3]);
%             if all(size(patch) >= minPatchSize)
%                 patch_denoised = imnlmfilt(patch, 'DegreeOfSmoothing', filterStrength);
%             else
%                 patch_denoised = mean(patch(:)) * ones(size(patch));  
%             end    
%             X(j, :) = patch_denoised(:);  
%         end
%     end
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
    batchsize = 50000;  
    options = trainingOptions('adam', ...
        'MaxEpochs',25, ...
        'Verbose', true, ...
        'MiniBatchSize', batchsize, ...
        'InitialLearnRate', 0.01, ...
        'L2Regularization', lam, ...   % <-- L2 regularization factor
        'Plots', 'training-progress');                      
    %%
    %% Train Network
    net = trainNetwork(X_train, Y_train, lgraph, options);   
    %% Predict Using the Network
    outDN = haddl_DL_Predict(net, X, length(X), batchsize, 1);   
    %% Unpatch the Result to Get the Full Denoised Signal for the next iterations
    rng(42, 'twister'); % ensures repeatability of random operations%% numHeads = 2;  % Number of attention heads
    d1_denoised = haddl_patch3d_inv(outDN', 1, n2, n1, n3, w1, w2, w3, s1z, s2z, s3z); 
    %  end
    d_prev=d1_denoised;

    mharn_microDAS=d1_denoised;
    % psnr_Noisy = yc_snr(d,dn,2) % Noisy
    % psnr_denoised = yc_snr(d,d1_denoised,2) % Denoised 
    %% Compute SNR
    % if i > 1
        % snr_value = computeSNR(d1_denoised, dn);
        % fprintf('SNR after iteration %d: %.2f dB\n', i, psnr_denoised)
    %% Compute SSIM
    % % if i > 1
    %     ssim_value = ssim(d1_denoised, d_prev);
    %     fprintf('SSIM after iteration %d: %.4f\n', i, ssim_value)
    % % end
    %%
%%
figure('Position',[100 100 1200 900]);

% DAS waveform
subplot(1,3,1)
imagesc(chan_start:chan_end, time_window_focused, dn'); 
colormap(seis);
caxis([-1 1]);
colorbar;
xlabel('Channel'); ylabel('Time (s)');
title('Noisy');
ylim([min(time_window_focused) max(time_window_focused)]);

% DAS waveform
subplot(1,3,2)
imagesc(chan_start:chan_end, time_window_focused, mharn_microDAS'); 
colormap(seis);
caxis([-1 1]);
colorbar;
xlabel('Channel'); ylabel('Time (s)');
title('HADDL');
ylim([min(time_window_focused) max(time_window_focused)]);

dn1=dn-d1_denoised;

subplot(1,3,3)
imagesc(chan_start:chan_end, time_window, dn1'); 
colormap(seis);       % or jet/redblue if seismic unavailable
caxis([-1 1]);        % normalized amplitude
colorbar;
xlabel('Channel');
ylabel('Time (s)');
title('Removed noise');
% Time axis from 0 → max
ylim([0 max(time_window)]);
%% Save denoised data
filename = fullfile(outputFolder, sprintf('mharn_microDAS%d.mat', i));
save(filename, 'mharn_microDAS');
%% local similarity
rect=[20,20,1];niter=20;eps=0;verb=0; 
[simi1]=haddl_localsimi(dn-mharn_microDAS,mharn_microDAS,rect,niter,eps,verb);

figure('units','normalized','Position',[0.0 0.0 0.6, 1],'color','w');
imagesc(chan_start:chan_end, time_window_focused, simi1');
% colorbar;
xlabel('Channel'); ylabel('Time (s)');
title('DAS Data (Normalized, Channels 150–575, Time ≥ 3 s)');
ylim([min(time_window_focused) max(time_window_focused)]);

% ylim([0 max(time_window)]); 
colormap(jet);
c = colorbar;c.Label.String = 'Local similarity';c.Label.FontSize = 16;%c.Label.FontWeight = bold;
caxis([0,1]);
xlabel('Channel');
ylabel('Time (s)');

set(gca,'Linewidth',2,'Fontsize',16);
%%
figure('Position',[100 100 1200 900]);

plot(time_window_focused, avg_trace,'k','LineWidth',1); hold on;
xline(p_wave_time,'--r','LineWidth',2,'Label','P-wave arrival','LabelOrientation','horizontal');
xlabel('Time (s)'); ylabel('Normalized Amplitude');
ylim([-1 1]); 
% grid on;
xlim([2.5 5.12]);           % focus on 0 → 5.5 s
ylim([-0.5 0.5]);
title('Single Trace DAS Amplitude (Average Channels 150–575, Time ≥ 3 s)');
legend('Average trace','P-wave arrival');

avg_tracedenoised_norm = mean(mharn_microDAS,1);

plot(time_window_focused, avg_tracedenoised_norm,'r','LineWidth',1); hold on;
xline(p_wave_time,'--r','LineWidth',2,'Label','P-wave arrival','LabelOrientation','horizontal');
xlabel('Time (s)'); ylabel('Normalized Amplitude');
ylim([-1 1]); 
% grid on;
xlim([2.5 5.12]);           % focus on 0 → 5.5 s
ylim([-0.5 0.5]);
title('Single Trace DAS Amplitude (Average Channels 150–575, Time ≥ 3 s)');
legend('Average trace','P-wave arrival');

%%
end
