% =========================================================================
% SCRIPT: test_sgk_microDAS.m 
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
%   This script implements the SGK dictionary learning algorithm to 
%   benchmark the performance of the proposed HADDL framework.
%   It features bandpass filtering, DCT dictionary initialization, 
%   and sparse coding via OMP.
%--------------------------------------------------------------------------
% Reference:   Oboué et al., "A Hybrid Attention-Driven Deep Learning 
%              Framework for Denoising DAS Data", Geophysics, 2026.
%
%% GNU General Public License Notice:
%   This program is free software: you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.

% Copyright (c) 2026. All rights reserved.
% =========================================================================

clc; clear; close all;

%% SECTION 1: PATH MANAGEMENT & TOOLBOX INITIALIZATION
% -------------------------------------------------------------------------
fprintf('--- Initializing Environment & Toolboxes ---\n');

addpath(genpath('../subroutines'));
addpath(genpath('../Field_data'));   

% Ensures repeatability for research validation
rng(42, 'twister'); 

%% SECTION 2: DATA LOADING & PRE-PROCESSING
% -------------------------------------------------------------------------
fprintf('--- Loading microDAS Dataset for SGK Benchmark ---\n');

if exist('microDAS_data.mat', 'file')
    S = load('microDAS_data.mat');
    data_raw = S.data_raw;
    dt = S.dt;
else
    error('File microDAS_data.mat not found. Check your Field_data folder.');
end

% 1. Spatial & Temporal Selection (Channels 175-550, t >= 2.5s)
chan_start = 175;
chan_end   = 550;
data_subset = data_raw(chan_start:chan_end, :);
[n_chan, n_time] = size(data_subset);

time_window = (0:n_time-1) * dt;
time_mask = time_window >= 2.5; 

data_subset_focused = data_subset(:, time_mask);
time_window_focused = time_window(time_mask);

% 2. Normalization
dn = data_subset_focused / max(abs(data_subset_focused(:)));
[n2, n1] = size(dn); % n2: channels, n1: time samples

% 3. Pre-filtering (Bandpass)
% Preparing the data for sparse dictionary learning
d_bp = haddl_bandpass(dn', 0.0005, 0, 200, 6, 6, 0, 0);
d = d_bp;

%% SECTION 3: SGK DICTIONARY LEARNING CORE
% -------------------------------------------------------------------------
% 1. Initial DCT Dictionary Construction
l1 = 32; l2 = 1;
c1 = l1; c2 = 100; 
D = zeros(c1, c2);
for k = 0:c2-1
    V = cos([0:c1-1]' * k * pi / c2);
    if k > 0, V = V - mean(V); end
    D(:, k+1) = V / norm(V);
end

% 2. Patch Decomposition & Sparse Coding (OMP)
X = haddl_patch(d, 1, l1, 1, l1/2, 1);
nd = size(X, 2);
K = 3; ph = 2;

fprintf('-> Running SGK Iterations (niter=30)...\n');
param.T = K; param.D = D; param.niter = 30; param.mode = 1; param.K = c2;

tic;
[Dsgk, Gsgk0] = haddl_sgk(X, param);
toc;

% 3. Reconstruction
Gsgk = haddl_pthresh(Gsgk0, 'ph', ph);
X_rec = Dsgk * Gsgk;
sgk_mirodas = haddl_patch_inv(X_rec, 1, n1, n2, l1, 1, l1/2, 1);

%% SECTION 4: VISUALIZATION & QC
% -------------------------------------------------------------------------
figure('Name', 'SGK Benchmark Results', 'Position', [100 100 1200 900], 'Color', 'w');

% Subplot 1: Noisy Data
subplot(1,3,1)
imagesc(chan_start:chan_end, time_window_focused, dn'); 
colormap(seis); caxis([-1 1]); colorbar;
title('Noisy Input', 'FontSize', 14);
ylabel('Time (s)'); xlabel('Channel');
set(gca, 'YDir', 'reverse', 'LineWidth', 1.5, 'FontWeight', 'bold');

% Subplot 2: SGK Denoised
subplot(1,3,2)
imagesc(chan_start:chan_end, time_window_focused, sgk_mirodas); 
colormap(seis); caxis([-1 1]); colorbar;
title('BP+SGK (Benchmark)', 'FontSize', 14);
set(gca, 'YDir', 'reverse', 'LineWidth', 1.5, 'FontWeight', 'bold');

% Subplot 3: Residuals
subplot(1,3,3)
imagesc(chan_start:chan_end, time_window_focused, dn' - sgk_mirodas); 
colormap(seis); caxis([-1 1]); colorbar;
title('Removed Noise', 'FontSize', 14);
set(gca, 'YDir', 'reverse', 'LineWidth', 1.5, 'FontWeight', 'bold');

%% SECTION 5: DATA EXPORT
% -------------------------------------------------------------------------
outputFolder = 'Output_Field_microDAS_SGK';
if ~exist(outputFolder, 'dir'), mkdir(outputFolder); end

filename = fullfile(outputFolder, 'sgk_mirodas.mat');
save(filename, 'sgk_mirodas');
fprintf('-> Processed data saved successfully to: %s\n', filename);