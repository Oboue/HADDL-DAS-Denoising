% =========================================================================
% SCRIPT: test_sgk_synth.m
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
%   benchmark the performance of the proposed HADDL (Hybrid Attention-Driven 
%   Deep Learning) framework. It features bandpass filtering, DCT 
%   dictionary initialization, and sparse coding via OMP.

% Reference:   Oboué et al., "A Hybrid Attention-Driven Deep Learning 
%              Framework for Denoising DAS Data", Geophysics, 2026.
%%
% GNU General Public License Notice:
%   This program is free software: you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.
% =========================================================================
% Copyright (c) 2026. All rights reserved.
% =========================================================================

clc; clear; close all; 

%% SECTION 1: PATH MANAGEMENT & TOOLBOX INITIALIZATION
% -------------------------------------------------------------------------
fprintf('--- Initializing Environment for: HADDL Project Benchmark ---\n');
% (Suite du script...)

rng(42, 'twister'); % ensures repeatability

%% SECTION 1: DATA LOADING AND INITIALIZATION
% -------------------------------------------------------------------------
fprintf('--- Loading Dataset ---\n');
load micro_sf_3001_334_3.mat 
d_clean = haddl_scale(data(:,:,1), 2);
[n1, n2, n3] = size(d_clean);

load dnoiseSynthDAS.mat 
dn = dnoiseSynthDAS;

% Output Management
outputFolder = 'Output_Synth_SGK'; 
if ~exist(outputFolder, 'dir'), mkdir(outputFolder); end

% Temporal/Spatial parameters
dt = 0.004;
t = (0:n1-1) * dt; 
x = 1:n2;

psnr_Noisy = haddl_snr(d_clean, dn, 2);
fprintf('Initial Noisy SNR: %.2f dB\n', psnr_Noisy);

%% SECTION 2: PRE-PROCESSING (BANDPASS FILTER)
% -------------------------------------------------------------------------
% Applying bandpass filter to suppress out-of-band noise before learning
fprintf('Applying bandpass filter...\n');
d_filtered = haddl_bandpass(dn, 0.0005, 0, 200, 6, 6, 0, 0);

%% SECTION 3: DCT DICTIONARY CONFIGURATION
% -------------------------------------------------------------------------
l1 = 140; l2 = 1; % Patch dimensions (Temporal focus)
c1 = l1;          % 1D Dictionary size
c2 = 40;          % Number of atoms (Undercomplete representation)

% Manual construction of 1D DCT Dictionary
D = zeros(c1, c2);
for k = 0:c2-1
    V = cos((0:c1-1)' * k * pi / c2);
    if k > 0
        V = V - mean(V);
    end
    D(:, k+1) = V / norm(V);
end

%% SECTION 4: PATCH DECOMPOSITION
% -------------------------------------------------------------------------
fprintf('Extracting patches...\n');
% Overlapping patches: step = l1/2 (50% overlap)
X = haddl_patch(d_filtered, 1, l1, 1, l1/2, 1);

%% SECTION 5: DICTIONARY LEARNING (SGK)
% -------------------------------------------------------------------------
param.D = D;        % Initial DCT Dictionary
param.T = 1;        % Sparsity constraint (K-atoms per patch)
param.niter = 10;   % SGK Iterations
param.mode = 1;     
param.K = c2;       

fprintf('Starting SGK Dictionary Learning...\n');
tic
[Dsgk, Gsgk] = haddl_sgk(X, param);
toc

% Soft-thresholding of coefficients for reconstruction
ph = 2;
Gsgk_thresh = haddl_pthresh(Gsgk, 'ph', ph);
X_rec = Dsgk * Gsgk_thresh;

%% SECTION 6: RECONSTRUCTION AND EVALUATION
% -------------------------------------------------------------------------
fprintf('Reconstructing signal...\n');
d_denoised = haddl_patch_inv(X_rec, 1, n1, n2, l1, 1, l1/2, 1);
d_denoisedsgk = reshape(d_denoised, n1, n2, n3);

% Performance evaluation
psnr_denoised = haddl_snr(d_clean, d_denoisedsgk, 2);
fprintf('Denoised SNR (SGK): %.2f dB\n', psnr_denoised);

% Save results
save(fullfile(outputFolder, 'result_sgk.mat'), 'd_denoisedsgk', 'psnr_denoised');

%% SECTION 7: COMPREHENSIVE ANALYSIS (CLEAN, NOISY, SGK, RESIDUAL, SIMI)
% -------------------------------------------------------------------------
% 1. Data Preparation
% We use the result from Section 6
noise_sgk = dn - d_denoisedsgk; 

% 2. Local Similarity Analysis
% This evaluates if any signal was 'leaked' into the noise section
rect = [20 20 1]; nsim_iter = 20; eps_val = 0; verb = 0;
simi_sgk = haddl_localsimi(noise_sgk, d_denoisedsgk, rect, nsim_iter, eps_val, verb); 

% 3. Figure Initialization
figure('Name', 'Figure: SGK Performance Analysis', ...
       'units', 'normalized', 'Position', [0.1 0.1 0.8 0.8], 'color', 'w');
t_layout_sgk = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% --- ROW 1: WAVEFORM COMPARISON ---

% (a) Clean Data
nexttile;
    haddl_imagesc(d_clean, 100, 2, x, t); title('Clean (d)');
    ylabel('Time (s)'); xlabel('Channel');
    text(-0.2, 1.1, '(a)', 'Units', 'normalized', 'FontSize', 22, 'FontWeight', 'bold');

% (b) Noisy Input
nexttile;
    haddl_imagesc(dn, 100, 2, x, t); title('Noisy (dn)');
    ylabel('Time (s)'); xlabel('Channel');
    text(-0.2, 1.1, '(b)', 'Units', 'normalized', 'FontSize', 22, 'FontWeight', 'bold');

% (c) SGK Result
nexttile;
    haddl_imagesc(d_denoisedsgk, 100, 2, x, t); title('SGK Denoised');
    ylabel('Time (s)'); xlabel('Channel');
    text(-0.2, 1.1, '(c)', 'Units', 'normalized', 'FontSize', 22, 'FontWeight', 'bold');

% --- ROW 2: ERROR ANALYSIS ---

% (d) Difference Section (Residuals)
nexttile;
    haddl_imagesc(noise_sgk, 100, 2, x, t); title('Difference (dn - SGK)');
    ylabel('Time (s)'); xlabel('Channel');
    text(-0.2, 1.1, '(d)', 'Units', 'normalized', 'FontSize', 22, 'FontWeight', 'bold');
    cb1 = colorbar; cb1.Label.String = 'Amplitude';

% (e) Local Similarity Map
nexttile;
    haddl_imagesc(simi_sgk, 100, 2, x, t); title('Local Similarity (SGK)');
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
title(t_layout_sgk, 'SGK Dictionary Learning Evaluation: Residual and Similarity Analysis', ...
      'FontSize', 24, 'FontWeight', 'bold');