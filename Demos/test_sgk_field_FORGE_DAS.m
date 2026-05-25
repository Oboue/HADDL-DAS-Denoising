% =========================================================================
% SCRIPT: test_sgk_field_FORGE_DAS.m
% -------------------------------------------------------------------------
% Original Author:  Yangkang Chen (2022, 2023)
% Adapted by:       Oboué et al.,2026
% Purpose:          Baseline comparison using BP+SGK (Sequential 
%                   Generalization of K-SVD) for FORGE Field DAS Data.
% -------------------------------------------------------------------------
% Description:
%   This script implements the SGK dictionary learning algorithm as a 
%   benchmark to evaluate the performance of the proposed HADDL method.
%   It includes bandpass filtering followed by sparse coding (OMP) and 
%   dictionary update via SGK using the MATseisdl toolbox.
%
% Reference:   Oboué et al., "A Hybrid Attention-Driven Deep Learning 
%              Framework for Denoising DAS Data", Geophysics, 2026.
%
% Copyright (c) 2026. All rights reserved.
%%
clc; clear; close all;
%% SECTION 1: PATH MANAGEMENT & ENVIRONMENT SETUP
% -------------------------------------------------------------------------
fprintf('--- Initializing FORGE Dataset: BP+SGK Benchmark ---\n');
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
fprintf('-> Ready for BP+SGK baseline execution.\n');

%% --- OUTPUT DIRECTORY SETUP ---
% Automatic folder creation for high-fidelity result storage
resultsFolder = 'Output_Field_FORGE_DAS_SGK';
if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end
%% --- 3. PRE-PROCESSING ---
% Apply bandpass filter to prepare the data for dictionary learning
d_bp = haddl_bandpass(dn, 0.0005, 0, 200, 6, 6, 0, 0);
d = d_bp;

%% --- 4. DCT DICTIONARY CONSTRUCTION ---
% Patch size parameters
l1 = 32; l2 = 1;
c1 = l1; c2 = 100; % Size of the 1D cosine dictionary (overcomplete if c2 > c1)

% Generating the DCT dictionary atoms
dct = zeros(c1, c2);
for k = 0:c2-1
    V = cos([0:c1-1]' * k * pi / c2);
    if k > 0
        V = V - mean(V);
    end
    dct(:, k+1) = V / norm(V);
end
D = dct;

%% --- 5. PATCH DECOMPOSITION & INITIAL SPARSE CODING ---
% Decompose the seismic section into overlapping patches
X = haddl_patch(d, 1, l1, 1, l1/2, 1);
nd = size(X, 2);
K = 3;  % Sparsity level
ph = 2; % Threshold parameter

% OMP (Orthogonal Matching Pursuit) using the initial DCT dictionary
fprintf('Starting initial OMP coding...\n');
tic
for i2 = 1:nd
    G(:, i2) = haddl_omp0(D, X(:, i2), K);
end
toc

% Apply Sparsity Thresholding
G = haddl_pthresh(G, 'ph', ph);
X2 = D * G;
d2 = haddl_patch_inv(X2, 1, n1, n2, l1, 1, l1/2, 1);

%% --- 6. SGK DICTIONARY LEARNING (CORE ALGORITHM) ---
% Set parameters for SGK (Sequential Generalization of K-SVD)
param.T = K;      % Target sparsity level
param.D = D;      % Initial Dictionary
param.niter = 30; % Number of iterations
param.mode = 1;   % Sparsity mode
param.K = c2;     % Number of atoms

fprintf('Starting SGK Dictionary Learning...\n');
tic
% Call SGK function to learn a data-adaptive dictionary
[Dsgk, Gsgk] = haddl_sgk(X, param);
toc

% Sparse representation and reconstruction using learned dictionary
Gsgk0 = Gsgk;
Gsgk = haddl_pthresh(Gsgk0, 'ph', ph);
X11 = Dsgk * Gsgk;
d11 = haddl_patch_inv(X11, 1, n1, n2, l1, 1, l1/2, 1);
sgkdas = d11; % Final denoised output

%% --- 7. VISUALIZATION OF RESULTS ---
% Time and space axis for plotting
t_plot = [0:n1] * 0.0005;
ngap = 50;
x_plot = 1:n2*5 + 4*ngap;

figure('units', 'normalized', 'Position', [0.0 0.0 0.5 0.5], 'color', 'w');

% Panel 1: Original Noisy Data
subplot(1, 3, 1); 
haddl_imagesc(dn(:, :), 98, 1, x_plot, t_plot(:)); hold on
ylabel('Time (s)', 'Fontsize', 14, 'fontweight', 'bold');
xlabel('Channel', 'Fontsize', 14, 'fontweight', 'bold');
title('Raw Data', 'Fontsize', 14);
set(gca, 'Linewidth', 2, 'Fontsize', 14, 'Fontweight', 'bold');
caxis([-100 100]);

% Panel 2: SGK Denoised Result
subplot(1, 3, 2); 
haddl_imagesc(sgkdas(:, :), 98, 1, x_plot, t_plot(:)); hold on
ylabel('Time (s)', 'Fontsize', 14, 'fontweight', 'bold');
xlabel('Channel', 'Fontsize', 14, 'fontweight', 'bold');
title('Denoised (SGK)', 'Fontsize', 14);
set(gca, 'Linewidth', 2, 'Fontsize', 14, 'Fontweight', 'bold');
caxis([-100 100]);

% Panel 3: Removed Noise (Residuals)
dn1 = dn - sgkdas;
subplot(1, 3, 3); 
haddl_imagesc(dn1(:, :), 98, 1, x_plot, t_plot(:)); hold on
ylabel('Time (s)', 'Fontsize', 14, 'fontweight', 'bold');
xlabel('Channel', 'Fontsize', 14, 'fontweight', 'bold');
title('Removed Noise', 'Fontsize', 14);
set(gca, 'Linewidth', 2, 'Fontsize', 14, 'Fontweight', 'bold');
caxis([-100 100]);

%% --- 8. DATA EXPORT & PERSISTENCE ---
% Saving the high-fidelity results for subsequent research steps
filename = fullfile(resultsFolder, sprintf('sgkdas.mat'));
save(filename, 'sgkdas');

fprintf('Processed data saved successfully to: %s\n', resultsFolder);