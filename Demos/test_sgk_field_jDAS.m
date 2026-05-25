% =========================================================================
% SCRIPT: test_sgk_field_jDAS.m
% -------------------------------------------------------------------------
% Original Author:  Yangkang Chen (2022, 2023)
% Adapted by:       Oboué et al.,2026
% Purpose:          Baseline comparison using BP+SGK (Sequential 
%                   Generalization of K-SVD) for jDAS Field Data.
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
%
% Copyright (c) 2026. All rights reserved.
% =========================================================================

clc; clear; close all;

%% SECTION 1: PATH MANAGEMENT & TOOLBOX INITIALIZATION
% -------------------------------------------------------------------------
addpath(genpath('./subroutines'));
addpath(genpath('./Field_data'));   

% Ensures repeatability for benchmark consistency
rng(42, 'twister'); 

%% SECTION 2: DATA LOADING & PHYSICAL SCALING
% -------------------------------------------------------------------------
fprintf('--- Loading jDAS Field Dataset for Benchmark ---\n');

if exist('jDAS_data.mat', 'file')
    load('jDAS_data.mat');  
    [nt, nx] = size(jdas_data);
    fprintf('-> Success: jDAS data loaded (%d samples x %d channels).\n', nt, nx);
else
    error('File jDAS_data.mat not found. Check your Field_data folder.');
end

% Physical Coordinate Definitions
tmax = 42;          % seconds
xmax = 12e3;        % meters
time = linspace(0, tmax, nt);
distance = linspace(0, xmax, nx);

% Output Management
outputFolder = 'Output_Field_jDAS_SGK';
if ~exist(outputFolder, 'dir'), mkdir(outputFolder); end

%% SECTION 3: PRE-PROCESSING (BANDPASS FILTERING)
% -------------------------------------------------------------------------
% Initial noise suppression using a standard Butterworth bandpass filter
% before applying the dictionary learning stage.
d_bp = haddl_bandpass(jdas_data, 0.0005, 0, 200, 6, 6, 0, 0);
d = d_bp;

%% SECTION 4: SGK DICTIONARY LEARNING & SPARSE CODING
% -------------------------------------------------------------------------
% 1. Patch Extraction (l1 x l2)
l1 = 32; l2 = 1;
X = haddl_patch(d, 1, l1, 1, l1/2, 1);

% 2. DCT Dictionary Initialization (Overcomplete)
c1 = l1; c2 = 100;
dct = zeros(c1, c2);
for k = 0:c2-1
    V = cos([0:c1-1]' * k * pi / c2);
    if k > 0, V = V - mean(V); end
    dct(:, k+1) = V / norm(V);
end
D = dct;

% 3. Sparse Coding via OMP
fprintf('-> Performing Sparse Coding (OMP)...\n');
K = 3; ph = 2; nd = size(X, 2);
G = zeros(c2, nd); 
tic;
for i2 = 1:nd
    G(:, i2) = haddl_omp0(D, X(:, i2), K);
end
fprintf('-> OMP completed in %.2f seconds.\n', toc);

% 4. SGK Dictionary Update
fprintf('-> Starting SGK Iterations...\n');
param.T = K;        % Sparsity level
param.D = D;        % Initial dictionary
param.niter = 30;   % Number of iterations
param.mode = 1;     % Sparsity mode
param.K = c2;       % Number of atoms

tic;
[Dsgk, Gsgk] = haddl_sgk(X, param);
fprintf('-> SGK training completed in %.2f seconds.\n', toc);

% 5. Reconstruction with Hard Thresholding
Gsgk = haddl_pthresh(Gsgk, 'ph', ph);
X_recon = Dsgk * Gsgk;
[n1, n2] = size(d);
sgkjdas = haddl_patch_inv(X_recon, 1, n1, n2, l1, 1, l1/2, 1);

%% SECTION 5: VISUALIZATION & QUALITY CONTROL
% -------------------------------------------------------------------------
figure('Name', 'Benchmark: SGK Results (jDAS)', 'Position', [100 100 1200 800], 'Color', 'w');
data_plot = {jdas_data, sgkjdas, jdas_data - sgkjdas};
plot_titles = {'Noisy Input', 'Denoised (BP+SGK Baseline)', 'Removed Noise (Residual)'};

for k = 1:3
    subplot(1, 3, k);
    imagesc(distance/1000, time, data_plot{k});
    colormap(seis);
    caxis([-5 5]);
    xlabel('Distance along cable (km)');
    ylabel('Time (s)');
    title(plot_titles{k});
    set(gca, 'YDir', 'reverse', 'LineWidth', 2, 'FontSize', 12, 'FontWeight', 'bold');
    colorbar;
end

%% SECTION 6: DATA EXPORT
% -------------------------------------------------------------------------
filename = fullfile(outputFolder, 'sgkjdas.mat');
save(filename, 'sgkjdas');
fprintf('-> Results saved to %s\n', filename);

