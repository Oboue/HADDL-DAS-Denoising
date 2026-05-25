% =========================================================================
% SCRIPT: plot_haddl_microDAS_fig_14_17.m
% -------------------------------------------------------------------------
% Article Title: "A Hybrid Attention-Driven Deep Learning Framework for 
%                Denoising DAS Data"
% -------------------------------------------------------------------------
% Authors:      Oboué, Y. A. S. I., Chen, Y., & Chen, Y.
% Date:         May 2026
% Affiliations: Zhejiang University | University of Texas at Austin
% Journal:      Geophysics 
% -------------------------------------------------------------------------
%% Description:
%   This script reproduces Figures 14 to 17 for the microseismic DAS dataset.
%   It compares the proposed HADDL framework against BP+SGK, SSDL, and MHA-RN.
% 
%   NOTE ON SSDL DATA:
%   The SSDL (Self-Supervised Deep Learning) benchmark results were generated 
%   using the official Python/Jupyter Notebook framework provided by Saad et 
%   al. [4]. The denoised outputs from their pipeline were exported and saved 
%   as 'ssdlmicroDAS.mat' for direct plotting and analysis within this 
%   MATLAB environment.
%
%   PREREQUISITES:
%   Before running this script, ensure the following field processing 
%   scripts have been executed to generate the required baseline results:
%   1. test_haddl_microDAS.m (Proposed HADDL)
%   2. test_sgk_microDAS.m   (BP+SGK Baseline)
%   3. test_mharn_microDAS.m (MHA-RN Baseline)
% =========================================================================
%%
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
%%
clc; clear; close all; 
%% SECTION 1: PATH MANAGEMENT & ENVIRONMENT SETUP 
% -------------------------------------------------------------------------
fprintf('--- Initializing Benchmark Environment ---\n');

% Toolboxes and Subroutines
addpath(genpath('../subroutines')); 
addpath(genpath('../Field_data'));   

% Results Folders (Output from individual method scripts)
addpath(genpath('./Output_Field_microDAS_SGK')); 
addpath(genpath('./Output_Field_microDAS_SSDL'));
addpath(genpath('./Output_Field_microDAS_MHARN'));
addpath(genpath('./Output_Field_microDAS_HADDL'));

% Ensures repeatable results for plotting and analysis
rng(42, 'twister'); 

%% SECTION 2: REFERENCE DATA LOADING & PRE-PROCESSING
% -------------------------------------------------------------------------
fprintf('--- Loading Reference microDAS Data ---\n');

if exist('microDAS_data.mat', 'file')
    S = load('microDAS_data.mat');
    data_raw = S.data_raw;
    dt = S.dt;
else
    error('Reference file microDAS_data.mat not found.');
end

% 1. Spatial & Temporal Selection
chan_start = 175;
chan_end   = 550;
data_subset = data_raw(chan_start:chan_end, :);
[n_chan, n_time] = size(data_subset);

time_window = (0:n_time-1) * dt;
time_mask = time_window >= 2.5; % Focalization on microseismic events

data_subset_focused = data_subset(:, time_mask);
time_window_focused = time_window(time_mask);

% 2. Relative Normalization (Crucial for fair comparison across methods)
norm_factor = max(abs(data_subset_focused(:)));
dn_noisy = (data_subset_focused / norm_factor)'; % Transposed to [Time x Channels]

%% SECTION 3: LOADING BENCHMARK RESULTS
% -------------------------------------------------------------------------
fprintf('--- Loading Competitor Method Results ---\n');

% Loading SGK
if exist('sgk_mirodas.mat', 'file'), load('sgk_mirodas.mat'); else warning('SGK result missing.'); end

% Loading SSDL (Ensure correct variable name from Python/Matlab bridge)
if exist('ssdlmicroDAS.mat', 'file'), load('ssdlmicroDAS.mat'); else warning('SSDL result missing.'); end

% Loading MHA-RN
if exist('mharn_microDAS1.mat', 'file'), load('mharn_microDAS1.mat'); else warning('MHA-RN result missing.'); end

% Loading HADDL (Proposed Method)
if exist('haddl_microDAS.mat', 'file'), load('haddl_microDAS.mat'); else warning('HADDL result missing.'); end

% Data aggregation for plotting loops
% Note: Using transpose (') where necessary to match [Time x Channels]
data_list = {dn_noisy, sgk_mirodas, ssdl, mharn_microDAS', haddl_microDAS'};
titles = {'Noisy', 'BP+SGK', 'SSDL', 'MHA-RN', 'HADDL (Proposed)'};
labels = {'(a)', '(b)', '(c)', '(d)', '(e)'};
chan_range = chan_start:chan_end;

%% SECTION 4: ROI ZOOM DEFINITION (EVENT CAPTURE)
% -------------------------------------------------------------------------
% Focus window on the microseismic event around 3.35s
% Format: [Channel_start, Time_start, Channel_width, Time_width]
rect_pos = [400, 3.1, 100, 0.5]; 

fprintf('-> Success: Environment ready. Proceeding to Figure generation.\n');

%% SECTION 7: GLOBAL COMPARISON (GLOBAL VIEW 2x3) WITH ROI FRAMES
% -------------------------------------------------------------------------
% This section generates a comprehensive 2x3 panel comparison to evaluate
% the performance of all benchmarked methods side-by-side.
% -------------------------------------------------------------------------

figure('Name', 'Figure: Global Benchmark microDAS Comparison', ...
       'Position', [50 50 1400 900], 'Color', 'w');

% Use tiledlayout for better control over spacing and common colorbars
tlo = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

for k = 1:5
    ax = nexttile;
    
    % Wavefield Visualization
    % data_list{k} must be [Time x Channels]
    imagesc(chan_range, time_window_focused, data_list{k}); 
    
    % Geophysical Formatting
    colormap(ax, seis);
    caxis([-0.8 0.8]); % Slightly tightened for better contrast on micro-events
    set(gca, 'YDir', 'reverse', 'LineWidth', 2, 'FontSize', 12, 'FontWeight', 'bold');
    
    title(titles{k}, 'FontSize', 16, 'FontWeight', 'bold');
    
    % Axis Labels (Only for the left-most and bottom-most plots to keep it clean)
    if k == 1 || k == 4
        ylabel('Time (s)', 'FontSize', 14);
    end
    if k >= 3
        xlabel('Channel index', 'FontSize', 14);
    end

    % Annotations and Labels
    hold on;
    % Place identifying letter (a, b, c...)
    text(-0.095, 1.06, labels{k}, 'Units', 'normalized', ...
         'FontSize', 22, 'FontWeight', 'bold', 'Color', 'k');
    
    % Draw the ROI (Region of Interest) rectangle for zoom reference
    % rect_pos = [Channel_start, Time_start, Width_chan, Width_time]
    rectangle('Position', rect_pos, 'EdgeColor', 'r', ...
              'LineWidth', 2.5, 'LineStyle', '-');
    hold off;
end

% Title and Global Labels
% title(tlo, 'Comparative Analysis of Denoising Frameworks on microDAS Data', ...
%       'FontSize', 18, 'FontWeight', 'bold');

% Common Colorbar for the normalized domain
% cb = colorbar; 
% cb.Layout.Tile = 'east'; 
% cb.Label.String = 'Normalized Relative Amplitude';
% cb.Label.FontSize = 14; 
% cb.Label.FontWeight = 'bold';

fprintf('-> Global benchmark figure generated successfully.\n');
%% SECTION 8: ZOOM-IN SECTIONS (FOCUS ON ROI PRESERVATION)
% -------------------------------------------------------------------------
% This section provides a high-resolution zoom into the Region of Interest
% (ROI) to evaluate signal continuity and structural preservation.
% -------------------------------------------------------------------------

figure('Name', 'Figure: Zoom-In Reflector Analysis (ROI)', ...
       'Position', [100 100 1400 900], 'Color', 'w');

tlo_zoom = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% --- DÉFINITION DES POSITIONS DES CIBLES (Seismic Events) ---
% These coordinates point to specific weak reflectors to verify preservation
arrow_pos = [455, 3.16; 
             415, 3.26; 
             475, 3.32; 
             495, 3.43; 
             445, 3.58; 
             490, 3.56];

% Calculate physical zoom limits from previously defined rect_pos
zCh = [rect_pos(1), rect_pos(1) + rect_pos(3)];
zT  = [rect_pos(2), rect_pos(2) + rect_pos(4)];

for k = 1:5
    ax = nexttile;
    
    % Display data with spatial-temporal constraints
    imagesc(chan_range, time_window_focused, data_list{k}); 
    xlim(zCh); 
    ylim(zT);
    
    % Format axis
    colormap(ax, seis); 
    caxis([-0.7 0.7]); % Tightened contrast for better visibility of weak events
    set(gca, 'YDir', 'reverse', 'LineWidth', 2.5, 'FontSize', 12, 'FontWeight', 'bold');
    
    title(titles{k}, 'FontSize', 16, 'FontWeight', 'bold');
    
    hold on;
    % --- PLOTTING CUSTOM ANNOTATION ARROWS ---
    % Drawing manual arrows to highlight key seismic features across all subplots
    for i = 1:size(arrow_pos, 1)
        x_tip = arrow_pos(i,1);
        y_tip = arrow_pos(i,2);
        
        % Arrow body: Short inclined segment
        plot([x_tip+5, x_tip], [y_tip-0.03, y_tip], 'r', 'LineWidth', 2.5);
        
        % Arrow head: Triangular marker
        plot(x_tip, y_tip, 'r>', 'MarkerFaceColor', 'r', 'MarkerSize', 8);
    end
    
    % Subplot identifier (a, b, c...)
    text(-0.095, 1.06, labels{k}, 'Units', 'normalized', ...
         'FontSize', 22, 'FontWeight', 'bold');
    hold off;
end

% Global annotations for the zoom layout
xlabel(tlo_zoom, 'Channel', 'FontSize', 16, 'FontWeight', 'bold');
ylabel(tlo_zoom, 'Time (s)', 'FontSize', 16, 'FontWeight', 'bold');

% title(tlo_zoom, 'High-Resolution ROI Zoom: Signal Preservation Benchmark', ...
%       'FontSize', 18, 'FontWeight', 'bold');

fprintf('-> ROI zoom figure generated. Signal integrity validated.\n');
%%
%% SECTION 9: ADVANCED QC - RESIDUAL ANALYSIS & LOCAL SIMILARITY
% -------------------------------------------------------------------------
% This section quantifies Signal Leakage. A perfect denoising would show:
% 1. Residuals: Only incoherent noise, no coherent seismic events.
% 2. Local Similarity: Values close to 0 (indicating no signal in the noise).
% -------------------------------------------------------------------------

fprintf('--- Running Advanced QC Analysis (Residuals & Leakage) ---\n');

% 1. Energy Residual Calculation
% Compute the difference between original noisy data and denoised results
residuals = {dn_noisy - sgk_mirodas, ...
             dn_noisy - ssdl, ...
             dn_noisy - mharn_microDAS', ...
             dn_noisy - haddl_microDAS'};

% 2. Local Similarity Parameters
rect = [20 20 1]; niter = 20; eps = 0; verb = 0;

% Calculation of Local Similarity between denoised signal and removed noise
% High similarity = High leakage (Bad) | Low similarity = High fidelity (Good)
s1 = haddl_localsimi(sgk_mirodas,          residuals{1}, rect, niter, eps, verb);
s2 = haddl_localsimi(ssdl,                 residuals{2}, rect, niter, eps, verb);
s3 = haddl_localsimi(mharn_microDAS',      residuals{3}, rect, niter, eps, verb);
s4 = haddl_localsimi(haddl_microDAS',      residuals{4}, rect, niter, eps, verb);

simi_list = {s1', s2', s3', s4'};

%% SECTION 10: COMBINED QC VISUALIZATION (2x4 PANEL)
% -------------------------------------------------------------------------
figure('Name', 'QC: Residuals and Local Similarity Comparison', ...
       'Position', [50 50 1500 900], 'Color', 'w');

tlo_qc = tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

methods_names = {titles{2}, titles{3}, titles{4}, titles{5}}; 
sub_labels_qc = {'(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)'};

% Define Zoom coordinates for detailed inspection
zCh = [rect_pos(1), rect_pos(1) + rect_pos(3)];
zT  = [rect_pos(2), rect_pos(2) + rect_pos(4)];

%% ROW 1: NOISE REMOVED (RESIDUALS)
for k = 1:4
    ax_res = nexttile(k);
    
    % Displaying the removed noise component
    imagesc(chan_range, time_window_focused, residuals{k}); 
    xlim(zCh); ylim(zT);
    
    colormap(ax_res, seis); 
    caxis([-0.5 0.5]); % Tightened to detect faint coherent events in noise
    
    set(gca, 'YDir', 'reverse', 'LineWidth', 1.5, 'FontWeight', 'bold', 'XTickLabel', []);
    title([methods_names{k}], 'FontSize', 14, 'FontWeight', 'bold');
    
    % Panel identification
    text(-0.1, 1.06, sub_labels_qc{k}, 'Units', 'normalized', ...
         'FontSize', 18, 'FontWeight', 'bold');
end

%% ROW 2: LOCAL SIMILARITY MAPS (LEAKAGE DETECTION)
for k = 1:4
    ax_sim = nexttile(k+4);
    
    % Displaying the leakage coefficient map
    imagesc(chan_range, time_window_focused, simi_list{k}'); 
    xlim(zCh); ylim(zT);
    
    colormap(ax_sim, jet); 
    caxis([0 1]); % Scale focused on detecting even minor leakage
    
    set(gca, 'YDir', 'reverse', 'LineWidth', 1.5, 'FontWeight', 'bold');
    title([methods_names{k}], 'FontSize', 14, 'FontWeight', 'bold');
    
    % Panel identification
    text(-0.1, 1.06, sub_labels_qc{k+4}, 'Units', 'normalized', ...
         'FontSize', 18, 'FontWeight', 'bold');
end

% Global Figure Annotations
xlabel(tlo_qc, 'Channel Index', 'FontSize', 14, 'FontWeight', 'bold');
ylabel(tlo_qc, 'Time (s)', 'FontSize', 14, 'FontWeight', 'bold');

% Colorbars for each metric type
% cb1 = colorbar(nexttile(4)); 
% cb1.Label.String = 'Residual Amplitude';
% cb1.Label.FontWeight = 'bold';

cb2 = colorbar(nexttile(8)); 
cb2.Label.String = 'Local Similarity';
cb2.Label.FontWeight = 'bold';

fprintf('-> QC Comparison Complete. Analyze (h) for HADDL superiority.\n');

%% SECTION 11: AVERAGE TRACE FIDELITY ANALYSIS (BENCHMARK)
% -------------------------------------------------------------------------
% This section evaluates the signal preservation by averaging across all 
% channels. It allows for a direct comparison of the waveform reconstruction 
% and P-wave arrival clarity across all methods.
% -------------------------------------------------------------------------

figure('Name', 'Figure: Average Trace Comparison (Method Benchmark)', ...
       'Position', [100 100 600 800], 'Color', 'w'); % Adjusted position/aspect ratio for vertical plots
hold on; 

% 1. Noisy Reference (The background noise level)
v_noisy = mean(dn_noisy, 2); 
t_axis = linspace(2.5, 5.1, length(v_noisy)); 
plot(v_noisy, t_axis, 'k', 'LineWidth', 1, 'DisplayName', 'Noisy (Reference)');

% 2. BP+SGK Benchmark
v_sgk = mean(sgk_mirodas, 2);
plot(v_sgk, t_axis, 'b', 'LineWidth', 1.5, 'DisplayName', 'BP+SGK');

% 3. MHA-RN Benchmark
v_mharn = mean(mharn_microDAS', 2);
plot(v_mharn, t_axis, 'g', 'LineWidth', 1.5, 'DisplayName', 'MHA-RN');

% 4. SSDL Benchmark
v_ssdl = mean(ssdl, 2);
plot(v_ssdl, t_axis, 'Color', [0.9 0.7 0], 'LineWidth', 1.5, 'DisplayName', 'SSDL');

% 5. HADDL (Proposed Method) - Highlighted in Red
v_haddl = mean(haddl_microDAS', 2);
plot(v_haddl, t_axis, 'r', 'LineWidth', 1.5, 'DisplayName', 'HADDL (Proposed)');

% --- ARCHITECTURAL ANNOTATIONS ---
% Horizontal line marking the P-wave arrival for phase check (Swapped from xline to yline)
yline(3.35, '--', 'Color', [0.3 0.3 0.3], 'LineWidth', 3, ...
      'Label', 'P-wave arrival', 'LabelOrientation', 'horizontal', ...
      'FontSize', 14, 'FontWeight', 'bold', 'HandleVisibility', 'off');

% Figure Formatting
xlabel('Normalized Amplitude', 'FontSize', 16, 'FontWeight', 'bold'); % Swapped label
ylabel('Time (s)', 'FontSize', 16, 'FontWeight', 'bold');             % Swapped label

% Set axes limits and reverse Y-direction to ensure time flows downwards
set(gca, 'LineWidth', 2, 'FontSize', 16, 'FontWeight', 'bold', ...
    'YDir', 'reverse', ...     % Crucial: inverted time depth axis
    'YLim', [2.5 5.1], ...     % Time is now on the Y-axis
    'XLim', [-0.5 0.5]);       % Amplitude is now on the X-axis

legend('Location', 'southoutside', 'FontSize', 14, 'FontWeight', 'bold'); % Placed at bottom for narrow layout
title('Average Trace Amplitude Comparison');
% grid on;
hold off;
fprintf('-> Re-oriented vertical average trace comparison plot generated.\n');

