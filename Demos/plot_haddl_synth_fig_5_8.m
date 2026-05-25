clc; clear; close all; 
% =========================================================================
% SCRIPT: plot_haddl_synth_fig_5_8.m
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
%   This script reproduces Figures 5, 6, 7, and 8 of the manuscript using 
%   SYNTHETIC DATA. It provides a multi-method benchmark comparing the 
%   proposed HADDL framework against BP+SGK, SSDL, and MHA-RN.
%
%   NOTE ON SSDL DATA:
%   The SSDL (Self-Supervised Deep Learning) benchmark results were generated 
%   using the official Python/Jupyter Notebook framework provided by Saad et 
%   al. [4]. The denoised outputs from their pipeline were exported and saved 
%   as 'd_ssdlsynth.mat' for direct plotting and analysis within this 
%   MATLAB environment.
%
%   PREREQUISITES:
%   Prior to executing this script, run the following workflows to generate 
%   the required benchmark data data matrices:
%   1. test_haddl_synth.m  (Proposed HADDL framework)
%   2. test_sgk_synth.m    (BP+SGK Baseline)
%   3. test_mharn_synth.m  (MHA-RN Baseline)
%%
%   Primary References:   
%   [1] Oboué, Y. A. S. I., Chen, Y., & Chen, Y. (2026). A Hybrid 
%       Attention-Driven Deep Learning Framework for Denoising DAS Data. 
%       Geophysics (Under Review).
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
%% SECTION 1: PATH MANAGEMENT & ENVIRONMENT SETUP 
% -------------------------------------------------------------------------
fprintf('--- Initializing Environment ---\n');
addpath(genpath('../subroutines')); 
addpath(genpath('../Synth_data'));   
addpath(genpath('./Output_Synth_SGK')); 
addpath(genpath('./Output_Synth_SSDL')); 
addpath(genpath('./Output_Synth_MHA-RN'));
addpath(genpath('./Output_Synth_HADDL_Proposed'));
%%
rng(42, 'twister'); 

%% SECTION 2: UNIVERSAL DATA LOADING
% -------------------------------------------------------------------------
fprintf('--- Loading Multi-Method Results ---\n');

% 1. Load Synthetic Ground Truth and Noisy Input
load micro_sf_3001_334_3.mat 
d = haddl_scale(data(:,:,1), 2); % Clean Reference
[n1, n2, n3] = size(d);

load dnoiseSynthDAS.mat 
dn = dnoiseSynthDAS;          % Noisy Input

% 2. Load SGK Baseline Results
if exist('result_sgk.mat', 'file')
    load('result_sgk.mat'); 
else
    warning('SGK result file not found. Run test_sgk_synth.m first.');
end

% 3. Load SSDL Result (Pre-processed via Python)
if exist('d_ssdlsynth.mat', 'file')
    load('d_ssdlsynth.mat');
    fprintf('-> SSDL data loaded successfully (External Python Output).\n');
else
    warning('SSDL result file (d_ssdlsynth.mat) not found.');
end

% 4. Load MHA-RN Result
if exist('d_mharn.mat', 'file')
    load('d_mharn.mat'); 
else
    warning('MHA-RN result file not found. Run test_mharn_synth.m first.');
end

% 5. Load HADDL Proposed Result
if exist('d_synth_haddl.mat', 'file')
    load('d_synth_haddl.mat'); 
else
    warning('HADDL result file not found. Run test_haddl_synth.m first.');
end

fprintf('Success: All available benchmark data loaded.\n');
%%
% Grid Definition for consistent plotting
fprintf('Success: Environment ready and data loaded.\n');

% =========================================================================
%% SECTION 3: PERFORMANCE METRICS CALCULATION
% -------------------------------------------------------------------------
% Calculate the Signal-to-Noise Ratio (SNR) for the noisy input
snr_noisy = haddl_snr(d, dn, 2);

% Optional: Calculate SNR for all denoised results to display in titles
snr_sgk    = haddl_snr(d, d_denoisedsgk, 2);
snr_ssdl   = haddl_snr(d, d_ssdl, 2);
snr_mharn  = haddl_snr(d, d_mharn, 2);
snr_haddl  = haddl_snr(d, d_synth_haddl, 2);

%% SECTION 4: FIGURE 5 GENERATION - MULTI-METHOD COMPARISON
% -------------------------------------------------------------------------
% Setup grid dimensions and sampling interval
[n1, n2] = size(d);
dt = 0.0005; % Sampling rate in seconds
t = [0:n1-1] * dt; 
x = [1:n2];

% Initialize figure for comparative analysis
figure('Name', 'Figure 5: Comparative Denoising Performance', ...
       'units', 'normalized', 'Position', [0.1 0.1 0.7 0.85], 'color', 'w');

% Create a 2x3 tiled layout for compact visualization
tlo = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% Annotation parameters for visual analysis
line_x   = 92;                % Spatial position for the vertical profile line
rect_pos = [150, 0.5, 150, 0.5]; % [x_start, y_start, width, height] for zoom area

% Compile data arrays and metadata for the automated plotting loop
data_list = {d, dn, d_denoisedsgk, d_ssdl, d_mharn, d_synth_haddl};
titles    = {'Clean Reference', ...
             sprintf('Noisy (%.2f dB)', snr_noisy), ...
             sprintf('BP+SGK (%.2f dB)', snr_sgk), ...
             sprintf('SSDL (%.2f dB)', snr_ssdl), ...
             sprintf('MHA-RN (%.2f dB)', snr_mharn), ...
             sprintf('HADDL (Proposed, %.2f dB)', snr_haddl)};
labels    = {'(a)', '(b)', '(c)', '(d)', '(e)', '(f)'};

% Automated plotting loop
for k = 1:6
    nexttile(k);
    
    % Display the seismic section using custom imaging function
    haddl_imagesc(data_list{k}, 100, 2, x, t);
    title(titles{k}, 'FontSize', 14, 'FontWeight', 'bold');
    
    % Axis styling and color mapping
    ylabel('Time (s)');
    xlabel('Channel');
    colormap(seis);
    caxis([-0.5 0.5]);
    set(gca, 'LineWidth', 1.5, 'FontSize', 12);
    
    % Overlay reference indicators
    hold on;
    
    % Add red dashed line for vertical profile comparison
    xline(line_x, '--r', 'LineWidth', 2);
    
    % Add subplot index label (e.g., (a), (b)) at top-left
    text(-0.15, 1.05, labels{k}, 'Units', 'normalized', ...
         'FontSize', 20, 'FontWeight', 'bold', 'Color', 'k');
    
    % Draw red rectangle to highlight the specific zoom-in analysis zone
    rectangle('Position', rect_pos, 'EdgeColor', 'r', 'LineWidth', 2);
    
    hold off;
end
%% SECTION 5: FIGURE 6 GENERATION - DETAILED ZOOM-IN ANALYSIS
% -------------------------------------------------------------------------
% This section extracts and visualizes specific regions defined by [rect_pos]
% to highlight the preservation of weak seismic signals and noise removal.
% -------------------------------------------------------------------------

% Initialize figure for zoomed comparisons
figure('Name', 'Figure 6: Zoomed Performance Comparison', ...
       'units', 'normalized', 'Position', [0.1 0.1 0.7 0.85], 'color', 'w');

tlo_zoom = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% --- COORDINATES FOR ANNOTATION ARROWS ---
% Precise arrow tip locations (based on feature analysis)
arrow_tips = [
    188, 0.575;  % Feature 1
    282, 0.550;  % Feature 2
    172, 0.815;  % Feature 3
    292, 0.855;  % Feature 4
    162, 0.955;  % Feature 5
    288, 0.975]; % Feature 6

% Direction: -1 for left-pointing arrows (<)
directions = -ones(size(arrow_tips, 1), 1); 

% Define window limits based on the previous rectangle position
x_min = rect_pos(1); x_max = rect_pos(1) + rect_pos(3);
y_min = rect_pos(2); y_max = rect_pos(2) + rect_pos(4);

for k = 1:6
    nexttile(k);
    
    % 1. Plot the seismic data
    haddl_imagesc(data_list{k}, 100, 2, x, t);
    
    % 2. Apply zoom limits
    xlim([x_min, x_max]);
    ylim([y_min, y_max]);
    hold on; 
    
    % --- PLOT CUSTOM ANNOTATION ARROWS (STEM + HEAD) ---
    for i = 1:size(arrow_tips, 1)
        xtip = arrow_tips(i,1);
        ytip = arrow_tips(i,2);
        dir  = directions(i);
        
        % Define arrow stem (fixed length of 10 channels)
        % Stem is slightly tilted for better visibility (0.04s offset)
        x_tail = xtip - (dir * 10); 
        y_tail = ytip + 0.04; 
        
        % A. Draw the stem (Solid red line)
        plot([x_tail, xtip], [y_tail, ytip], 'r', 'LineWidth', 2);
        
        % B. Draw the head (Filled triangle marker)
        plot(xtip, ytip, 'r>', 'MarkerFaceColor', 'r', 'MarkerSize', 12);
    end
    
    % 3. Vertical profile reference line (if within zoom window)
    if exist('line_x', 'var') && line_x >= x_min && line_x <= x_max
        xline(line_x, '--r', 'LineWidth', 1.5);
    end
    
    % 4. Professional formatting and labels
    text(-0.15, 1.05, labels{k}, 'Units', 'normalized', ...
         'FontSize', 18, 'FontWeight', 'bold');
    title(titles{k}, 'FontSize', 13, 'FontWeight', 'bold');
    
    ylabel('Time (s)'); xlabel('Channel');
    colormap(seis); 
    caxis([-0.5 0.5]);
    
    % Ensure axes follow seismic conventions (Time increasing downwards)
    set(gca, 'LineWidth', 1.5, 'FontSize', 11, 'YDir', 'reverse');
    
    hold off;
end

% Final Title
% title(tlo_zoom, 'Zoom-In View: Detailed Reflector Preservation Analysis', ...
%       'FontSize', 18, 'FontWeight', 'bold');

%% SECTION 6: COMPREHENSIVE PERFORMANCE EVALUATION (RESIDUALS & LEAKAGE)
% -------------------------------------------------------------------------
% This section evaluates "Signal Leakage" by analyzing the residuals 
% (Noise removed) and their local similarity with the denoised results.
% -------------------------------------------------------------------------

% 1. PRE-COMPUTATION OF LOCAL SIMILARITY MAPS
% -------------------------------------------------------------------------
% Similarity parameters: rect defines the local window size
rect = [20 20 1]; nsim_iter = 20; eps_val = 0; verb = 0;

fprintf('Calculating Local Similarity Maps (Checking for Signal Leakage)... ');

% Calculate similarity between the removed noise (residual) and the denoised signal
% High similarity values (close to 1) indicate undesired signal damage.
simi1 = haddl_localsimi(dn - d_denoisedsgk, d_denoisedsgk, rect, nsim_iter, eps_val, verb);
simi2 = haddl_localsimi(dn - d_ssdl, d_ssdl, rect, nsim_iter, eps_val, verb); % Python-sourced data
simi3 = haddl_localsimi(dn - d_mharn, d_mharn, rect, nsim_iter, eps_val, verb);
simi4 = haddl_localsimi(dn - d_synth_haddl, d_synth_haddl, rect, nsim_iter, eps_val, verb);

fprintf('Done.\n');
%
figure('Name', 'Figure 7: Residual and Signal Leakage Analysis', ...
       'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.6], 'Color', 'w');

% Create a 2x4 tiled layout for residuals (Top) and similarity (Bottom)
tlo = tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

% Labels for publication compliance (a-h)
labels = {'(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)'};

% --- ROW 1: RESIDUAL SECTIONS (Noise Removed) ---
% Residual = Noisy Input - Denoised Output
nexttile(1); haddl_imagesc(dn - d_denoisedsgk, 100, 2, x, t); title('BP+SGK');
nexttile(2); haddl_imagesc(dn - d_ssdl, 100, 2, x, t); title('SSDL');
nexttile(3); haddl_imagesc(dn - d_mharn, 100, 2, x, t); title('MHA-RN');
nexttile(4); haddl_imagesc(dn - d_synth_haddl, 100, 2, x, t); title('Proposed HADDL');

% --- ROW 2: LOCAL SIMILARITY MAPS (Visualizing Signal Leakage) ---
% High values in similarity maps indicate unwanted removal of coherent signals
nexttile(5); haddl_imagesc(simi1, 100, 2, x, t); 
nexttile(6); haddl_imagesc(simi2, 100, 2, x, t); 
nexttile(7); haddl_imagesc(simi3, 100, 2, x, t); 
nexttile(8); haddl_imagesc(simi4, 100, 2, x, t); 

% --- FINAL FORMATTING, STYLING, AND ANNOTATION ---
% -------------------------------------------------------------------------

for k = 1:8
    ax = nexttile(k);
    hold on;
    
    % Add Labels (a, b, c...) at the top-left corner of each panel
    text(-0.12, 1.1, labels{k}, 'Units', 'normalized', ...
         'FontSize', 18, 'FontWeight', 'bold', 'Color', 'k');
    
    if k <= 4
        % TOP ROW: Residual Analysis Styling
        colormap(ax, seis); % Use seismic colormap
        set(ax, 'CLim', [-0.5 0.5], 'LineWidth', 1.5, 'FontSize', 14);
        
        if k == 1
            ylabel('Time (s)', 'FontWeight', 'bold'); 
        end
        
        % Add a standard colorbar for residuals on the last panel of the row
        if k == 4
            cb_res = colorbar(ax);
            ylabel(cb_res, 'Residual Amplitude', 'FontSize', 12, 'FontWeight', 'bold');
        end
    else
        % BOTTOM ROW: Local Similarity Analysis Styling
        colormap(ax, jet); % Jet scale is standard for similarity index [0 to 1]
        set(ax, 'CLim', [0 1], 'LineWidth', 1.5, 'FontSize', 14);
        xlabel('Channel', 'FontWeight', 'bold');
        
        if k == 5
            ylabel('Time (s)', 'FontWeight', 'bold'); 
        end
        
        % ADDING THE "LOCAL SIMILARITY" LABEL TO THE COLORBAR
        if k == 8
            cb_sim = colorbar(ax);
            % Title added to the colorbar for clarity
            ylabel(cb_sim, 'Local Similarity', 'FontSize', 13, 'FontWeight', 'bold');
        end
    end
    
    set(ax, 'FontWeight', 'bold');
    grid off;
end

fprintf('-> Figure 7 rendering complete. Ready for export.\n');
% Title for each row can be added via the Layout if needed
% title(tlo, 'Comparative Analysis of Residuals and Signal Preservation', 'FontSize', 16);
% title(tlo, 'Comparative Signal Leakage and Residual Analysis', ...
      % 'FontSize', 22, 'FontWeight', 'bold');


%% SECTION 7: $P$-WAVE ARRIVAL FIDELITY & WAVEFORM ANALYSIS (CHANNEL 90)
% -------------------------------------------------------------------------
% This section extracts a single trace to evaluate the preservation 
% of the $P$-wave onset and overall waveform phase/amplitude accuracy.
% -------------------------------------------------------------------------

ch = 90;           % Target channel for trace analysis
t_pwave = 0.212;   % Theoretical $P$-wave arrival time
pwave_roi = [0.20, 0.25]; % Focus window for zoom-in analysis

figure('Name', 'Figure 8: Waveform Fidelity Analysis', 'Color', 'w', ...
       'Units', 'normalized', 'Position', [0.1 0.1 0.4 0.8]);

hold on;

% Plotting chronological sequence for visual clarity
plot(dn(:, ch), t, 'k', 'LineWidth', 1, 'DisplayName', 'Noisy'); 
plot(d_denoisedsgk(:, ch), t, 'b', 'LineWidth', 2.5, 'DisplayName', 'BP+SGK');

if exist('d_mharn', 'var')
    plot(d_mharn(:, ch), t, 'y', 'LineWidth', 2, 'DisplayName', 'MHA-RN'); 
end

plot(d_ssdl(:, ch), t, 'cyan', 'LineWidth', 2.5, 'DisplayName', 'SSDL');
plot(d(:, ch), t, 'g', 'LineWidth', 2.5, 'DisplayName', 'Clean Reference');
plot(d_synth_haddl(:, ch), t, 'r', 'LineWidth', 2.5, 'DisplayName', 'Proposed HADDL');

% Add the $P$-wave Arrival Marker
yline(t_pwave, '-.b', 'P-wave arrival', 'LineWidth', 2.5, ...
      'LabelVerticalAlignment', 'bottom', 'FontSize', 14, 'Color', [0 0.447 0.741]);

% Professional styling
set(gca, 'YDir', 'reverse'); % Standard seismic time axis
xlabel('Normalized Amplitude'); 
ylabel('Time (s)');
title(['Single-Trace Amplitude Comparison - Channel ', num2str(ch)], 'FontSize', 18);
% title(['Single-Trace Amplitude Comparison - Channel ', num2str(ch)]);

legend('Location', 'southwest', 'FontSize', 12);

% Set limits to focus on the seismic event
xlim([-3 3]); 
set(gca, 'LineWidth', 2.5, 'FontSize', 16, 'FontWeight', 'bold');

% grid on;
hold off;

% --- SUB-SECTION 6.2: ZOOMED P-WAVE ARRIVAL ---
% figure('Name', 'Zoomed P-Wave Arrival', 'Color', 'w', ...
%        'Units', 'normalized', 'Position', [0.55 0.3 0.35 0.5]);
% 
% hold on;
% plot(dn(:, ch), t, 'k', 'LineWidth', 0.5); 
% plot(d_denoisedsgk(:, ch), t, 'b', 'LineWidth', 1.2);
% if exist('d_mharn', 'var'), plot(d_mharn(:, ch), t, 'y', 'LineWidth', 1.2); end
% plot(d_ssdl(:, ch), t, 'cyan', 'LineWidth', 3);
% plot(d(:, ch), t, 'g', 'LineWidth', 3);
% plot(d_synth_haddl(:, ch), t, 'r', 'LineWidth', 1.5);
% 
% % Formatting the Zoom
% set(gca, 'YDir', 'reverse'); 
% % grid on; 
% box on;
% xlim(amp_roi); 
% ylim(pwave_roi);
% 
% xlabel('Amplitude'); ylabel('Time (s)');
% title('Zoomed P-wave Arrival Fidelity');
% % text(-0.15, 1.05, '(h)', 'Units', 'normalized', 'FontSize', 22, 'FontWeight', 'bold');
% set(gca, 'LineWidth', 2, 'FontSize', 16, 'FontWeight', 'bold');
% hold off;