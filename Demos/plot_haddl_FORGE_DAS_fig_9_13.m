% =========================================================================
% SCRIPT: plot_haddl_field_FORGE_DAS_fig_9_13.m
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
%   This script reproduces Figures 9, 10, 11, and 12 of the manuscript using 
%   REAL FIELD DATA (FORGE DAS). It provides a comprehensive comparison 
%   between the proposed HADDL framework, BP+SGK, SSDL, and MHA-RN.
%
%   NOTE ON SSDL DATA:
%   The SSDL (Self-Supervised Deep Learning) benchmark results were generated 
%   using the official Python/Jupyter Notebook framework provided by Saad et 
%   al. [4]. The denoised outputs from their pipeline were exported and saved 
%   as 'd_ssdl10.mat' for direct plotting and analysis within this 
%   MATLAB environment.

%   PREREQUISITES:
%   Before running this script, ensure the following field processing 
%   scripts have been executed to generate the required baseline results:
%   1. test_haddl_field_FORGE_DAS.m (Proposed HADDL)
%   2. test_sgk_field_FORGE_DAS.m   (BP+SGK Baseline)
%   3. test_mharn_field_FORGE_DAS.m (MHA-RN Baseline)
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
clc; clear; close all; 
%% SECTION 1: PATH MANAGEMENT & ENVIRONMENT SETUP 
% -------------------------------------------------------------------------
% Automatically add all subfolders to the MATLAB path
fprintf('--- Initializing Environment for Field Data Analysis ---\n');

addpath(genpath('../subroutines')); % Helper functions
addpath(genpath('../Field_data'));   
addpath(genpath('./Output_Field_FORGE_DAS_SGK')); 
addpath(genpath('./Output_Field_FORGE_DAS_SSDL'));
addpath(genpath('./Output_Field_FORGE_DAS_MHARN'));
addpath(genpath('./Output_Field_FORGE_DAS_HADDL'));

% Ensures repeatable results for any randomization used in plotting
rng(42, 'twister'); 

%% SECTION 2: FIELD DATA LOADING
% -------------------------------------------------------------------------
fprintf('--- Loading Field Benchmark Results (FORGE DAS) ---\n');

% 1. Load Noisy Field Data
if exist('forgeDAS_data.mat', 'file')
    load('forgeDAS_data.mat'); 
    dn = d1; % Input noisy data assigned to 'dn'
    fprintf('-> Success: Original noisy field data loaded.\n');
else
    warning('Original field data file (forgeDAS_data.mat) not found.');
end

% 2. Load BP+SGK Field Results
if exist('sgkdas.mat', 'file')
    % If the variable inside is named 'sgk_mirodas' or similar, assign it explicitly after this line
    load('sgkdas.mat'); 
    % Example safety assignment: d_sgk = [variable_inside_file];
    fprintf('-> Success: BP+SGK field results loaded.\n');
else
    warning('SGK field results (sgkdas10.mat) not found. Run test_sgk_field_FORGE_DAS.m.');
end

% 3. Load SSDL Field Results (Processed via Python)
if exist('d_ssdl10.mat', 'file')
    load('d_ssdl10.mat');
    fprintf('-> Success: SSDL field data loaded.\n');
else
    warning('SSDL field result file (d_ssdl10.mat) not found.');
end

% 4. Load MHA-RN Field Results
if exist('d_mharn_f1.mat', 'file')
    load('d_mharn_f1.mat'); 
    fprintf('-> Success: MHA-RN field results loaded.\n');
else
    warning('MHA-RN field results (d_mharn_f1.mat) not found. Run test_mharn_field_FORGE_DAS.m.');
end

% 5. Load HADDL Field Results
if exist('d_haddl_f1.mat', 'file')
    load('d_haddl_f1.mat'); 
    fprintf('-> Success: Proposed HADDL field results loaded.\n');
else
    warning('HADDL field results (d_haddl_f1.mat) not found. Run test_haddl_field_FORGE_DAS.m.');
end

fprintf('--- Field environment ready and data loaded. ---\n');
% Ensures repeatable results for any randomization used in plotting
rng(42, 'twister'); 
% ========================================================================= 
%% SECTION 3: DATA INITIALIZATION & GRID SETUP
% -------------------------------------------------------------------------
% Define spatial and temporal axes based on field acquisition parameters
[n1, n2, n3] = size(dn);
dt = 0.004;         % Time sampling interval (s) - FORGE DAS standard
t = (0:n1-1) * dt;  % Temporal vector
x = 1:n2;           % Spatial vector (Channel index)

%% SECTION 4: FIGURE 9 GENERATION - FIELD DATA BENCHMARK (2x3 Layout)
% -------------------------------------------------------------------------
% This figure provides a side-by-side comparison of different denoising 
% workflows applied to the FORGE field dataset.
% -------------------------------------------------------------------------

figure('Name', 'Figure 9: Field Data Denoising Comparison (FORGE DAS)', ...
       'units', 'normalized', 'Position', [0.1 0.1 0.7 0.85], 'color', 'w');

% Create a 2x3 tiled layout (6 tiles total, 5 datasets + 1 empty/summary)
tlo = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% --- ANNOTATION PARAMETERS ---
line_x    = 860;           % Position for vertical profile analysis
rect_pos  = [200, 1, 200, 1]; % Primary zoom area [x, y, w, h]
rect_pos2 = [600, 4, 250, 1]; % Secondary zoom area (near-field/deeper events)

% --- DATA & METADATA ORCHESTRATION ---
% Note: Using field-specific variables (dn, sgkdas10, d_ssdl, etc.)
data_list = {dn, sgkdas, d_ssdl, d_mharn_f, d_haddl_f};
titles    = {'Noisy', 'BP+SGK', 'SSDL', 'MHA-RN', 'HADDL'};
labels    = {'(a)', '(b)', '(c)', '(d)', '(e)', '(f)'};

% Automated plotting loop for the 5 available datasets
for k = 1:length(data_list)
    nexttile(k);
    
    % 1. Seismic Imaging
    % Using 100% clip and specific gain for field data visibility
    haddl_imagesc(data_list{k}, 100, 2, x, t);
    title(titles{k}, 'FontSize', 14, 'FontWeight', 'bold');
    
    % 2. Axis Configuration
    ylabel('Time (s)');
    xlabel('Channel');
    colormap(seis);
    caxis([-100 100]); % Amplitude scaling for field data
    set(gca, 'LineWidth', 1.5, 'FontSize', 12);
    
    % 3. Overlays & Annotations
    hold on;
    
    % Red dashed line for profile comparison
    xline(line_x, '--r', 'LineWidth', 2);
    
    % Subplot indexing (Top-left position)
    text(-0.15, 1.05, labels{k}, 'Units', 'normalized', ...
         'FontSize', 20, 'FontWeight', 'bold', 'Color', 'k');
    
    % Primary and Secondary ROI (Region of Interest) boxes
    rectangle('Position', rect_pos,  'EdgeColor', 'r', 'LineWidth', 2);
    rectangle('Position', rect_pos2, 'EdgeColor', 'r', 'LineWidth', 2);
    
    hold off;
end

% Optional: Set common title for the entire tiled layout
% title(tlo, 'Comparative Denoising Performance on FORGE Field Data', ...
%       'FontSize', 18, 'FontWeight', 'bold');
%% SECTION 5: DETAILED ZOOM-IN ANALYSIS - REGION OF INTEREST (ROI) 1
% -------------------------------------------------------------------------
% This section extracts the first zoom area defined by [rect_pos] to 
% evaluate noise attenuation and signal preservation on field data.
% -------------------------------------------------------------------------

figure('Name', 'Figure 10: Field Data Zoom-In (ROI 1)', ...
       'units', 'normalized', 'Position', [0.1 0.1 0.7 0.85], 'color', 'w');

% Create 2x3 layout to accommodate the 5 comparison methods
tlo_zoom = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% --- ARROW ANNOTATION COORDINATES (Channel, Time) ---
% Based on visual identification of specific seismic features in ROI 1
% Arrow 1: Upper right (Channel 290, Time 1.05s)
% Arrow 2: Middle left (Channel 233, Time 1.22s)
arrow_coords = [
    290, 1.05; 
    233, 1.22
];

% Extraction of zoom limits from the previously defined rect_pos
x_min = rect_pos(1);
x_max = rect_pos(1) + rect_pos(3);
y_min = rect_pos(2);
y_max = rect_pos(2) + rect_pos(4);

% Loop through the 5 datasets for consistent plotting
for k = 1:5
    nexttile(k);
    
    % 1. Display seismic data using custom imaging function
    haddl_imagesc(data_list{k}, 100, 2, x, t);
    
    % 2. Apply spatial and temporal zoom constraints
    xlim([x_min, x_max]);
    ylim([y_min, y_max]);
    
    % 3. Overlay Red Annotation Arrows for feature tracking
    hold on;
    for i = 1:size(arrow_coords, 1)
        % Using LaTeX interpretation for high-quality arrow markers (\leftarrow)
        text(arrow_coords(i,1), arrow_coords(i,2), ' \leftarrow', ...
            'Color', 'red', 'FontSize', 30, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
    end
    
    % 4. Subplot Labeling (e.g., (a), (b)...)
    text(-0.15, 1.05, labels{k}, 'Units', 'normalized', ...
         'FontSize', 20, 'FontWeight', 'bold', 'Color', 'k');
    
    % 5. Method Title
    title(titles{k}, 'FontSize', 14, 'FontWeight', 'bold');
    
    % 6. Axis and Styling Configuration
    ylabel('Time (s)');
    xlabel('Channel');
    colormap(seis);
    caxis([-100 100]); % Maintain consistent amplitude scaling for field data
    set(gca, 'LineWidth', 1.5, 'FontSize', 12);
    
    % 7. Red Profile Reference Line (if within the zoom window)
    if line_x >= x_min && line_x <= x_max
        xline(line_x, '--r', 'LineWidth', 2);
    end
    
    hold off;
end
%%
%% SECTION 6: DETAILED ZOOM-IN ANALYSIS - REGION OF INTEREST (ROI) 2
% -------------------------------------------------------------------------
% This section focuses on the second zoom area [rect_pos2], targeting 
% complex noise patterns and deeper seismic arrivals in the FORGE dataset.
% -------------------------------------------------------------------------

figure('Name', 'Figure 11: Field Data Zoom-In (ROI 2)', ...
       'units', 'normalized', 'Position', [0.1 0.1 0.7 0.85], 'color', 'w');

% Create 2x3 layout to maintain consistency with the main benchmark
tlo_zoom = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% Indices labels list
labels = {'(a)', '(b)', '(c)', '(d)', '(e)', '(f)'};

% --- ARROW ANNOTATION COORDINATES (Channel, Time) ---
% Based on visual analysis of noise artifacts and signal intersections:
% Arrow 1: Upper vertical noise
% Arrow 2: Central intersection
% Arrow 3: Right-side random noise
% Arrow 4: Bottom noisy trace
arrow_coords = [
    675, 4.12;  
    675, 4.41;
    760, 4.55;
    600, 4.92];

% Extraction of zoom limits from the second ROI definition [rect_pos2]
x_min = rect_pos2(1);
x_max = rect_pos2(1) + rect_pos2(3);
y_min = rect_pos2(2);
y_max = rect_pos2(2) + rect_pos2(4);

% Loop through available datasets (Noisy + 4 Denoising methods)
for k = 1:5
    nexttile(k);
    
    % 1. Display seismic data
    haddl_imagesc(data_list{k}, 100, 2, x, t);
    
    % 2. Apply specific zoom limits for ROI 2
    xlim([x_min, x_max]);
    ylim([y_min, y_max]);
    
    % 3. Overlay Red Annotation Arrows
    hold on;
    for i = 1:size(arrow_coords, 1)
        % Using LaTeX interpretation for professional arrow rendering (\leftarrow)
        text(arrow_coords(i,1), arrow_coords(i,2), ' \leftarrow', ...
            'Color', 'red', 'FontSize', 30, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
    end
    
    % 4. Subplot Labeling (e.g., (a), (b)...)
    text(-0.15, 1.05, labels{k}, 'Units', 'normalized', ...
         'FontSize', 20, 'FontWeight', 'bold', 'Color', 'k');
    
    % 5. Method Title
    title(titles{k}, 'FontSize', 16, 'FontWeight', 'bold');
    
    % 6. Axis and Styling Configuration
    ylabel('Time (s)');
    xlabel('Channel');
    colormap(seis);
    caxis([-100 100]); % Consistent amplitude scaling for field data comparison
    set(gca, 'LineWidth', 2, 'FontSize', 12);
    
    % 7. Red Profile Reference Line (emphasized width for ROI 2)
    if line_x >= x_min && line_x <= x_max
        xline(line_x, '--r', 'LineWidth', 5);
    end
    
    hold off;
end

% Final Title
% title(tlo_zoom, 'Zoom-In View: Detailed Reflector Preservation Analysis', ...
%       'FontSize', 18, 'FontWeight', 'bold');

%%
%% SECTION 7: COMPREHENSIVE PERFORMANCE EVALUATION
% -------------------------------------------------------------------------
% Row 1: Residual Sections | Row 2: Local Similarity Maps
% -------------------------------------------------------------------------
% 1. PRE-COMPUTATION
% -------------------------------------------------------------------------
% Similarity parameters
rect = [20 20 1]; nsim_iter = 20; eps_val = 0; verb = 0;

fprintf('Calculating Similarity Maps... ');
simi1 = haddl_localsimi(dn - sgkdas, sgkdas, rect, nsim_iter, eps_val, verb);
simi2 = haddl_localsimi(dn - d_ssdl, d_ssdl, rect, nsim_iter, eps_val, verb);
simi3 = haddl_localsimi(dn - d_mharn_f, d_mharn_f, rect, nsim_iter, eps_val, verb);
simi4 = haddl_localsimi(dn - d_haddl_f, d_haddl_f, rect, nsim_iter, eps_val, verb);
fprintf('Done.\n');

% -------------------------------------------------------------------------
% 2. FIGURE & PLOTTING
% -------------------------------------------------------------------------
figure('Name', 'Residual and Signal Leakage Analysis', ...
       'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.6], 'Color', 'w');

% Create a 2x4 layout matrix
tlo = tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

% Standardized labels for all 8 subplot windows
labels = {'(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)'};

%% --- ROW 1: RESIDUALS (NOISE REMOVED) ---
nexttile(1); haddl_imagesc(dn - sgkdas, 100, 2, x, t); title('BP+SGK');
nexttile(2); haddl_imagesc(dn - d_ssdl, 100, 2, x, t); title('SSDL');
nexttile(3); haddl_imagesc(dn - d_mharn_f, 100, 2, x, t); title('MHA-RN');
nexttile(4); haddl_imagesc(dn - d_haddl_f, 100, 2, x, t); title('Proposed HADDL');

%% --- ROW 2: LOCAL SIMILARITY MAPS ---
nexttile(5); haddl_imagesc(simi1, 100, 2, x, t); 
nexttile(6); haddl_imagesc(simi2, 100, 2, x, t); 
nexttile(7); haddl_imagesc(simi3, 100, 2, x, t); 
nexttile(8); haddl_imagesc(simi4, 100, 2, x, t); 


%% --- FINAL FORMATTING AND COLORBAR INTEGRATION ---
% -------------------------------------------------------------------------

% 1. Format Row 1: Residual Sections (Tiles 1-4)
for k = 1:4
    ax = nexttile(k);
    colormap(ax, seis);
    set(ax, 'CLim', [-100 100], 'LineWidth', 1.5, 'FontSize', 14, 'FontWeight', 'bold');
    
    % Label placement slightly elevated above the panel border
    text(-0.15, 1.08, labels{k}, 'Units', 'normalized', 'FontSize', 18, 'FontWeight', 'bold', 'Color', 'k');
    
    if k == 1
        ylabel('Time (s)', 'FontWeight', 'bold'); 
    end
    
    % Add master residual colorbar to the last plot of Row 1
    if k == 4
        cb1 = colorbar(ax);
        ylabel(cb1, 'Residual Amplitude', 'FontSize', 12, 'FontWeight', 'bold');
    end
end

% 2. Format Row 2: Local Similarity Maps (Tiles 5-8)
for k = 5:8
    ax_sim = nexttile(k);
    colormap(ax_sim, jet);
    set(ax_sim, 'CLim', [0 1], 'LineWidth', 1.5, 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Channel', 'FontWeight', 'bold');
    
    % Label placement for the bottom row
    text(-0.15, 1.08, labels{k}, 'Units', 'normalized', 'FontSize', 18, 'FontWeight', 'bold', 'Color', 'k');
    
    if k == 5
        ylabel('Time (s)', 'FontWeight', 'bold'); 
    end
    
    % Add master local similarity colorbar to the last plot of Row 2
    if k == 8
        cb2 = colorbar(ax_sim);
        ylabel(cb2, 'Local Similarity', 'FontSize', 12, 'FontWeight', 'bold');
    end
end

% Global Super-Title layout configurations
% title(tlo, 'Comparative Signal Leakage and Residual Analysis', ...
%       'FontSize', 20, 'FontWeight', 'bold');

fprintf('-> Field residual analysis figure formatted with dedicated colorbars.\n');
%% SECTION 8: P-WAVE ARRIVAL FIDELITY & ZOOMED ANALYSIS (CHANNEL 860)
% -------------------------------------------------------------------------
% This section adds the P-wave onset marker and extracts the ROI 
% (Region of Interest) defined by the red frame box.
% -------------------------------------------------------------------------

ch = 850; 
t_pwave = 4.0; % 
pwave_roi = [0.20, 0.25]; % Time window for the zoom (y-axis)
% amp_roi = [-2, 2.5];      % Amplitude window for the zoom (x-axis)

% --- SUB-SECTION 8.1: FULL TRACE WITH ARRIVAL MARKER ---
figure('Name', 'Full Trace with P-Wave Marker', 'Color', 'w', ...
       'Units', 'normalized', 'Position', [0.1 0.1 0.4 0.8]);

hold on;
% Plot Noisy, Baselines, Clean, and HADDL
plot(dn(:, ch), t, 'k', 'LineWidth', 0.5, 'DisplayName', 'Noisy'); 
plot(sgkdas(:, ch), t, 'b', 'LineWidth', 1.5, 'DisplayName', 'BP+SGK');
if exist('d_mharn_f', 'var'), plot(d_mharn_f(:, ch), t, 'y', 'DisplayName', 'MHA-RN'); end
plot(d_ssdl(:, ch), t, 'cyan', 'LineWidth', 1.5, 'DisplayName', 'SSDL');
plot(d_haddl_f(:, ch), t, 'r', 'LineWidth', 1.5, 'DisplayName', 'HADDL');

% Add the P-wave Arrival Line (Blue Dash-Dot)
yline(t_pwave, '-.b', 'P-wave arrival', 'LineWidth', 3.5, ...
      'LabelVerticalAlignment', 'bottom', 'FontSize', 16, 'Color', [0 0.447 0.741]);

% Add the Red Frame Box (ROI) for the zoom
% rectangle('Position', [amp_roi(1), pwave_roi(1), diff(amp_roi), diff(pwave_roi)], ...
%           'EdgeColor', 'r', 'LineStyle', '--', 'LineWidth', 3);

set(gca, 'YDir', 'reverse'); 
% grid on;
xlabel('Amplitude'); ylabel('Time (s)');
title(['Single-Trace Amplitude Comparison - Channel ', num2str(ch)]);
legend('Location', 'southwest');
% xlim([-3 3]); 
% text(-0.15, 1.05, '(g)', 'Units', 'normalized', 'FontSize', 22, 'FontWeight', 'bold');
   set(gca, 'LineWidth', 3, 'FontSize', 16, 'FontWeight', 'bold');
hold off;

% --- SUB-SECTION 8.2: ZOOMED P-WAVE ARRIVAL ---
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