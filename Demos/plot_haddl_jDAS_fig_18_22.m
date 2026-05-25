clc; clear; close all;
% =========================================================================
% SCRIPT: plot_haddl_jDAS_fig_18_22.m
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
%   This script reproduces Figures 13 to 16 for the jDAS dataset. It 
%   compares the proposed HADDL framework against BP+SGK, SSDL, and MHA-RN 
%   using high-fidelity physical scaling (Distance in km, Time in s).
%
%   NOTE ON SSDL DATA:
%   The SSDL (Self-Supervised Deep Learning) benchmark results were generated 
%   using the official Python/Jupyter Notebook framework provided by Saad et 
%   al. [4]. The denoised outputs from their pipeline were exported and saved 
%   as 'ssdljdas.mat' for direct plotting and analysis within this 
%   MATLAB environment.
%
%   PREREQUISITES:
%   Before running this script, ensure the following processing scripts 
%   have been executed to generate the required baseline results:
%   1. test_haddl_field_jDAS.m (Proposed HADDL)
%   2. test_sgk_field_jDAS.m   (BP+SGK Baseline)
%   3. test_mharn_field_jDAS.m (MHA-RN Baseline)
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
%% SECTION 1: PATH MANAGEMENT & ENVIRONMENT SETUP 
% -------------------------------------------------------------------------
fprintf('--- Initializing Environment for jDAS Reconstruction ---\n');
addpath(genpath('../subroutines')); % Helper functions
addpath(genpath('../Field_data'));  

addpath(genpath('./Output_Field_jDAS_SGK')); 
addpath(genpath('./Output_Field_jDAS_SSDL'));
addpath(genpath('./Output_Field_jDAS_MHARN'));
addpath(genpath('./Output_Field_jDAS_HADDL'));

% Ensures repeatable results for any randomization used in plotting
rng(42, 'twister'); 

%% SECTION 2: UNIVERSAL DATA LOADING & PHYSICAL SCALING
% -------------------------------------------------------------------------
fprintf('--- Loading Multi-Method Results (jDAS Dataset) ---\n');

% 1. Load Raw jDAS Input Data
load('jdas_data.mat');  
[nt, nx] = size(jdas_data);

% Physical Axis Definition
tmax = 42;          % Seconds
xmax = 12e3;        % Meters (12 km)
time = linspace(0, tmax, nt);
dist_km = linspace(0, xmax, nx) / 1000; % Conversion to Kilometers

% 2. Load Baselines & Proposed Methods
if exist('sgkjdas.mat', 'file'), load('sgkjdas.mat'); else warning('SGK not found.'); end

% Loading SSDL Data (Note: Externally processed in Python)
if exist('ssdljdas.mat', 'file')
    load('ssdljdas.mat'); 
    fprintf('-> SSDL jDAS data loaded successfully (Python Output).\n');
else 
    warning('SSDL jDAS results not found.'); 
end

if exist('d_mharn_jDAS.mat', 'file'), load('d_mharn_jDAS.mat'); else warning('MHA-RN not found.'); end
if exist('d_haddl_jDAS1.mat', 'file'), load('d_haddl_jDAS1.mat'); else warning('HADDL not found.'); end

% Define Analysis Regions (Zoom ROI in physical units)
% Format: [km_start, t_start, km_width, t_width]
rect_pos1 = [8.0, 31, 2.0, 2.0];  % Deep Zone A (Red)
rect_pos2 = [7.0, 34, 3.0, 3.5];  % Deep Zone B (Magenta)
line_x = 5.5; % Trace of interest (km)

fprintf('Success: Environment ready and data loaded.\n');

%% SECTION 3: COMPARATIVE VISUALIZATION (GLOBAL VIEW)
% -------------------------------------------------------------------------
% Generate a comprehensive 2x3 grid to compare global reconstruction quality
% -------------------------------------------------------------------------

% =========================================================================
% FIGURE : GLOBAL jDAS BENCHMARK WITH SINGLE-TRACE PROFILE
% -------------------------------------------------------------------------
% This script displays the 2D seismic records for all methods and overlays
% a single-trace comparative amplitude plot in the final remaining tile.
% =========================================================================

data_list = {jdas_data, sgkjdas, ssdl, d_mharn, d_haddl_jDAS};
titles = {'Noisy', 'BP+SGK', 'SSDL', 'MHA-RN', 'HADDL'};
labels = {'(a)', '(b)', '(c)', '(d)', '(e)', '(f)'}; % Added (f) for the trace plot
colors = {'k', 'b', 'y', 'g', 'r'}; % Color scheme for the lines

figure('Name', 'Figure 13: Global jDAS Benchmark', 'Position', [100 100 1200 800], 'Color', 'w');

% Create a 2x3 tiled layout for high-resolution output
tlo = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

%% --- PART 1: 2D SEISMIC IMAGING (TILES 1 TO 5) ---

%% --- PART 1: 2D SEISMIC IMAGING (TILES 1 TO 5) ---
for k = 1:5
    ax = nexttile(k);
    
    % Imaging with physical axis scaling (km vs seconds)
    imagesc(dist_km, time, data_list{k}); 
    
    % High-fidelity formatting for publication
    colormap(ax, seis);
    caxis([-5 5]); % Amplitude clipping adjusted for jDAS range
    set(gca, 'YDir', 'reverse', 'LineWidth', 2, 'FontSize', 12, 'FontWeight', 'bold');
    title(titles{k}, 'FontSize', 14);
    
    % =====================================================================
    % AJOUT DES LABELS D'AXES (En Anglais pour publication)
    % =====================================================================
    xlabel('Distance (km)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
    % =====================================================================
    
    % Overlays & Annotations
    hold on;
    % Position labels slightly above the axis for better scannability
    text(-0.095, 1.07, labels{k}, 'Units', 'normalized', 'FontSize', 18, 'FontWeight', 'bold');
    
    % Primary and Secondary Zoom Windows
    rectangle('Position', rect_pos1, 'EdgeColor', 'r', 'LineWidth', 2);
    rectangle('Position', rect_pos2, 'EdgeColor', 'm', 'LineWidth', 2);
    hold off;
end

%% --- PART 2: SINGLE-TRACE AMPLITUDE COMPARISON (INDEPENDENT FIGURE) ---
% Create a new standalone window for the trace profile
figure('Name', 'Figure 14: Single-Trace Profile Analysis', 'Position', [200 200 600 800], 'Color', 'w');
ax_trace = subplot(1,1,1); 

hold on;
% Define the target channel index for extraction (e.g., center of the fiber or a key event)
% Change this index to target a specific noisy or high-signal channel
target_channel_idx = round(length(dist_km) / 2); 

% Plot the single trace over time for each dataset
for k = 1:5
    current_dataset = data_list{k};
    single_trace = current_dataset(:, target_channel_idx);
    
    % Overlay lines with distinct colors and line-widths
    if k == 5
        % Make the proposed HADDL stand out with a thicker red line
        plot(single_trace, time, 'Color', colors{k}, 'LineWidth', 2.0);
    else
        plot(single_trace, time, 'Color', colors{k}, 'LineWidth', 1.2);
    end
end

% Formatting to match the seismic panels' depth/time axis
set(gca, 'YDir', 'reverse', 'LineWidth', 2, 'FontSize', 16, 'FontWeight', 'bold');
ylim([min(time) max(time)]);
xlim([-6 6]); 
xlabel('Amplitude', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Time (s)', 'FontSize', 16, 'FontWeight', 'bold'); % Added ylabel since it's a standalone figure now
title(sprintf('Single-Trace Amplitude Comparison (Channel %d)', target_channel_idx), 'FontSize', 14);

% Add the panel label (f)
% text(-0.095, 1.07, labels{6}, 'Units', 'normalized', 'FontSize', 18, 'FontWeight', 'bold');

% Add a professional legend
legend(titles, 'Location', 'best', 'FontSize', 14, 'FontWeight', 'bold');
% grid on;
hold off;
%%
% % % %% --- PART 3: GLOBAL LAYOUT LABELS & COLORBAR ---
% % % % Global Grid Labels
% % % xlabel(tlo, 'Distance (km)', 'FontSize', 14, 'FontWeight', 'bold');
% % % ylabel(tlo, 'Time (s)', 'FontSize', 14, 'FontWeight', 'bold');
% % % 
% % % % Global Colorbar Integration (Applies to the seismic images)
% % % cb = colorbar(nexttile(5)); % Anchor colorbar to a seismic tile instead of the trace tile
% % % cb.Layout.Tile = 'east';
% % % cb.Label.String = 'Normalized Amplitude';
% % % cb.Label.FontSize = 12;
% % % cb.Label.FontWeight = 'bold';
% % % 
% % % fprintf('-> Figure 13 updated with single-trace comparison plot.\n');
%%
%% SECTION 3.2: ZOOMED COMPARISON (FOCUS ON ROI)
% -------------------------------------------------------------------------
% Cette section extrait la zone définie par rect_pos [km_start, t_start, width, height]
% -------------------------------------------------------------------------

%% SECTION 4: ZOOM-IN WITH FEATURE POINTERS (ZONE A)
% -------------------------------------------------------------------------
% 1. Définition des coordonnées des annotations (Distance (km), Temps (s))
% x,y : point de départ | u,v : direction et longueur
% Ces coordonnées sont basées sur ton image de référence.
% -------------------------------------------------------------------------

% Vecteurs pour les flèches (Distance, Temps, DirectionX, DirectionY)
arrow_x = [8.2,  9.8,  8.7,  9.8,  8.4,   8.3];  % Positions Distance (km)
arrow_y = [31.2, 31.2, 31.9, 32.4, 32.6,  32.7];  % Positions Time (s)
arrow_u = [0.15, 0.15, -0.15, 0.15, -0.10, -0.10]; % Décalage horizontal (Distance)
arrow_v = [0.15, 0.15, -0.15, 0.15, 0.05,  0.03]; % Décalage vertical (Temps)

% Coordonnées de la boîte bleue [x_min, y_min, largeur, hauteur]
box_pos = [8.25, 32.55, 0.1, 0.1]; 

% 2. Définition des limites de zoom (basées sur ton image)
z_x_min = 8;
z_x_max = 10;
z_y_min = 31;
z_y_max = 33;

figure('Name', 'Zoom-In Analysis (Zone B) with Feature Pointers', 'Position', [150 150 1200 800], 'Color', 'w');
tlo_zoom = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

for k = 1:5
    ax = nexttile;
    
    % Affichage des données (Toujours avec les axes physiques)
    imagesc(dist_km, time, data_list{k}); 
    
    % --- AJOUT DES ANNOTATIONS COMPARATIVES ---
    hold on; % Indispensable pour superposer
    
    % Flèches Rouges
    % 'AutoScale','off' garantit que la taille de la flèche est la même partout.
    quiver(arrow_x, arrow_y, arrow_u, arrow_v, 0, ...
        'Color', 'r', 'LineWidth', 2.5, 'MaxHeadSize', 3, 'AutoScale', 'off');
    
    % Boîte Bleue (rectangle)
    rectangle('Position', box_pos, 'EdgeColor', 'b', 'LineWidth', 3, 'LineStyle', '-');
    
    hold off;
    
    % Application du Zoom
    xlim([z_x_min, z_x_max]);
    ylim([z_y_min, z_y_max]);
    
    % Formatage identique pour la cohérence scientifique
    colormap(ax, seis);
    caxis([-5 5]); 
    set(gca, 'YDir', 'reverse', 'LineWidth', 2, 'FontSize', 14, 'FontWeight', 'bold');
    
    % Titre et Label de sous-figure
    title(titles{k}, 'FontSize', 14);
    text(-0.17, 1.05, labels{k}, 'Units', 'normalized', 'FontSize', 18, 'FontWeight', 'bold');
end

% Labels globaux pour le zoom
xlabel(tlo_zoom, 'Distance (km)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel(tlo_zoom, 'Time (s)', 'FontSize', 16, 'FontWeight', 'bold');

% Colorbar partagée pour le zoom
% cbz = colorbar;
% cbz.Layout.Tile = 'east';
% cbz.Label.String = 'Amplitude (Zoomed)';
% cbz.Label.FontWeight = 'bold';

% Titre de la figure de zoom pour le dossier
% title(tlo_zoom, 'Zoom-In View: Detailed Reflector Preservation (Zone B)', ...
% 'FontSize', 16, 'FontWeight', 'bold');

% Restauration (pour la reproductibilité de ton script global)
rng(42, 'twister');
%%
%% SECTION 4: ZOOM-IN WITH FEATURE POINTERS (ZONE B)
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% 1. Définition des limites dynamiques basées sur rect_pos2
z2_x_min = rect_pos2(1);
z2_x_max = rect_pos2(1) + rect_pos2(3);
z2_y_min = rect_pos2(2);
z2_y_max = rect_pos2(2) + rect_pos2(4);

% 2. Coordonnées des flèches pour cette zone spécifique
arrow_x2 = [7.2, 9.1, 7.4, 8.2, 9.6, 8.6]; 
arrow_y2 = [34.4, 34.7, 36.1, 37.3, 37.3, 36.7]; 
arrow_u2 = [0.4, 0.5, 0.4, 0.2, -0.2, 0.2]; 
arrow_v2 = [0.7, 0.7, 0.4, -0.5, -0.5, -0.4]; 

figure('Name', 'Zoom-In Analysis (Zone C - Deep)', 'Position', [160 160 1200 800], 'Color', 'w');
tlo_zoom2 = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

for k = 1:5
    ax = nexttile;
    
    % Affichage des données
    imagesc(dist_km, time, data_list{k}); 
    hold on; 
    
    % % Application des flèches spécifiques à la Zone C
    % quiver(arrow_x2, arrow_y2, arrow_u2, arrow_v2, 0, ...
    %     'Color', 'r', 'LineWidth', 2, 'MaxHeadSize', 2, 'AutoScale', 'off');
    
    % Application of specific arrows to Zone C
% OLD CODE:
% quiver(arrow_x2, arrow_y2, arrow_u2, arrow_v2, 0, ...
%     'Color', 'r', 'LineWidth', 2, 'MaxHeadSize', 2, 'AutoScale', 'off');

% NEW CODE (Reduced length):
scale_factor = 0.65; % Adjust this value (e.g., 0.3 for even shorter, 0.7 for longer)
quiver(arrow_x2, arrow_y2, arrow_u2 * scale_factor, arrow_v2 * scale_factor, 0, ...
    'Color', 'r', 'LineWidth', 2, 'MaxHeadSize', 3, 'AutoScale', 'off');

    % --- LA CORRECTION EST ICI ---
    % On utilise z2 au lieu de z (les limites de rect_pos2)
    xlim([z2_x_min, z2_x_max]);
    ylim([z2_y_min, z2_y_max]);
    % -----------------------------
    
    % Formatage
    colormap(ax, seis);
    caxis([-5 5]); 
    set(gca, 'YDir', 'reverse', 'LineWidth', 2, 'FontSize', 14, 'FontWeight', 'bold');
    
    title(titles{k}, 'FontSize', 16);
    text(-0.17, 1.05, labels{k}, 'Units', 'normalized', 'FontSize', 18, 'FontWeight', 'bold');
    
    hold off;
end

% Configuration globale
xlabel(tlo_zoom2, 'Distance (km)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel(tlo_zoom2, 'Time (s)', 'FontSize', 16, 'FontWeight', 'bold');
% cbz2 = colorbar;
% cbz2.Layout.Tile = 'east';
% cbz2.Label.String = 'Amplitude (Zoomed)';
% cbz2.Label.FontWeight = 'bold';
% title(tlo_zoom2, 'Zoom-In View: Detailed Reflector Preservation (Zone C)', ...
%       'FontSize', 16, 'FontWeight', 'bold');
%
%% SECTION 4: RESIDUAL ANALYSIS (QC)
% -------------------------------------------------------------------------
% Évaluation de l'énergie du bruit retiré pour chaque méthode
residuals = {jdas_data - sgkjdas, jdas_data - ssdl, ...
             jdas_data - d_mharn, jdas_data - d_haddl_jDAS};

% Subplot labels for structural navigation (a-d)
qc_labels = {'(a)', '(b)', '(c)', '(d)'};

figure('Name', 'QC Residuals Analysis', 'Position', [100 100 1000 800], 'Color', 'w');

% A 2x2 tiled layout is mathematically optimal and visually balanced for 4 items
tlo_qc = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for k = 1:4
    ax = nexttile(k);
    
    % Imaging with physical spatial-temporal coordinates
    imagesc(dist_km, time, residuals{k});
    
    % Technical display and amplitude clipping properties
    colormap(ax, seis); 
    caxis([-5 5]);
    set(ax, 'YDir', 'reverse', 'LineWidth', 2, 'FontSize', 14, 'FontWeight', 'bold');
    
    % Titles map to benchmarks (Skipping index 1 'Noisy' as these are differences)
    title(titles{k+1}, 'FontSize', 16, 'FontWeight', 'bold');
    
    % Add panel labels slightly above the axis borders
    hold on;
    text(-0.10, 1.07, qc_labels{k}, 'Units', 'normalized', ...
         'FontSize', 18, 'FontWeight', 'bold', 'Color', 'k');
    hold off;
    
    % Integrate a clear amplitude scale bar on the final panel (Tile 4)
    if k == 4
        cb_qc = colorbar(ax);
        ylabel(cb_qc, 'Residual Amplitude', 'FontSize', 12, 'FontWeight', 'bold');
    end
end

% Centered Global Grid Layout Constraints
xlabel(tlo_qc, 'Distance (km)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel(tlo_qc, 'Time (s)', 'FontSize', 16, 'FontWeight', 'bold');

fprintf('-> QC Residuals figure updated with sequential panel labels.\n');
% Ajout d'une colorbar globale pour les résidus
% cb_res = colorbar;
% cb_res.Layout.Tile = 'east';
% cb_res.Label.String = 'Amplitude Residual';
% cb_res.Label.FontWeight = 'bold';

%%
% Calculation of Local Similarity between denoised signal and removed noise
%  Local Similarity Parameters
rect = [20 20 1]; niter = 20; eps = 0; verb = 0;
% High similarity = High leakage (Bad) | Low similarity = High fidelity (Good)
s1 = haddl_localsimi(sgkjdas,          residuals{1}, rect, niter, eps, verb);
s2 = haddl_localsimi(ssdl,                 residuals{2}, rect, niter, eps, verb);
s3 = haddl_localsimi(d_mharn,      residuals{3}, rect, niter, eps, verb);
s4 = haddl_localsimi(d_haddl_jDAS',      residuals{4}, rect, niter, eps, verb);

simi_list = {s1, s2, s3, s4};

%% SECTION 10: COMBINED QC VISUALIZATION (2x4 PANEL)
% -------------------------------------------------------------------------
figure('Name', 'QC: Residuals and Local Similarity Comparison', ...
       'Position', [50 50 1500 900], 'Color', 'w');

tlo_qc = tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

methods_names = {titles{2}, titles{3}, titles{4}, titles{5}}; 
sub_labels_qc = {'(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)'};

% Define Zoom coordinates for detailed inspection
zCh = [rect_pos1(1), rect_pos1(1) + rect_pos1(3)];
zT  = [rect_pos1(2), rect_pos1(2) + rect_pos1(4)];

%% ROW 1: NOISE REMOVED (RESIDUALS)
for k = 1:4
    ax_res = nexttile(k);
    
    % Displaying the removed noise component
    imagesc(dist_km, time, residuals{k});
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
    imagesc(dist_km, time, simi_list{k}'); 
        % imagesc(dist_km, time, residuals{k});

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
%%
%% SECTION 10: COMBINED QC VISUALIZATION (2x4 PANEL)
% -------------------------------------------------------------------------
figure('Name', 'QC: Residuals and Local Similarity Comparison', ...
       'Position', [50 50 1500 900], 'Color', 'w');

tlo_qc = tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

methods_names = {titles{2}, titles{3}, titles{4}, titles{5}}; 
sub_labels_qc = {'(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)'};

% Define Zoom coordinates for detailed inspection
zCh = [rect_pos2(1), rect_pos2(1) + rect_pos2(3)];
zT  = [rect_pos2(2), rect_pos2(2) + rect_pos2(4)];

%% ROW 1: NOISE REMOVED (RESIDUALS)
for k = 1:4
    ax_res = nexttile(k);
    
    % Displaying the removed noise component
    imagesc(dist_km, time, residuals{k});
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
    imagesc(dist_km, time, simi_list{k}'); 
        % imagesc(dist_km, time, residuals{k});

    xlim(zCh); ylim(zT);
    
    colormap(ax_sim, jet); 
    caxis([0 0.75]); % Scale focused on detecting even minor leakage
    
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