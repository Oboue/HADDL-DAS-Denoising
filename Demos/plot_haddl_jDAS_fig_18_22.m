clc; clear; close all;
% =========================================================================
% SCRIPT: plot_haddl_jDAS_fig_18_22.m
% -------------------------------------------------------------------------
% Article Title: "A Hybrid Attention-Driven Deep Learning Framework for 
%                Denoising DAS Data"
% Authors:      Oboué, Y. A. S. I., Chen, Y., & Chen, Y.
% Journal:      Geophysics 
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

rng(42, 'twister'); 

%% SECTION 2: UNIVERSAL DATA LOADING & PHYSICAL SCALING
% -------------------------------------------------------------------------
fprintf('--- Loading Multi-Method Results (jDAS Dataset) ---\n');

% 1. Load Raw jDAS Input Data
if exist('jdas_data.mat', 'file'), load('jdas_data.mat'); else, error('jdas_data.mat not found'); end

% Ensure data is oriented [nt x nx] internally for calculations
if size(jdas_data, 1) < size(jdas_data, 2)
    % Size is [nt x nx] = (650 x 2048)
    nt = size(jdas_data, 1);
    nx = size(jdas_data, 2);
else
    % Size was loaded as [nx x nt]
    jdas_data = jdas_data';
    [nt, nx] = size(jdas_data);
end

% Physical Axis Definition
tmax = 42;          % Seconds
xmax = 12e3;        % Meters (12 km)
time = linspace(0, tmax, nt);
dist_km = linspace(0, xmax, nx) / 1000; % Conversion to Kilometers

% 2. Load Baselines & Proposed Methods
if exist('sgkjdas.mat', 'file'), load('sgkjdas.mat'); else, warning('SGK not found.'); end
if exist('ssdljdas.mat', 'file'), load('ssdljdas.mat'); else, warning('SSDL not found.'); end
if exist('d_mharn_jDAS1.mat', 'file'), load('d_mharn_jDAS1.mat'); else, warning('MHA-RN not found.'); end
if exist('d_haddl_jDAS1.mat', 'file'), load('d_haddl_jDAS1.mat'); else, warning('HADDL not found.'); end

% Orientation check for loaded datasets
if exist('sgkjdas', 'var') && size(sgkjdas, 1) ~= nt, sgkjdas = sgkjdas'; end
if exist('ssdl', 'var') && size(ssdl, 1) ~= nt, ssdl = ssdl'; end
if exist('d_mharn', 'var') && size(d_mharn, 1) ~= nt, d_mharn = d_mharn'; end
if exist('d_haddl_jDAS', 'var') && size(d_haddl_jDAS, 1) ~= nt, d_haddl_jDAS = d_haddl_jDAS'; end

% Define Analysis Regions (Zoom ROI in physical units)
% Format: [km_start, t_start, km_width, t_width]
rect_pos1 = [8.0, 31.0, 2.0, 2.0];  % Deep Zone A (Red)
rect_pos2 = [7.0, 34.0, 3.0, 3.5];  % Deep Zone B (Magenta)

fprintf('Success: Environment ready and data loaded.\n');

%% SECTION 3: COMPARATIVE VISUALIZATION (GLOBAL VIEW)
% -------------------------------------------------------------------------
data_list = {jdas_data, sgkjdas, ssdl, d_mharn, d_haddl_jDAS};
titles = {'Noisy', 'BP+SGK', 'SSDL', 'MHA-RN', 'HADDL'};
labels = {'(a)', '(b)', '(c)', '(d)', '(e)', '(f)'};
colors = {'k', 'b', 'y', 'g', 'r'};


% Définition de la nouvelle zone d'intérêt (ROI)
% Format: [x_min (km), y_min (s), width (km), height (s)]
rect_new = [5.0, 25.0, 1.5, 10.0]; 

figure('Name', 'Figure 13: Global jDAS Benchmark', 'Position', [100 100 1200 800], 'Color', 'w');
tlo = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% 2D SEISMIC IMAGING (TILES 1 TO 5)
for k = 1:5
    ax = nexttile(k);
    
    % Transposition data_list{k}' for correct channel-time alignment
    imagesc(dist_km, time, data_list{k}'); 
    
    colormap(ax, seis);
    caxis([-5 5]); 
    set(gca, 'YDir', 'reverse', 'LineWidth', 2, 'FontSize', 12, 'FontWeight', 'bold');
    title(titles{k}, 'FontSize', 14);
    
    xlabel('Distance (km)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
    
    hold on;
    text(-0.095, 1.07, labels{k}, 'Units', 'normalized', 'FontSize', 18, 'FontWeight', 'bold');
    
    % Nouveau cadre ROI (Zone 5-6.5 km / 25-35 s)
    rectangle('Position', rect_new, 'EdgeColor', 'r', 'LineWidth', 2, 'LineStyle', '--');
    
    hold off;
end

%% SECTION 3.1: SINGLE-TRACE AMPLITUDE COMPARISON
% -------------------------------------------------------------------------
figure('Name', 'Figure 14: Single-Trace Profile Analysis', 'Position', [200 200 600 800], 'Color', 'w');
hold on;

target_channel_idx = round(length(dist_km) / 2); 

for k = 1:5
    current_dataset = data_list{k};
    single_trace = current_dataset(:, target_channel_idx);
    
    if k == 5
        plot(single_trace, time, 'Color', colors{k}, 'LineWidth', 2.0);
    else
        plot(single_trace, time, 'Color', colors{k}, 'LineWidth', 1.2);
    end
end

set(gca, 'YDir', 'reverse', 'LineWidth', 2, 'FontSize', 16, 'FontWeight', 'bold');
ylim([min(time) max(time)]);
xlim([-6 6]); 
xlabel('Amplitude', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Time (s)', 'FontSize', 16, 'FontWeight', 'bold');
title(sprintf('Single-Trace Amplitude Comparison (Channel %d)', target_channel_idx), 'FontSize', 14);
legend(titles, 'Location', 'best', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
hold off;

% -------------------------------------------------------------------------
%% SECTION 4: ZOOM-IN ANALYSIS (NEW ROI: 5-6.5 km / 25-35 s) WITH FEATURE POINTERS
% -------------------------------------------------------------------------
z_x_min = 5.0;
z_x_max = 6.5;
z_y_min = 25.0;
z_y_max = 35.0;

% Coordonnées des 3 flèches rouges d'après la figure (a)
% arrow_x / arrow_y : origine de la flèche
% arrow_u / arrow_v : composantes du vecteur (u = dx, v = dy)
arrow_x = [5.75, 6.17, 5.58]; 
arrow_y = [25.50, 30.00, 34.10]; 
arrow_u = [0.12, -0.15, -0.13]; 
arrow_v = [0.80, -0.80, -0.90]; 

figure('Name', 'Zoom-In Analysis (ROI: 5-6.5 km / 25-35 s)', 'Position', [150 150 1200 800], 'Color', 'w');
tlo_zoom = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

for k = 1:5
    ax = nexttile;
    
    % Affichage des données [nt x nx]
    imagesc(dist_km, time, data_list{k}'); 
    hold on;
    
    % Tracé des 3 flèches rouges sur toutes les sous-figures (a à e)
    quiver(arrow_x, arrow_y, arrow_u, arrow_v, 0, ...
        'Color', 'r', 'LineWidth', 2.5, 'MaxHeadSize', 2.5, 'AutoScale', 'off');
    
    % Limites strictes de la fenêtre d'intérêt
    xlim([z_x_min, z_x_max]);
    ylim([z_y_min, z_y_max]);
    
    % Style graphique
    colormap(ax, seis);
    caxis([-5 5]); 
    set(gca, 'YDir', 'reverse', 'LineWidth', 2, 'FontSize', 14, 'FontWeight', 'bold');
    
    title(titles{k}, 'FontSize', 14);
    text(-0.17, 1.05, labels{k}, 'Units', 'normalized', 'FontSize', 18, 'FontWeight', 'bold');
    
    hold off;
end

xlabel(tlo_zoom, 'Distance (km)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel(tlo_zoom, 'Time (s)', 'FontSize', 16, 'FontWeight', 'bold');

%% SECTION 6: RESIDUAL ANALYSIS (QC)
% -------------------------------------------------------------------------
residuals = {jdas_data - sgkjdas, ...
             jdas_data - ssdl, ...
             jdas_data - d_mharn, ...
             jdas_data - d_haddl_jDAS};

qc_labels = {'(a)', '(b)', '(c)', '(d)'};

figure('Name', 'QC Residuals Analysis', 'Position', [100 100 1000 800], 'Color', 'w');
tlo_qc = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for k = 1:4
    ax = nexttile(k);
    
    imagesc(dist_km, time, residuals{k}');
    
    colormap(ax, seis); 
    caxis([-5 5]);
    set(ax, 'YDir', 'reverse', 'LineWidth', 2, 'FontSize', 14, 'FontWeight', 'bold');
    
    title(titles{k+1}, 'FontSize', 16, 'FontWeight', 'bold');
    
    hold on;
    text(-0.10, 1.07, qc_labels{k}, 'Units', 'normalized', ...
         'FontSize', 18, 'FontWeight', 'bold', 'Color', 'k');
    hold off;
    
    if k == 4
        cb_qc = colorbar(ax);
        ylabel(cb_qc, 'Residual Amplitude', 'FontSize', 12, 'FontWeight', 'bold');
    end
end

xlabel(tlo_qc, 'Distance (km)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel(tlo_qc, 'Time (s)', 'FontSize', 16, 'FontWeight', 'bold');
fprintf('-> QC Residuals figure updated.\n');
%%
%% SECTION 7: LOCAL SIMILARITY ANALYSIS
% -------------------------------------------------------------------------
fprintf('--- Computing Local Similarity Maps ---\n');

% Paramètres de la similarité locale : [fênetre_temps, fenêtre_espace, pas]
rect = [20, 20, 1]; 
niter = 20; 
eps = 0; 
verb = 0;

% Calcul de la similarité locale entre la donnée filtrée et le résidu (bruit extrait)
% Assure que les matrices transmises ont des dimensions identiques [nt x nx]
s1 = haddl_localsimi(sgkjdas,      residuals{1}, rect, niter, eps, verb);
s2 = haddl_localsimi(ssdl,         residuals{2}, rect, niter, eps, verb);
s3 = haddl_localsimi(d_mharn,      residuals{3}, rect, niter, eps, verb);
s4 = haddl_localsimi(d_haddl_jDAS, residuals{4}, rect, niter, eps, verb);

simi_list = {s1, s2, s3, s4};
fprintf('-> Local similarity computation complete.\n');
%% SECTION 8: COMBINED QC VISUALIZATION (NEW ROI: 5-6.5 km / 25-35 s - 2x4 PANEL)
% -------------------------------------------------------------------------
% figure('Name', 'QC: Residuals and Local Similarity Comparison (New ROI)', ...
%        'Position', [50 50 1500 900], 'Color', 'w');
% tlo_qc1 = tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
% 
% methods_names = {titles{2}, titles{3}, titles{4}, titles{5}}; 
% sub_labels_qc = {'(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)'};
% 
% % Définition des limites de la nouvelle ROI
% zCh1 = [5.0, 6.5];  % Distance (km)
% zT1  = [25.0, 35.0]; % Temps (s)
% 
% % ROW 1: RESIDUALS (NOISE REMOVED)
% for k = 1:4
%     ax_res = nexttile(k);
%     imagesc(dist_km, time, residuals{k}');
%     xlim(zCh1); ylim(zT1);
% 
%     colormap(ax_res, seis); 
%     caxis([-0.5 0.5]); 
% 
%     set(gca, 'YDir', 'reverse', 'LineWidth', 1.5, 'FontWeight', 'bold', 'XTickLabel', []);
%     title(methods_names{k}, 'FontSize', 14, 'FontWeight', 'bold');
%     text(-0.1, 1.06, sub_labels_qc{k}, 'Units', 'normalized', 'FontSize', 18, 'FontWeight', 'bold');
% end
% 
% % ROW 2: LOCAL SIMILARITY MAPS
% for k = 1:4
%     ax_sim = nexttile(k+4);
%     imagesc(dist_km, time, simi_list{k}'); 
%     xlim(zCh1); ylim(zT1);
% 
%     colormap(ax_sim, jet); 
%     caxis([0 1]); 
% 
%     set(gca, 'YDir', 'reverse', 'LineWidth', 1.5, 'FontWeight', 'bold');
%     title(methods_names{k}, 'FontSize', 14, 'FontWeight', 'bold');
%     text(-0.1, 1.06, sub_labels_qc{k+4}, 'Units', 'normalized', 'FontSize', 18, 'FontWeight', 'bold');
% end
% 
% xlabel(tlo_qc1, 'Distance (km)', 'FontSize', 14, 'FontWeight', 'bold');
% ylabel(tlo_qc1, 'Time (s)', 'FontSize', 14, 'FontWeight', 'bold');
% 
% cb2 = colorbar(nexttile(8)); 
% cb2.Label.String = 'Local Similarity';
% cb2.Label.FontWeight = 'bold';