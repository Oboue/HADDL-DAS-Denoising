clc;clear;close all; 

% clc;clear;close all;
load micro_sf_3001_334_3.mat 
d=LO_scale(data(:,:,1),2);
[n1,n2, n3]=size(d);

% load synthdas4.mat           

% load dnoise3.mat

% load dsynthDAS3.mat
% 
load dnoiseSynthDAS.mat 

dn=dnoiseSynthDAS; 

psnr_Noisy = yc_snr(d,dn,2) % Noisy

  figure;
    LO_imagesc(d);
    % % % % % title(sprintf('Denoised Data at Iteration %d', i));
    ylabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Panel', 'FontSize', 12, 'FontWeight', 'bold');
    colormap(seis);
    caxis([-0.5 0.5]);
    set(gca, 'LineWidth', 2, 'FontSize', 12, 'FontWeight', 'bold');

    figure;
    LO_imagesc(dn);
    title(sprintf('Denoised Data at Iteration %d', i));
    ylabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Panel', 'FontSize', 12, 'FontWeight', 'bold');
    colormap(seis);
    caxis([-0.5 0.5]);
    set(gca, 'LineWidth', 2, 'FontSize', 12, 'FontWeight', 'bold');

[n1,n2,n3]=size(dn)
dt = 0.004;
t = [0:n1-1] * dt; 
x = [1:n2];

%%

load('/Users/oboue/Desktop/Desktop/HUDL/DenoisedSynth2FORGEDASData_rmse/denoised_data_iteration_1.mat')
% % % % % % %% local similarity
rect=[20,20,1];niter=20;eps=0;verb=0;
[simi1]=localsimi(dn-d1_denoised,d1_denoised,rect,niter,eps,verb);

psnr_Noisy = yc_snr(d,d1_denoised,2) % Noisy

load('/Users/oboue/Desktop/Desktop/HUDL/DenoisedSynth2FORGEDASData_rmse/denoised_data_iteration_2.mat')
rect=[20,20,1];niter=20;eps=0;verb=0;
[simi2]=localsimi(dn-d1_denoised,d1_denoised,rect,niter,eps,verb);

psnr_Noisy = yc_snr(d,d1_denoised,2) % Noisy

load('/Users/oboue/Desktop/Desktop/HUDL/DenoisedSynth2FORGEDASData_rmse/denoised_data_iteration_3.mat')
rect=[20,20,1];niter=20;eps=0;verb=0;
[simi3]=localsimi(dn-d1_denoised,d1_denoised,rect,niter,eps,verb);

load('/Users/oboue/Desktop/Desktop/HUDL/DenoisedSynth2FORGEDASData_rmse/denoised_data_iteration_4.mat')
rect=[20,20,1];niter=20;eps=0;verb=0;
[simi4]=localsimi(dn-d1_denoised,d1_denoised,rect,niter,eps,verb);

load('/Users/oboue/Desktop/Desktop/HUDL/DenoisedSynth2FORGEDASData_rmse/denoised_data_iteration_5.mat')
rect=[20,20,1];niter=20;eps=0;verb=0;
[simi5]=localsimi(dn-d1_denoised,d1_denoised,rect,niter,eps,verb);

load('/Users/oboue/Desktop/Desktop/HUDL/DenoisedSynth2FORGEDASData_rmse/denoised_data_iteration_6.mat')
rect=[20,20,1];niter=20;eps=0;verb=0;
[simi6]=localsimi(dn-d1_denoised,d1_denoised,rect,niter,eps,verb);

load('/Users/oboue/Desktop/Desktop/HUDL/DenoisedSynth2FORGEDASData_rmse/denoised_data_iteration_7.mat')
rect=[20,20,1];niter=20;eps=0;verb=0;
[simi7]=localsimi(dn-d1_denoised,d1_denoised,rect,niter,eps,verb);

load('/Users/oboue/Desktop/Desktop/HUDL/DenoisedSynth2FORGEDASData_rmse/denoised_data_iteration_8.mat')
rect=[20,20,1];niter=20;eps=0;verb=0;
[simi8]=localsimi(dn-d1_denoised,d1_denoised,rect,niter,eps,verb);

load('/Users/oboue/Desktop/Desktop/HUDL/DenoisedSynth2FORGEDASData_rmse/denoised_data_iteration_9.mat')
rect=[20,20,1];niter=20;eps=0;verb=0;
[simi9]=localsimi(dn-d1_denoised,d1_denoised,rect,niter,eps,verb);

load('/Users/oboue/Desktop/Desktop/HUDL/DenoisedSynth2FORGEDASData_rmse/denoised_data_iteration_10.mat')
rect=[20,20,1];niter=20;eps=0;verb=0;
[simi10]=localsimi(dn-d1_denoised,d1_denoised,rect,niter,eps,verb);

%%
figure('units','normalized','Position',[0.0 0.0 1, 1],'color','w');
subplot(251); hold on
imagesc(x,t,simi1);colormap(jet);colormap(jet);
c = colorbar;c.Label.String = 'Local similarity';c.Label.FontSize = 30;%c.Label.FontWeight = bold;
caxis([0,1]);
ylabel('Time (s)','Fontsize',30);
xlabel('Trace','Fontsize',30);
% title('Noise','Fontsize',30);
set(gca,'Linewidth',2,'Fontsize',30);
% print(gcf,'-depsc','-r200','l_simi_sosvmf.eps');

subplot(252); hold on
imagesc(x,t,simi2);colormap(jet);colormap(jet);
c = colorbar;c.Label.String = 'Local similarity';c.Label.FontSize = 30;%c.Label.FontWeight = bold;
caxis([0,1]);
ylabel('Time (s)','Fontsize',30);
xlabel('Trace','Fontsize',30);
% title('Noise','Fontsize',30);
set(gca,'Linewidth',2,'Fontsize',30);
% print(gcf,'-depsc','-r200','l_simi_sosvmf.eps');

subplot(253); hold on
imagesc(x,t,simi3);colormap(jet);colormap(jet);
c = colorbar;c.Label.String = 'Local similarity';c.Label.FontSize = 30;%c.Label.FontWeight = bold;
caxis([0,1]);
ylabel('Time (s)','Fontsize',30);
xlabel('Trace','Fontsize',30);
% title('Noise','Fontsize',30);
set(gca,'Linewidth',2,'Fontsize',30);
% print(gcf,'-depsc','-r200','l_simi_sosvmf.eps');

subplot(254); hold on
imagesc(x,t,simi4);colormap(jet);colormap(jet);
c = colorbar;c.Label.String = 'Local similarity';c.Label.FontSize = 30;%c.Label.FontWeight = bold;
caxis([0,1]);
ylabel('Time (s)','Fontsize',30);
xlabel('Trace','Fontsize',30);
% title('Noise','Fontsize',30);
set(gca,'Linewidth',2,'Fontsize',30);
% print(gcf,'-depsc','-r200','l_simi_sosvmf.eps');

subplot(255); hold on
imagesc(x,t,simi5);colormap(jet);colormap(jet);
c = colorbar;c.Label.String = 'Local similarity';c.Label.FontSize = 30;%c.Label.FontWeight = bold;
caxis([0,1]);
ylabel('Time (s)','Fontsize',30);
xlabel('Trace','Fontsize',30);
% title('Noise','Fontsize',30);
set(gca,'Linewidth',2,'Fontsize',30);
% print(gcf,'-depsc','-r200','l_simi_sosvmf.eps');

subplot(256); hold on
imagesc(x,t,simi6);colormap(jet);colormap(jet);
c = colorbar;c.Label.String = 'Local similarity';c.Label.FontSize = 30;%c.Label.FontWeight = bold;
caxis([0,1]);
ylabel('Time (s)','Fontsize',30);
xlabel('Trace','Fontsize',30);
% title('Noise','Fontsize',30);
set(gca,'Linewidth',2,'Fontsize',30);
% print(gcf,'-depsc','-r200','l_simi_sosvmf.eps');

subplot(257); hold on
imagesc(x,t,simi7);colormap(jet);colormap(jet);
c = colorbar;c.Label.String = 'Local similarity';c.Label.FontSize = 30;%c.Label.FontWeight = bold;
caxis([0,1]);
ylabel('Time (s)','Fontsize',30);
xlabel('Trace','Fontsize',30);
% title('Noise','Fontsize',30);
set(gca,'Linewidth',2,'Fontsize',30);
% print(gcf,'-depsc','-r200','l_simi_sosvmf.eps');

subplot(258); hold on
imagesc(x,t,simi8);colormap(jet);colormap(jet);
c = colorbar;c.Label.String = 'Local similarity';c.Label.FontSize = 30;%c.Label.FontWeight = bold;
caxis([0,1]);
ylabel('Time (s)','Fontsize',30);
xlabel('Trace','Fontsize',30);
% title('Noise','Fontsize',30);
set(gca,'Linewidth',2,'Fontsize',30);
% print(gcf,'-depsc','-r200','l_simi_sosvmf.eps');

subplot(259); hold on
imagesc(x,t,simi9);colormap(jet);colormap(jet);
c = colorbar;c.Label.String = 'Local similarity';c.Label.FontSize = 30;%c.Label.FontWeight = bold;
caxis([0,1]);
ylabel('Time (s)','Fontsize',30);
xlabel('Trace','Fontsize',30);
% title('Noise','Fontsize',30);
set(gca,'Linewidth',2,'Fontsize',30);
% print(gcf,'-depsc','-r200','l_simi_sosvmf.eps');

subplot(2,5,10); hold on
imagesc(x,t,simi10);colormap(jet);colormap(jet);
c = colorbar;c.Label.String = 'Local similarity';c.Label.FontSize = 30;%c.Label.FontWeight = bold;
caxis([0,1]);
ylabel('Time (s)','Fontsize',30);
xlabel('Trace','Fontsize',30);
% title('Noise','Fontsize',30);
set(gca,'Linewidth',2,'Fontsize',30);
% print(gcf,'-depsc','-r200','l_simi_sosvmf.eps');
%%

load('/Users/oboue/Desktop/Desktop/HUDL/DenoisedSynth2FORGEDASData_rmse/denoised_data_iteration_1.mat')
% % % % % % %% local similarity
% rect=[20,20,1];niter=20;eps=0;verb=0;
% [simi1]=localsimi(dn-d1_denoised,d1_denoised,rect,niter,eps,verb);

  figure;
    LO_imagesc(d1_denoised);
    % % % % % title(sprintf('Denoised Data at Iteration %d', i));
    ylabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Panel', 'FontSize', 12, 'FontWeight', 'bold');
    colormap(seis);
    caxis([-0.5 0.5]);
    set(gca, 'LineWidth', 2, 'FontSize', 12, 'FontWeight', 'bold');

psnr_Noisy = yc_snr(d,d1_denoised,2) % Noisy

load('/Users/oboue/Desktop/Desktop/HUDL/DenoisedSynth2FORGEDASData_rmse/denoised_data_iteration_2.mat')
% rect=[20,20,1];niter=20;eps=0;verb=0;
% [simi2]=localsimi(dn-d1_denoised,d1_denoised,rect,niter,eps,verb);

psnr_Noisy = yc_snr(d,d1_denoised,2) % Noisy

    figure;
    LO_imagesc(d1_denoised);
    title(sprintf('Denoised Data at Iteration %d', i));
    ylabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Panel', 'FontSize', 12, 'FontWeight', 'bold');
    colormap(seis);
    caxis([-0.5 0.5]);
    set(gca, 'LineWidth', 2, 'FontSize', 12, 'FontWeight', 'bold');

%%

figure('units','normalized','Position',[0.0 0.0 1, 1],'color','w');
% subplot(251); hold on
imagesc(x,t,simi1);colormap(jet);colormap(jet);
c = colorbar;c.Label.String = 'Local similarity';c.Label.FontSize = 30;%c.Label.FontWeight = bold;
caxis([0,1]);
ylabel('Time (s)','Fontsize',30);
xlabel('Trace','Fontsize',30);
% title('Noise','Fontsize',30);
set(gca,'Linewidth',2,'Fontsize',30);
% print(gcf,'-depsc','-r200','l_simi_sosvmf.eps');

figure('units','normalized','Position',[0.0 0.0 1, 1],'color','w');
% subplot(251); hold on
imagesc(x,t,simi2);colormap(jet);colormap(jet);
c = colorbar;c.Label.String = 'Local similarity';c.Label.FontSize = 30;%c.Label.FontWeight = bold;
caxis([0,1]);
ylabel('Time (s)','Fontsize',30);
xlabel('Trace','Fontsize',30);
% title('Noise','Fontsize',30);
set(gca,'Linewidth',2,'Fontsize',30);
% print(gcf,'-depsc','-r200','l_simi_sosvmf.eps');