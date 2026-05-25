% Demo for erratic noise suppression using iterative SOSVMF with sparsity constraint
% Prepared By Guangtan Huang, Min Bai, and Yangkang Chen
% Dec, 2020
%
% References
% Huang, G., M. Bai, Q. Zhao, W. Chen, and Y. Chen, 2021, Erratic noise suppression using iterative structure-oriented space-varying median filtering with sparsity constraint, Geophysical Prospecting, 69, 101-121.
% Chen, Y., S. Zu, Y. Wang, and X. Chen, 2020, Deblending of simultaneous-source data using a structure-oriented space varying median filter, Geophysical Journal International, 222, 1805�1�723.
% Zhao, Q., Q. Du, X. Gong, and Y. Chen, 2018, Signal-preserving erratic noise attenuation via iterative robust sparsity-promoting filter, IEEE Transactions on Geoscience and Remote Sensing, 56, 1558-0644.

clc;clear;close all;
%please download seistr package from https://github.com/chenyk1990/seistr
addpath(genpath('seistr/'));
addpath(genpath('subroutines/'));
% is_real=1;           % Type of the transform(0: complex-valued curvelets,1: real-valued curvelets)
% finest=2;            % Chooses one of two possibilities for the coefficients at the finest level(1: curvelets,2: wavelets)
% % alpha=1.2;         % ������׼���alpha����ֵ��1.2���ҽ�Ϊ���룩
% alpha=1.7;
% niter=10;

dc=levents(200);dc=yc_scale(dc);

[n1,n2]=size(dc);

mask=rand(1,n2);
mask(logical(mask<0.9))=0;
mask(logical(mask>=0.9))=1;

err_n=zeros(size(dc));
for i=1:n1
    randn('state',123456+i);
    err_n(i,:)=0.5*randn(1,n2).*mask;
end

randn('state',201920);
ran_n=0.1*randn(n1,n2);

dn=dc+err_n+ran_n;

dt=0.004;
t=[0:n1-1]*dt; x=[1:n2];

% figure;imagesc([dc,dn]);caxis([-0.5,0.5]);colormap(seis);
%% BP
    d1=das_bandpass(dn,0.0005,0,200,6,6,0,0);%
    d_bp=d1;
%% FK
%    d1=d1-das_fk_dip(d1,0.02);%
%    d_bpsomffk=d1;
t1=9; beta=1;
d_bpsp=fkt1(d_bp,'ps',t1);
 %% SOMF
    [pp]=str_dip2d(d_bpsp,2,10,2,0.01,1,0.000001,[20,8,1],1);%figure;das_imagesc(pp);colormap(jet);
    ns=3;
    order=2;
    eps=0.01;
    type_mf=0;
    ifsmooth=0;
    
    [~,d1]=das_pwsmooth_lop_mf(pp,[],n1,n2,ns,order,eps,n1*n2,n1*n2,type_mf,ifsmooth,d_bpsp,[]);%SOMF
    d1=reshape(d1,n1,n2);
    d_bpspsomf=d1;
        %%
is_real=1;           % Type of the transform(0: complex-valued curvelets,1: real-valued curvelets)
finest=1;            % Chooses one of two possibilities for the coefficients at the finest level(1: curvelets,2: wavelets)
% alpha=1.2;         % ������׼���alpha����ֵ��1.2���ҽ�Ϊ���룩
alpha=1.8;
niter=10;

% dc=levents(200);dc=yc_scale(dc);
% load Curveddata
% dc=data;


[n1,n2]=size(d_bpspsomf);

% mask=rand(1,n2);
% mask(logical(mask<0.9))=0;
% mask(logical(mask>=0.9))=1;
% 
% err_n=zeros(size(dc));
% for i=1:n1
%     randn('state',123456+i);
%     err_n(i,:)=0.5*randn(1,n2).*mask;
% end
% 
% randn('state',201920);
% ran_n=0.1*randn(n1,n2);
% 
% % dn=dc+err_n+ran_n;
% % 
% % save dncurved dn
% 
% load dncurved 

dt=0.004;
t=[0:n1-1]*dt; x=[1:n2];

% figure;imagesc([dc,dn]);caxis([-0.5,0.5]);colormap(seis);

F=ones(n1,n2);                                  % ones(n)����n*n��1����Ƶ����
X=fftshift(ifft2(F))*sqrt(prod(size(F)));  
% size(X)
% prod����size(F)�ĳ˻�,X��һ�������壬��׼��Ϊ1
C=fdct_wrapping(X,0,finest);                    % ������任�õ��Ǹ���,�����ΪС���任
% Compute norm of curvelets (exact)
E=cell(size(C));
for s=1:length(C)
    E{s}=cell(size(C{s}));
    for w=1:length(C{s})
         A=C{s}{w};
         E{s}{w}=sqrt(sum(sum(A.*conj(A)))/prod(size(A)));    %����A��ģ�������׼������򵥵����
    end
end

Cdn=fdct_wrapping(d_bpspsomf,is_real,finest);     %������任�õ���ʵ��,�����ΪС���任
Smax=length(Cdn);
Sigma0=alpha*median(median(abs(Cdn{Smax}{1})))/0.58;     %��ȡ������׼�ѡȡ���߶�
Sigma=Sigma0;
sigma=[Sigma,5*Sigma,2*Sigma, Sigma, 0.6*Sigma,Sigma/5];
sigma=[Sigma,5*Sigma,2*Sigma, Sigma, 0.6*Sigma,Sigma*0.1];
sigma=[Sigma,linspace(2.5*Sigma,0.5*Sigma,niter)];
Sigma=sigma(1);

Ct=Cdn;

for s=2:length(Cdn)
    thresh=Sigma+Sigma*s;    %���������Ϊ4*sigma
    for w=1:length(Cdn{s})
         Ct{s}{w}=Cdn{s}{w}.*(abs(Cdn{s}{w})>thresh*E{s}{w});  %������ֵ�ı���
    end
end
 
d_bpspsomfcuv=real(ifdct_wrapping(Ct,is_real,n1,n2));
%%    
noi1=dn-d_bpspsomfcuv;
% prepare paramters for ortho
rect = zeros(3,1);
rect(1) = 5;
rect(2) = 5;
rect(3) = 1;
eps=0.00000000000001;%too strong, do not know why
eps=0;
niter=20;
verb=1;

[d_bpspsomfcuvlow,noi2,low]=yc_localortho(d_bpspsomfcuv,noi1,rect,niter,eps,verb);    
%%

figure(1);das_imagesc([dc,dn]);

figure(2);das_imagesc([dn,d_bp,dn-d_bp]);

figure(3);das_imagesc([dn,d_bpsp,dn-d_bpsp]);

figure(4);das_imagesc([dn,d_bpspsomf,dn-d_bpspsomf]);

figure(5);das_imagesc([dn,d_bpspsomfcuv,dn-d_bpspsomfcuv]);

figure(6);das_imagesc([dn,d_bpspsomfcuvlow,dn-d_bpspsomfcuvlow]);

% %
% % combined figure
% % combined figure
% figure('units','normalized','Position',[0.0 0.0 0.5, 1],'color','w');
% % subplot(4,2,1);
% das_imagesc(eq2,100,2,x,t);
% ylabel('Time (s)','Fontsize',10,'fontweight','bold');
% xlabel('Channel','Fontsize',10,'fontweight','bold');
% set(gca,'Linewidth',2,'Fontsize',10,'Fontweight','bold');
% text(n2/2,-0.05,'Noisy','color','k','Fontsize',10,'fontweight','bold','HorizontalAlignment','center');
% text(-200,-0.1,'(a)','color','k','Fontsize',10,'fontweight','bold','HorizontalAlignment','center');
% % annotation(gcf,'rectangle',[0.13 0.88 0.334 0.020],'linewidth',2,'color','g');
% %
% figure;
% % subplot(4,2,2);
% das_imagesc(d_bp2,100,2,x,t);
% ylabel('Time (s)','Fontsize',10,'fontweight','bold');
% xlabel('Channel','Fontsize',10,'fontweight','bold');
% set(gca,'Linewidth',2,'Fontsize',10,'Fontweight','bold');
% text(n2/2,-0.05,'BP','color','k','Fontsize',10,'fontweight','bold','HorizontalAlignment','center');
% text(n2+ngap+n2/2,-0.05,'Removed noise','color','k','Fontsize',10,'fontweight','bold','HorizontalAlignment','center');
% annotation(gcf,'textarrow',[0.69 0.700],...
%     [0.85 0.90],'Color','r','TextColor','r','HorizontalAlignment','center',...
%     'String',{'High-amplitude erratic noise'},...
%     'LineWidth',2,...
%     'FontSize',10,'fontweight','bold');
% text(-200,-0.1,'(b)','color','k','Fontsize',10,'fontweight','bold','HorizontalAlignment','center')
% % annotation(gcf,'rectangle',[0.420921875 0.112752721617418 0.035328125 0.810264385692069],'linewidth',5,'color','[1 0 0]');
% %
% figure;
% das_imagesc(d_bpsp2,100,2,x,t);
% ylabel('Time (s)','Fontsize',10,'fontweight','bold');
% xlabel('Channel','Fontsize',10,'fontweight','bold');
% set(gca,'Linewidth',2,'Fontsize',10,'Fontweight','bold');
% text(n2/2,-0.05,'BP+SP','color','k','Fontsize',10,'fontweight','bold','HorizontalAlignment','center');
% text(n2+ngap+n2/2,-0.05,'Removed noise','color','k','Fontsize',10,'fontweight','bold','HorizontalAlignment','center');
% % annotation(gcf,'textarrow',[0.190 0.192],...
% %     [0.65 0.694],'Color','r','TextColor','r','HorizontalAlignment','center',...
% %     'String',{'Horizontal noise'},...
% %     'LineWidth',2,...
% %     'FontSize',10,'fontweight','bold');
% text(-200,-0.1,'(c)','color','k','Fontsize',10,'fontweight','bold','HorizontalAlignment','center');
% 
% figure;
% % subplot(4,2,3);
% das_imagesc(d_bpspsomf2,100,2,x,t);
% ylabel('Time (s)','Fontsize',10,'fontweight','bold');
% xlabel('Channel','Fontsize',10,'fontweight','bold');
% set(gca,'Linewidth',2,'Fontsize',10,'Fontweight','bold');
% text(n2/2,-0.05,'BP+SP+SOMF','color','k','Fontsize',10,'fontweight','bold','HorizontalAlignment','center');
% text(n2+ngap+n2/2,-0.05,'Removed noise','color','k','Fontsize',10,'fontweight','bold','HorizontalAlignment','center');
% % annotation(gcf,'textarrow',[0.190 0.192],...
% %     [0.65 0.694],'Color','r','TextColor','r','HorizontalAlignment','center',...
% %     'String',{'Horizontal noise'},...
% %     'LineWidth',2,...
% %     'FontSize',10,'fontweight','bold');
% text(-200,-0.1,'(c)','color','k','Fontsize',10,'fontweight','bold','HorizontalAlignment','center');
% % annotation(gcf,'rectangle',[0.420921875 0.112752721617418 0.035328125 0.810264385692069],'linewidth',5,'color','[1 0 0]');
% 
% %
% % figure;
% % % subplot(4,2,3);
% % das_imagesc(d_bpsomf2SP,100,2,x,t);
% % ylabel('Time (s)','Fontsize',10,'fontweight','bold');
% % xlabel('Channel','Fontsize',10,'fontweight','bold');
% % set(gca,'Linewidth',2,'Fontsize',10,'Fontweight','bold');
% % text(n2/2,-0.05,'BP+SOMF+SP','color','k','Fontsize',10,'fontweight','bold','HorizontalAlignment','center');
% % text(n2+ngap+n2/2,-0.05,'Removed noise','color','k','Fontsize',10,'fontweight','bold','HorizontalAlignment','center');
% % % annotation(gcf,'textarrow',[0.190 0.192],...
% % %     [0.65 0.694],'Color','r','TextColor','r','HorizontalAlignment','center',...
% % %     'String',{'Horizontal noise'},...
% % %     'LineWidth',2,...
% % %     'FontSize',10,'fontweight','bold');
% % text(-200,-0.1,'(c)','color','k','Fontsize',10,'fontweight','bold','HorizontalAlignment','center');
% % annotation(gcf,'rectangle',[0.420921875 0.112752721617418 0.035328125 0.810264385692069],'linewidth',5,'color','[1 0 0]');
% 
% %
% % subplot(4,2,4);
% 
% % annotation(gcf,'rectangle',[0.420921875 0.112752721617418 0.035328125 0.810264385692069],'linewidth',5,'color','[1 0 0]');
% %
% figure;
% % subplot(4,2,5);
% das_imagesc(d_bpspsomfcuv2,100,2,x,t);
% ylabel('Time (s)','Fontsize',10,'fontweight','bold');
% xlabel('Channel','Fontsize',10,'fontweight','bold');
% set(gca,'Linewidth',2,'Fontsize',10,'Fontweight','bold');
% text(n2/2,-0.05,'BP+SP+SOMF+Curvelet','color','k','Fontsize',10,'fontweight','bold','HorizontalAlignment','center');
% text(n2+ngap+n2/2,-0.05,'Removed noise','color','k','Fontsize',10,'fontweight','bold','HorizontalAlignment','center');
% text(-200,-0.1,'(d)','color','k','Fontsize',10,'fontweight','bold','HorizontalAlignment','center');
% % annotation(gcf,'rectangle',[0.57 0.675 0.334 0.030],'linewidth',2,'color','g');
% 
% % annotation(gcf,'rectangle',[0.420921875 0.112752721617418 0.035328125 0.810264385692069],'linewidth',5,'color','[1 0 0]');
% 
% figure;
% % subplot(4,2,5);
% das_imagesc(d_bpspsomfcuvlow2,100,2,x,t);
% ylabel('Time (s)','Fontsize',10,'fontweight','bold');
% xlabel('Channel','Fontsize',10,'fontweight','bold');
% set(gca,'Linewidth',2,'Fontsize',10,'Fontweight','bold');
% text(n2/2,-0.05,'BP+SP+SOMF+Curvelet+LOW','color','k','Fontsize',10,'fontweight','bold','HorizontalAlignment','center');
% text(n2+ngap+n2/2,-0.05,'Removed noise','color','k','Fontsize',10,'fontweight','bold','HorizontalAlignment','center');
% text(-200,-0.1,'(d)','color','k','Fontsize',10,'fontweight','bold','HorizontalAlignment','center');
% annotation(gcf,'rectangle',[0.57 0.675 0.334 0.030],'linewidth',2,'color','g');


yc_snr(dc,dn)

yc_snr(dc,d_bp)

yc_snr(dc,d_bpsp)

yc_snr(dc,d_bpspsomf)

yc_snr(dc,d_bpspsomfcuv)

yc_snr(dc,d_bpspsomfcuvlow)

 
 
 
 