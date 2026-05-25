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
is_real=1;           % Type of the transform(0: complex-valued curvelets,1: real-valued curvelets)
finest=2;            % Chooses one of two possibilities for the coefficients at the finest level(1: curvelets,2: wavelets)
% alpha=1.2;         % ������׼���alpha����ֵ��1.2���ҽ�Ϊ���룩
alpha=2.5;
niter=10;

% dc=levents(200);dc=yc_scale(dc);
% load Curveddata
% dc=data;
% 
% 
% [n1,n2]=size(dc);
% 
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

% dn=dc+err_n+ran_n;
% 
% save dncurved dn

% load dncurved 

eq=zeros(2000,960);
[n1,n2]=size(eq);

for ii=3
    if ~ismember(ii,[14,16,17,27,47,52])
        strcat('mat_raw/eq-',num2str(ii),'.mat')
        load(strcat('mat_raw/eq-',num2str(ii),'.mat'));
    end
    d1=d1;
    eq=d1;
end

dt=0.004;
t=[0:n1-1]*dt; x=[1:n2];

% figure;imagesc([dc,dn]);caxis([-0.5,0.5]);colormap(seis);

F=ones(n1,n2);                                  % ones(n)����n*n��1����Ƶ����
X=fftshift(ifft2(F))*sqrt(prod(size(F)));  
size(X)
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

Cdn=fdct_wrapping(d1,is_real,finest);     %������任�õ���ʵ��,�����ΪС���任
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
 d0=real(ifdct_wrapping(Ct,is_real,n1,n2));
%%
d2=das_bandpass(d0,0.0005,0,200,6,6,0,0);%
size(d1);
%
% dipc=str_dip2d(data);
dipn=str_dip2d(d2,2,10,2,0.01, 1, 0.000001,[50,50,1],1);

type_mf=1;

ifsmooth=0;
ns=3;

[~,ds]=pwsmooth_lop_mf(0,0,dipn,[],n1,n2,ns,2,0.01,n1*n2,n1*n2,type_mf,ifsmooth,d2,[]);

Z=reshape(ds,n1,n2);
    
%     Cdn=fdct_wrapping(Z,is_real,finest);     %������任�õ���ʵ��,�����ΪС���任
    
%     Smax=length(Cdn);
% %     Sigma=sigma(i+1);     %��ȡ������׼�ѡȡ���߶�
%     
%     Ct=Cdn;
%  for s=2:length(Cdn)
%         thresh=Sigma+Sigma*s;
%         for w=1:length(Cdn{s})
%             Ct{s}{w}=Cdn{s}{w}.*(abs(Cdn{s}{w})>thresh*E{s}{w});
%         end
%  end
% %     dipn=str_dip2d(d5,5,20,2,0.01,1,0.000001,[20,5,1],1);
% %   d11=fkt1(Z,'ps',t1);
%    
%     d5=real(ifdct_wrapping(Ct,is_real,n1,n2));
    
    t1=50;
    beta=0.9;
    d5=fkt1(Z,'ps',t1,beta);

%     d1=fkt1(d5,'ps',t1);
     
% d1=fxdecon(d5,1,120,0.003,10,25,1);
noi1=d1-d5;
% prepare paramters for ortho
rect = zeros(3, 1);
rect(1) = 5;
rect(2) = 5;
rect(3) = 1;
eps=0.00000000000001;%too strong, do not know why
eps=0;
niter=20;
verb=1;

[d6,noi2,low]=yc_localortho(d5,noi1,rect,niter,eps,verb);
 
figure(1);das_imagesc([d1,d0,d1-d0]);

% figure(2);das_imagesc([dn,ds,dn-ds]);

figure(2);das_imagesc([d1,d5,d1-d5]);

figure(3);das_imagesc([d1,d6,d1-d6]);

% yc_snr(dc,dn)
% 
% yc_snr(dc,d0)
% 
% % yc_snr(dc,Z)
% 
% yc_snr(dc,d5)
% 
% yc_snr(dc,d6)
