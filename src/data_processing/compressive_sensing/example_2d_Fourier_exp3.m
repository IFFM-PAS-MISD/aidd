% backbone script after
% https://cbilltang.wordpress.com/2017/12/09/matlab-demo-code-for-compressed-sensing-measurement-reconstruction-for-mri-image/
%% initialization
 
clear;clc; close all
%----------------load testing data---------
% n1=65;
% n1=512;% out of memory
%n1=256;
n1=128;
%load('/pkudela_odroid_laser/aidd/data/raw/exp/L3_S4_B/Compressed/65x65p_50kHz_5HC_14Vpp_x10.mat');
load('/pkudela_odroid_laser/aidd/data/raw/exp/L3_S4_B/Compressed/389286p_na_512x512p.mat');
frame=110;
Orig_Image = squeeze(Data(:,:,frame))/max(max(squeeze(Data(:,:,frame))))/2;% 
%Orig_Image = squeeze(Data3D(:,:,frame))/max(max(squeeze(Data3D(:,:,frame))))/2;% 
%imshow(Orig_Image)
clear Data;
%Orig_Image = Orig_Image(1:2:end,1:2:end);
Orig_Image = Orig_Image(1:4:end,1:4:end);
x = reshape(Orig_Image,n1*n1,1);% reshape into 1D
mx = mean(x);
 x = x - mx;
 Orig_Image = Orig_Image - mx;
figure; 
imagesc(Orig_Image);colormap jet; axis equal; axis off;
%% CS meaurement and recovery
% ------------CS measurement-------------
n = numel(x);
m = 4096;%2025;%3400;% Measurement number
%m=16000;
%perm = round(rand(m,1)*n); perm(perm==0)=1;
perm = randperm(n,m)';
y1 = x(perm); % compressed measurement
x_mask=zeros(n,1);
x_mask(perm)=1;
x2=x.*x_mask;
random_image=reshape(x2,n1,n1); % random measurements
figure;
%imagesc( reshape(x2,n1,n1) ); colormap jet;axis equal;axis off;
imagesc( reshape(x_mask,n1,n1) ); axis equal;axis off; colormap gray;
%-------------- reconstruct with orthogonal matching pursuit -----------
Th=1e-3;% residual threshold (sigma in spgl1 description)
% constant related to the noise level in the measurements
%sigma = sqrt(sum(x.^2))/N;
sigma = var(x) * 0.5;
Th = sigma;
Psi = dftmtx(n);% Fourier sparse basis
Theta = Psi(perm, :); % random rows of Psi
opts = spgSetParms('optTol',1e-4);
% basis-pursuit denoise
[xSparse,r,g,info] = spg_bpdn(Theta,y1,Th,opts);% reconstruction the sparse signal
Psi_inv = conj(Psi);
xRec = real(Psi_inv*xSparse); % calculate the original signal
recon_image =  reshape(flipud(xRec), n1,n1); % reconstructed image
figure
imagesc(recon_image);colormap jet;axis equal;axis off;

figure;
ax1 = subplot(1,3,1);
imshow(Orig_Image+mx+0.5);
title(['Orig ',num2str(n1),'x',num2str(n1)]);
ax2 = subplot(1,3,2);
imshow(random_image+mx+0.5);
title(['Rand ',num2str(m)]);
ax3 = subplot(1,3,3);
imshow(recon_image+mx+0.5);
title('Reconstr Fourier');
%print(['CS_2D_Fourier_exp2'],'-dpng', '-r600'); 
% upscale by bicubic interpolation
N=512;
[X,Y]=meshgrid(linspace(0, WL(1), n1),linspace(0, WL(2), n1));
[X1,Y1]=meshgrid(linspace(0, WL(1), N),linspace(0, WL(2), N));
upscaled_image = griddata(X,Y,squeeze(recon_image),X1,Y1,'cubic');
figure;
imshow(upscaled_image+mx+0.5);
% figure
% surf(upscaled_image); shading interp; view(2); axis square; colormap jet;
% title('upscaled');
