% backbone script after
% https://cbilltang.wordpress.com/2017/12/09/matlab-demo-code-for-compressed-sensing-measurement-reconstruction-for-mri-image/
%% initialization
 
clear;clc; close all
%----------------load testing data---------
n1=129;% (65-1) + 65
%Orig_Image = im2double(imread('/home/pkudela/work/projects/nawa-bekker/ma-shm/data/processed/num/wavefield_dataset2_bottom_out/41_output/111_flat_shell_Vz_41_500x500bottom.png'));
%load('/pkudela_odroid_laser/aidd/data/raw/exp/L3_S4_B/Compressed/7508p_compressive_50kHz_5HC_14Vpp_x10.mat');
load('/pkudela_odroid_laser/aidd/data/raw/exp/L3_S4_B/Compressed/65x65p_50kHz_5HC_14Vpp_x10.mat');
%load('/pkudela_odroid_laser/aidd/data/raw/exp/L3_S4_B/Compressed/4154p_compressive_50kHz_5HC_14Vpp_x10.mat')

frame=80;
figure
surf(squeeze(Data(:,:,frame))); shading interp; view(2); axis square; colormap jet;
title('measured');
al=0.8;
caxis([-al*max(max(squeeze(Data(:,:,frame)))) al*max(max(squeeze(Data(:,:,frame)))) ]);
[X,Y]=meshgrid(linspace(0, WL(1), 65),linspace(0, WL(2), 65));
[X1,Y1]=meshgrid(linspace(0, WL(1), n1),linspace(0, WL(2), n1));
Vq = griddata(X,Y,squeeze(Data(:,:,frame)),X1,Y1,'cubic');
figure
surf(Vq); shading interp; view(2); axis square; colormap jet;
title('interpolated');


figure;
plot(X,Y,'bx'); axis square;
hold on;
plot(X1,Y1,'r.');
% return;
%imshow(Orig_Image)
%Orig_Image = imresize(Orig_Image,[n1 n1]);
y1 = reshape(squeeze(Data(:,:,frame)),[],1);% reshape into 1D

my = mean(y1);
y1 = y1 - my;
y1=y1/max(y1);
% Orig_Image = Orig_Image - mx;
% figure; 
% imagesc(Orig_Image);colormap jet; axis equal; axis off;
%% CS meaurement and recovery
% ------------CS measurement-------------
n = n1*n1;
m = length(y1);% Measurement number

% perm = round(rand(m,1)*n);
% perm(perm==0)=1;
perm=[];
for k=1:2:n1
    perm=[perm,(k-1)*n1+(1:2:n1)];
end
perm=perm';
%x=reshape(Vq,n1*n1,[]);
% rand_pos = randperm(length(perm));
% perm=perm(rand_pos);
% y1 = x(perm); % compressed measurement
% x_mask=zeros(n,1);
% x_mask(perm)=1;
% x2=x.*x_mask;
% random_image=reshape(x2,n1,n1); % random measurements
% figure;
% imagesc( reshape(x2,n1,n1) ); colormap jet;axis equal;axis off;
%-------------- reconstruct with orthogonal matching pursuit -----------
Th=1e-4;% residual threshold 
Psi = dftmtx(n);% Fourier sparse basis
Theta = Psi(perm, :); % random rows of Psi
%X = Psi*x;              % FFT of x(t)

opts = spgSetParms('optTol',1e-4);
[xSparse,r,g,info] = spg_bpdn(Theta,y1,Th,opts);% reconstruction the sparse signal
Psi_inv = conj(Psi);
xRec = real(Psi_inv*xSparse); % calculate the original signal
recon_image =  reshape(flipud(xRec), n1,n1); % reconstructed image
figure
imagesc(recon_image);colormap jet;axis equal;
title('reconstructed');
% y2=reshape(xRec(perm),65,65);
% figure
% imagesc(y2);colormap jet;axis equal;axis off;
% figure;
% ax1 = subplot(1,3,1);
% imshow(Orig_Image+mx);
% title(['Orig ',num2str(n1),'x',num2str(n1)]);
% ax2 = subplot(1,3,2);
% imshow(random_image+mx);
% title(['Rand ',num2str(m)]);
% ax3 = subplot(1,3,3);
% imshow(recon_image+mx);
% title('Reconstr Fourier');
% 
% print(['CS_2D_Fourier'],'-dpng', '-r600'); 