% backbone script after
% https://cbilltang.wordpress.com/2017/12/09/matlab-demo-code-for-compressed-sensing-measurement-reconstruction-for-mri-image/
%% initialization
 
clear;clc; close all
%----------------load testing data---------
n1=128;
Orig_Image = im2double(imread('/home/pkudela/work/projects/nawa-bekker/ma-shm/data/processed/num/wavefield_dataset2_bottom_out/41_output/111_flat_shell_Vz_41_500x500bottom.png'));

%imshow(Orig_Image)
Orig_Image = imresize(Orig_Image,[n1 n1]);
x = reshape(Orig_Image,n1*n1,1);% reshape into 1D
mx = mean(x);
x = x - mx;
Orig_Image = Orig_Image - mx;
figure; 
imagesc(Orig_Image);colormap jet; axis equal; axis off;
%% CS meaurement and recovery
% ------------CS measurement-------------
n = numel(x);
m = 8000;% Measurement number
perm = round(rand(m,1)*n);
perm(perm==0)=1;
y1 = x(perm); % compressed measurement
x_mask=zeros(n,1);
x_mask(perm)=1;
x2=x.*x_mask;
random_image=reshape(x2,n1,n1); % random measurements
figure;
imagesc( reshape(x2,n1,n1) ); colormap jet;axis equal;axis off;
%-------------- reconstruct with orthogonal matching pursuit -----------
Th=1e-4;% residual threshold 
Psi = dftmtx(n);% Fourier sparse basis
Theta = Psi(perm, :); % random rows of Psi

opts = spgSetParms('optTol',1e-4);
[xSparse,r,g,info] = spg_bpdn(Theta,y1,Th,opts);% reconstruction the sparse signal
Psi_inv = conj(Psi);
xRec = real(Psi_inv*xSparse); % calculate the original signal
recon_image =  reshape(flipud(xRec), n1,n1); % reconstructed image
figure
imagesc(recon_image);colormap jet;axis equal;axis off;

figure;
ax1 = subplot(1,3,1);
imshow(Orig_Image+mx);
title(['Orig ',num2str(n1),'x',num2str(n1)]);
ax2 = subplot(1,3,2);
imshow(random_image+mx);
title(['Rand ',num2str(m)]);
ax3 = subplot(1,3,3);
imshow(recon_image+mx);
title('Reconstr Fourier');

print(['CS_2D_Fourier'],'-dpng', '-r600'); 