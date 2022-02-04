% backbone script after
% https://cbilltang.wordpress.com/2017/12/09/matlab-demo-code-for-compressed-sensing-measurement-reconstruction-for-mri-image/
%% initialization
 
clear;clc; close all
mask_type = 2;
% mask_type = 1; % random mask
% mask_type = 2; % jitter mask
%----------------load testing data---------
% n1=65;
% n1=512;% out of memory
%n1=256;
n1=128; % number of points in the targeted grid (n1xn1)
%load('/pkudela_odroid_laser/aidd/data/raw/exp/L3_S4_B/Compressed/65x65p_50kHz_5HC_14Vpp_x10.mat');
load('/pkudela_odroid_laser/aidd/data/raw/exp/L3_S4_B/Compressed/389286p_na_512x512p.mat');
frame=110;
Orig_Image = squeeze(Data(:,:,frame))/max(max(squeeze(Data(:,:,frame))));% 
%Orig_Image = squeeze(Data3D(:,:,frame))/max(max(squeeze(Data3D(:,:,frame))));% 
%imshow(Orig_Image)
clear Data;

% downsampling
N=512;
[X,Y]=meshgrid(linspace(0, WL(1), n1),linspace(0, WL(2), n1));
[X1,Y1]=meshgrid(linspace(0, WL(1), N),linspace(0, WL(2), N));
Orig_Image_n1 = griddata(X1(2:end-1,2:end-1),Y1(2:end-1,2:end-1),Orig_Image(2:end-1,2:end-1),X,Y,'cubic');
Orig_Image_n1(isnan(Orig_Image_n1))=0;
% subsampling
%Orig_Image = Orig_Image(1:2:end,1:2:end);
%Orig_Image_n1 = Orig_Image(2:4:end,2:4:end);
% 2D FFT
% Orig_Image_n1=fftshift(fft2(Orig_Image_n1));
x = reshape(Orig_Image_n1,n1*n1,1);% reshape into 1D
mx = mean(x);
x = x - mx;
Orig_Image_n1 = Orig_Image_n1 - mx;

figure(1); 
imagesc(Orig_Image_n1);
%colormap jet; 
colormap default;
axis equal; axis off;
%imagesc(abs(Orig_Image_n1));colormap jet; axis equal; axis off;
title(['Downsampled to ',num2str(n1),'x',num2str(n1)]);
%% CS meaurement and recovery
% ------------CS measurement-------------
n = numel(x);
m = 4096;%2025;%3400;% Measurement number
m = 6000;
% random mask
perm = randperm(n,m)';
x_mask=zeros(n,1);
x_mask(perm)=1;
x_=x.*x_mask;
random_image=reshape(x_,n1,n1); % random measurements
% my jitter mask for 128x128 grid
[ind_jitter,x_jitter_mask]=my_jitter_mask(m,n1);

figure(2);
%imagesc( reshape(x2,n1,n1) ); colormap jet;axis equal;axis off;
imagesc( reshape(x_mask,n1,n1) ); axis equal;axis off; colormap gray;
title('random mask')

figure(3);
imagesc( reshape(x_jitter_mask,n1,n1) ); axis equal;axis off; colormap gray;
title('jitter mask');
drawnow;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch mask_type
    case 1
        y1 = x(perm); % compressed measurement (random mask)
    case 2
        y1 = x(ind_jitter); % compressed measurement (jitter mask)
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%-------------- reconstruct with orthogonal matching pursuit -----------
Th=1e-4;% residual threshold (sigma in spgl1 description)
% constant related to the noise level in the measurements
%sigma = sqrt(sum(x.^2))/N;
sigma = var(x) * 0.5;
Th = sigma;
Psi = dftmtx(n);% Fourier sparse basis


switch mask_type
    case 1
        Theta = Psi(perm, :); % random rows of Psi (random mask)
    case 2
        Theta = Psi(ind_jitter, :); % random rows of Psi (jitter mask)
end

opts = spgSetParms('optTol',1e-4);
% basis-pursuit denoise
[xSparse,r,g,info] = spg_bpdn(Theta,y1,Th,opts);% reconstruction the sparse signal
Psi_inv = conj(Psi);
xRec = real(Psi_inv*xSparse); % calculate the original signal

recon_image =  reshape(flipud(xRec), n1,n1); % reconstructed image
figure(4);
imagesc(recon_image);
%colormap jet;
colormap default;
axis equal;axis off;
title(['Reconstructed image ', num2str(n1),'x',num2str(n1)]);
figure(5);
ax1 = subplot(1,3,1);
imshow(Orig_Image_n1+mx+0.5);
title(['Orig ',num2str(n1),'x',num2str(n1)]);
ax2 = subplot(1,3,2);
imshow(random_image+mx+0.5);
title(['Rand ',num2str(m)]);
ax3 = subplot(1,3,3);
imshow(recon_image+mx+0.5);
title('Reconstr Fourier');
drawnow;
%print(['CS_2D_Fourier_exp2'],'-dpng', '-r600'); 
% upscale by bicubic interpolation
N=512;
[X,Y]=meshgrid(linspace(0, WL(1), n1),linspace(0, WL(2), n1));
[X1,Y1]=meshgrid(linspace(0, WL(1), N),linspace(0, WL(2), N));
upscaled_image = griddata(X,Y,squeeze(recon_image),X1,Y1,'cubic');
figure(6);
imshow(upscaled_image+mx+0.5);
title({['recon+ bicubic upsc. to ',num2str(N),'x',num2str(N) ]});
drawnow;
% figure
% surf(upscaled_image); shading interp; view(2); axis square; colormap jet;
% title('upscaled');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% quality metrics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% peak signal-to-noise ratio
[peaksnr] = psnr(upscaled_image, Orig_Image); 
fprintf('The Peak-SNR value is %0.4f\n', peaksnr);


% Structural Similarity Index 
[ssimval, ssimmap] = ssim(upscaled_image,Orig_Image); 
fprintf('The ssim value is %0.4f.\n',ssimval);
pause(2); 

figure(7);
imshow(ssimmap,[]);
colorbar;
title('ssim Index Map');
drawnow;
% figure, imshow(abs(ssimmap),[]);
% colorbar
% title(sprintf('ssim Index Map - Mean ssim Value is %0.4f',ssimval));
pause(2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FFT play
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A1=fftshift(fft2(Orig_Image));
figure(8);
surf(abs(A1));shading interp; view(2);axis square;
xlim([0 N]);ylim([0 N]);
title('original-wavenumber');
drawnow;
A2=fftshift(fft2(upscaled_image));
figure(9);
surf(abs(A2));shading interp; view(2);axis square;
title({['recon+bicubic upsc. to ',num2str(N),'x',num2str(N) ]});
xlim([0 N]);ylim([0 N]);
drawnow;
R=34;
[Mask] = LowPassMaskButterworth(A2,R,5);

a2=ifft2(ifftshift(A2.*Mask));
figure(10);
surf(real(a2));shading interp; view(2);axis square;
xlim([0 N]); ylim([0 N]);
title({['recon+bicubic upsc.',num2str(N),'x',num2str(N) ' +filt.']});
drawnow;
figure(11);
imshow(real(a2)+mx+0.5);
title({['recon+bicubic upsc.',num2str(N),'x',num2str(N) ' +filt.']});
drawnow;
% bicubic upscaling only
[X,Y]=meshgrid(linspace(0, WL(1), n1),linspace(0, WL(2), n1));
[X1,Y1]=meshgrid(linspace(0, WL(1), N),linspace(0, WL(2), N));
X_=reshape(X,[n1*n1,1]);
Y_=reshape(Y,[n1*n1,1]);

switch mask_type
    case 1
        X_=X_(perm); % compressed measurement (random mask)
        Y_=Y_(perm);
    case 2
        X_=X_(ind_jitter); 
        Y_=Y_(ind_jitter);
end
bicubic_upscaled_image = griddata(X_,Y_,y1,X1,Y1,'cubic');
F = scatteredInterpolant(X_,Y_,y1,'linear');
linear_upscaled_image = F(X1,Y1);
figure(12);
imshow(linear_upscaled_image+mx+0.5);
title(['linear only ',num2str(N),'x',num2str(N) ]);
drawnow;
%bicubic_upscaled_image = griddata(X_,Y_,y1,X,Y,'cubic');
%bicubic_upscaled_image = interp2(X_,Y_,y1,X1,Y1,'cubic',0);

% figure(12);
% imshow(bicubic_upscaled_image+mx+0.5);
% title(['bicubic only ',num2str(N),'x',num2str(N) ]);
% drawnow;
pause(2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% inpaint method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% John D'Errico (2022). inpaint_nans (https://www.mathworks.com/matlabcentral/fileexchange/4551-inpaint_nans), 
% MATLAB Central File Exchange. Retrieved February 2, 2022.
zero_ind = find(random_image==0);
znan = random_image;
znan(zero_ind) = NaN;
% %% In-paint Over NaNs
z = inpaint_nans(znan,3);

figure(13); 
imagesc(z);
%colormap jet;
colormap default;
axis equal; axis off;
title(['Inpaint ',num2str(n1),'x',num2str(n1)]);
drawnow;

inpaint_bicubic_upscaled_image = griddata(X,Y,z,X1,Y1,'cubic');

figure(14); 
imagesc(inpaint_bicubic_upscaled_image);
%colormap jet; 
colormap default;
axis equal; axis off;
title(['Inpaint + bicubic ',num2str(N),'x',num2str(N)]);
drawnow;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compression in wavenumber domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
B2=fftshift(fft2(Orig_Image_n1));
figure(15);
surf(abs(B2));shading interp; view(2);axis square;
title({['Downsampled to ',num2str(n1),'x',num2str(n1)]});
xlim([0 n1]);ylim([0 n1]);
drawnow;

R=29;
[Mask] = LowPassMaskButterworth(B2,R,7);
Mask(abs(Mask)<5e-2)=0;
nnz(Mask)
CR=nnz(Mask)/(n1*n1)*100
B3=B2.*Mask;

b3=ifft2(ifftshift(flipud(B3)));
figure(16);
surf(real(b3));shading interp; view(2);axis square;
colormap jet;
colormap default;
title(['Compressed with CR ',num2str(CR),'%']);
xlim([0 n1]);ylim([0 n1]);
drawnow;