% Example of image processing by using 2D FFT
clear all; close all;
fig_width=14; fig_height=7;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rgbImage=imread('lena_std.tif');
% colors manipulations
size(rgbImage) % 512x512x3 (unsigned integers 8bit in range 0:255)
% convert RGB image to double
rgbImage2 = im2double(rgbImage);
size(rgbImage2) % 512x512x3 (doubles in range 0:1)
% Extract the individual red, green, and blue color channels.
redChannel = rgbImage(:, :, 1);
greenChannel = rgbImage(:, :, 2);
blueChannel = rgbImage(:, :, 3);
% convert to grayscale
%I = rgb2gray(rgbImage2); % 512x512x1 (doubles in range 0:1) 
I = rgb2gray(rgbImage); % 512x512x1 (uint8 in range 0:255)
figure;
subplot(1,2,1); imshow(rgbImage);
subplot(1,2,2); imshow(I);
fig = gcf;set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[1 19 fig_width fig_height]); 
fig.PaperPositionMode   = 'auto';
% 0 - black
% 255 - white
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Example 1 - wavenumber domain representation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Nx,Ny] = size(I);
I_hat = 2/(Nx*Ny) * fftshift(fft2(I,Nx,Ny));
a=2;b=5000;
I_hat_log = a*log10((1+b*abs(I_hat))/(1+max(max(abs(I_hat)))));
%I_hat_log = log10((1+b*(I_hat))/(1+max(max((I_hat)))));
figure;
subplot(1,2,1);imshow(abs((I_hat)));title('linear scale');
subplot(1,2,2);imshow(abs((I_hat_log)));title('log scale');
fig = gcf;set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[1 11 fig_width fig_height]); 
%figure;imshow(real((I_hat_log)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Example 2 - blur - low-pass
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
R=15;
%R=50;
sigma=R;
[M_l] = LowPassMask(I_hat,R);
[M_B] = LowPassMaskButterworth(I_hat,R,1); % lower radius - higher blur
[M_G] = LowPassMaskGauss(I_hat,sigma); % Gaussian blur
I_recon_l = Nx*Ny/2*ifft2(ifftshift(M_l.*I_hat));
I_recon_B = Nx*Ny/2*ifft2(ifftshift(M_B.*I_hat));
figure;
subplot(3,2,1);surf(M_l);shading interp;
subplot(3,2,2);surf(M_B);shading interp;
subplot(3,2,3);imshow(abs((M_l.*I_hat_log)));title('Low-pass');
subplot(3,2,4);imshow(abs((M_B.*I_hat_log)));title('Butterworth');
subplot(3,2,5);imshow(real(I_recon_l),[]);
subplot(3,2,6);imshow(real(I_recon_B),[]);
fig = gcf;set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[24 2 24 24]); 
fig.PaperPositionMode   = 'auto';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Example 3 - edges - high-pass
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
R=15;
[M_h] = HighPassMask(I_hat,R);
I_recon_h = Nx*Ny*ifft2(ifftshift(M_h.*I_hat));
figure;subplot(1,3,1);imshow(abs((M_h.*I_hat_log)));title('High-pass');
subplot(1,3,2);imshow(real(I_recon_h),[]); % use full color scale
subplot(1,3,3);imshow(real(I_recon_h)); % figure;imshow(im2double(real(I_recon_h)),[0 1]);
fig = gcf;set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[1 2 20 fig_height]); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
