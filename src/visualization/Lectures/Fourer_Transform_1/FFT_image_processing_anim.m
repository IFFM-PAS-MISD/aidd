% Example of image processing by using 2D FFT
clear all; close all;
% load projectroot path
load project_paths projectroot src_path;
output_path = fullfile(projectroot,'reports','beamer_presentations','Lectures','Fourier_Transform_1','figs',filesep);
fig_width = 6; fig_height = 6; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rgbImage=imread('lena_std.tif');
I = rgb2gray(rgbImage); % 512x512x1 (uint8 in range 0:255)
% figure;
% imshow(I);
figfilename = 'Lena2';
imwrite(I,[output_path,figfilename,'.png']);
% 0 - black
% 255 - white
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% wavenumber domain representation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Nx,Ny] = size(I);
I_hat = 2/(Nx*Ny) * fftshift(fft2(I,Nx,Ny));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Example 2 - blur - low-pass
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c=2;
for R=95:-8:5
    c=c+1;
    sigma=R;
    [M_B] = LowPassMaskButterworth(I_hat,R,1); % lower radius - higher blur
    I_recon_B = Nx*Ny/2*ifft2(ifftshift(M_B.*I_hat));
    A=real(I_recon_B);
    A=A/max(max(A));
    image8Bit = uint8(255 *A);
    figfilename = ['Lena',num2str(c)];
    imwrite(image8Bit,[output_path,figfilename,'.png']);
end
for R=5:8:95
    c=c+1;
    sigma=R;
    [M_G] = LowPassMaskGauss(I_hat,sigma); % Gaussian blur
    I_recon_G = Nx*Ny/2*ifft2(ifftshift(M_G.*I_hat));
    A=real(I_recon_G);
    A=A/max(max(A));
    image8Bit = uint8(255 *A);
    figfilename = ['Lena',num2str(c)];
    imwrite(image8Bit,[output_path,figfilename,'.png']);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Example 3 - edges - high-pass
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for R=5:5:50
    c=c+1;
    sigma=R;
    [M_h] = HighPassMask(I_hat,R);
    I_recon_h = Nx*Ny*ifft2(ifftshift(M_h.*I_hat));
    A=real(I_recon_h);
    A=0.2+A/max(max(A));
    image8Bit = uint8(255 *A);
    figfilename = ['Lena',num2str(c)];
    imwrite(image8Bit,[output_path,figfilename,'.png']);
end
for R=50:-5:5
    c=c+1;
    sigma=R;
    [M_h] = HighPassMask(I_hat,R);
    I_recon_h = Nx*Ny*ifft2(ifftshift(M_h.*I_hat));
    A=real(I_recon_h);
    %A=A/max(max(A));
    image8Bit = uint8(255 *A);
    figfilename = ['Lena',num2str(c)];
    imwrite(image8Bit,[output_path,figfilename,'.png']);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
