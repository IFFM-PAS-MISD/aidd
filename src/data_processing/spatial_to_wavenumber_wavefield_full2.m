function [KXKYF,kx_vec,ky_vec,f_vec] = spatial_to_wavenumber_wavefield_full2(Data,Length,Width,time)
% spatial_to_wavenumber_wavefield_full   3D FFT of full wavefield Data
%    returns all quadrants in wavenumber-frequency domain 
% 
% Syntax: [KXKYF_,kx_vec,ky_vec,f_vec] = spatial_to_wavenumber_wavefield_full(Data,Length,Width,time) 
% 
% Inputs: 
%    Data - Full wavefiled data, double, dimensions [nY,nX,nT], 
%    nY,nX - spatial dimensions, nT - time vector length, Units: m/s 
%    Length - length of wavefield area, double, Units: m 
%    Width - width of wavefield area, double, Units: m 
%    time - time vector, double, dimensions [1,nT], Units: s
% 
% Outputs: 
%    KXKYF - wavenumber-frequency domain wavefield, complex, dimensions [1024,1024,1024], Units: - 
%    kx_vec - vector of wavenumbers in x direction, double, dimensions [1,1024], Units: rad/m 
%    ky_vec - vector of wavenumbers in y direction, double, dimensions [1,1024], Units: rad/m 
%    f_vec - vector of frequency components (positive only), double, dimensions [1,512], Units: Hz
% 
% Example: 
%    [KXKYF,kx_vec,ky_vec,f_vec] = spatial_to_wavenumber_wavefield_full(Data,Length,Width,time) 
% 
% Other m-files required: none 
% Subfunctions: none 
% MAT-files required: none 
% See also:  
% 

% Author: Pawel Kudela, D.Sc., Ph.D., Eng. 
% Institute of Fluid Flow Machinery Polish Academy of Sciences 
% Mechanics of Intelligent Structures Department 
% email address: pk@imp.gda.pl 
% Website: https://www.imp.gda.pl/en/research-centres/o4/o4z1/people/ 

%---------------------- BEGIN CODE---------------------- 

% original Data size
[nY, nX, nT] = size(Data);
% padded size
nsY = 1024;
nsX = 1024;
nsT = 1024;
padded_size = [nsY nsX nsT];

%% axis vectors
Fs =  1/(time(3)-time(2));                % sampling frequency
f_vec = Fs/2*linspace(0,1,nsT/2);         % frequency vector

dx = Length/(nX-1);
dkx = 1/(nX*dx);
kxmax = 1/(2*dx)-dkx/2;
kx_vec = 2*pi*linspace(-kxmax,kxmax,nsX);    % rad/m

dy = Width/(nY-1);
dky = 1/(nY*dy);
kymax = 1/(2*dy)-dky/2;
ky_vec = 2*pi*linspace(-kymax,kymax,nsY);    % rad/m

%% 3D FFT
KXKYF = fftshift(fftn(Data,padded_size));

%---------------------- END OF CODE---------------------- 

% ================ [spatial_to_wavenumber_wavefield_full.m] ================  
