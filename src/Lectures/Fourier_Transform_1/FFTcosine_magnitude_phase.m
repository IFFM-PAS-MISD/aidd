clear all; close all;
% Basic example of Fourier transform of sine function
fig_width = 14; fig_height = 6; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Example 1 - cosine - method without fftshift
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = 0.5;                            %amplitude of the cosine wave
fc=10;                              %frequency of the cosine wave [Hz]
phase=30;                         %desired phase shift of the cosine in degrees
fs=32*fc;                           %sampling frequency with oversampling factor 32
dt = 1/fs;                          % sampling interval [s]
N=256;                              % number of points in FFT, also number of sampling points
tstart = 0;                          % initial time point [s]
j=1:N;                                 % sample index
t=tstart+(j-1)*dt;                % time vector, 0.77 seconds duration
phi = phase*pi/180;             %convert phase shift in degrees in radians
u=A*cos(2*pi*fc*t+phi);     %time domain signal with phase shift
% Fourier transform
U = 1/N*fft(u,N);     %N-point complex DFT
% amplitude spectrum
df=fs/N;                            %frequency resolution
sampleIndex = j;                  %non-ordered index for FFT plot
f=(sampleIndex-1)*df;          %x-axis index converted to frequencies
% Results in two-sided frequency plot which is symmetric with respect to frequency f(N/2+1) instead of 0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example 1 - plotting results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; subplot(2,2,1);
plot(t,u,'Color',[0,0,1],'linewidth',1); %plot the signal
xlabel('t [s]','Fontsize',11);
ylabel('u(t)','Fontsize',11);
set(gca,'Fontsize',10,'linewidth',1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% magnitude
subplot(2,2,2);
stem(f(1:N/2+1),2*abs(U(1:N/2+1)),'r','filled','MarkerSize',3); %magnitudes vs frequencies 
% (only half of the spectrum)
xlabel('f [Hz]','Fontsize',11); ylabel('|U(f)|','Fontsize',11);
title('Single-sided spectrum');
xlim([0 30]);
set(gca,'Fontsize',10,'linewidth',1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% real
subplot(2,2,3);
stem(f,real(U),'r','filled','MarkerSize',3); %Real vs frequencies
xlabel('f [Hz]','Fontsize',11); ylabel('Re(U(f))','Fontsize',11);
title('Two-sided frequency plot','Fontsize',11);
%xlim([-30 30]);
set(gca,'Fontsize',10,'linewidth',1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% imag
subplot(2,2,4);
stem(f,imag(U),'r','filled','MarkerSize',3); %Real vs frequencies
xlabel('f [Hz]','Fontsize',11); ylabel('Im(U(f))','Fontsize',11);
title('Two-sided frequency plot','Fontsize',11);
%xlim([-30 30]);
set(gca,'Fontsize',10,'linewidth',1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
fig.PaperPositionMode   = 'auto';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%return;
%% Example 2 - cosine - method with fftshift
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = 0.5;                            %amplitude of the cosine wave
fc=10;                              %frequency of the cosine wave [Hz]
phase=30;                         %desired phase shift of the cosine in degrees
fs=32*fc;                           %sampling frequency with oversampling factor 32
t=0:1/fs:2-1/fs;                  %time vector, 2 seconds duration
phi = phase*pi/180;             %convert phase shift in degrees in radians
u=A*cos(2*pi*fc*t+phi);     %time domain signal with phase shift (640 samples)
% Fourier transform
N=256; %N=2^nextpow2(length(u)); %number of points in FFT
U = 1/N*fftshift(fft(u,N));     %N-point complex DFT
% amplitude spectrum
df=fs/N;                            %frequency resolution
sampleIndex = -N/2:N/2-1;   %ordered index for FFT plot
f=sampleIndex*df;               %x-axis index converted to ordered frequencies
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example 2 - plotting results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; subplot(2,2,1);
plot(t,u,'Color',[0,0,1],'linewidth',1); %plot the signal
xlabel('t [s]','Fontsize',11);
ylabel('u(t)','Fontsize',11);
set(gca,'Fontsize',10,'linewidth',1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% magnitude
subplot(2,2,2);
stem(f,abs(U),'r','filled','MarkerSize',3); %magnitudes vs frequencies
xlabel('f [Hz]','Fontsize',11); ylabel('|U(f)|','Fontsize',11);
xlim([-30 30]);
set(gca,'Fontsize',10,'linewidth',1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% real
subplot(2,2,3);
stem(f,real(U),'r','filled','MarkerSize',3); %Real vs frequencies
xlabel('f [Hz]','Fontsize',11); ylabel('Re(U(f))','Fontsize',11);
xlim([-30 30]);
set(gca,'Fontsize',10,'linewidth',1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% imag
subplot(2,2,4);
stem(f,imag(U),'r','filled','MarkerSize',3); %Real vs frequencies
xlabel('f [Hz]','Fontsize',11); ylabel('Im(U(f))','Fontsize',11);
xlim([-30 30]);
set(gca,'Fontsize',10,'linewidth',1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig = gcf;set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
fig.PaperPositionMode   = 'auto';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Example 3 - Extract phase of frequency components (phase spectrum)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% atan function computes the inverse tangent over two quadrants only, 
% i.e, it will return values only in the [-pi/2 , pi/2]  interval. 
% atan2 is the four quadrant arctangent i.e, it will return values in the [-pi , pi]  interval. 
phase=atan2(imag(U),real(U))*180/pi; %phase information from definition
figure;
plot(f,phase); %phase vs frequencies
xlabel('f [Hz]','Fontsize',11); ylabel({'\phi'},'Fontsize',11);
xlim([-160,160]);
fig = gcf;set(fig,'Color','w');
% The phase spectrum is completely noisy. Unexpected !!!. 
% The phase spectrum is noisy due to fact that the inverse tangents are computed from the ratio 
% of imaginary part to real part of the FFT result. Even a small floating rounding off error will amplify the result 
% and manifest incorrectly as useful phase information.

% To understand, print the first few samples from the FFT result and observe that they are not absolute zeros 
% (they are very small numbers in the order 10^{-16}. Computing inverse tangent will result in incorrect results.
% real(U(1:4)) >> 1.0e-16 * [-0.7286   -0.3637   -0.4809   -0.3602]
% imag(U(1:4)) >> 1.0e-16 *[0   -0.2501   -0.1579   -0.5579]
% U(1:4) >> 1.0e-16 *-0.7286 + 0.0000i  -0.3637 - 0.2501i  -0.4809 - 0.1579i-0.3602 - 0.5579i

% The solution is to define a tolerance threshold and ignore all the computed phase values that are below the threshold.

U2=U;%store the FFT results in another array
%detect noise (very small numbers (eps)) and ignore them
threshold = max(abs(U))/10000; %tolerance threshold
U2(abs(U)<threshold) = 0; %maskout values that are below the threshold
phase=atan2(imag(U2),real(U2))*180/pi; %phase information
figure;
stem(f,phase,'r','filled','MarkerSize',3); %phase vs frequencies
xlabel('f [Hz]','Fontsize',11); ylabel({'\phi'},'Fontsize',11);
xlim([-160,160]);
fig = gcf;set(fig,'Color','w');
% The phase spectrum has correctly registered the 30deg phase shift at the frequency f=10 Hz. 
% The phase spectrum is anti-symmetric (\phi=-30deg  at f=-10 Hz ), which is expected for real-valued signals.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Example 4 - Reconstruct the time domain signal from the frequency domain samples
% Inverse Fourier transform
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u_recon = N*ifft(ifftshift(U),N); %reconstructed signal
t =[0:1:length(u_recon)-1]/fs; %recompute time index 
figure; 
plot(t,u_recon,'Color',[0,0,1],'linewidth',1);%reconstructed signal
xlabel('t [s]','Fontsize',11);
ylabel('u(t)','Fontsize',11);
title('Reconstructed signal');
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
fig.PaperPositionMode   = 'auto';
% The reconstructed signal has preserved the same initial phase shift and the frequency of the original signal. 
% Note: The length of the reconstructed signal is only 256 sample long (~ 0.8 seconds duration).
% This is because the size of FFT is considered as N=256. 
% Better to use N=2^nextpow2(length(u)); It assures tha all samples of signal u are utilized.
% If N> length(u), transformed signal is zero-padded