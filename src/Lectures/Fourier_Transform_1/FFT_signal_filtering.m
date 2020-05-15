% signal filtering example by using FFT

clear all; close all;

%load signal;
%h5create('signal.h5','/dataset',[13108 1]);
%h5write('signal.h5','/dataset',s);
fig_width = 16; fig_height = 12; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%h5disp('signal.h5');
u = h5read('signal.h5','/dataset');
fs = 10e6;                   % sampling frequency
dt = 1/fs;                    % time step
t=0:dt:dt*(size(u,1)-1); % time vector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute Fourier transform
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=2^nextpow2(length(u)); %number of points in FFT
U = 1/N*fftshift(fft(u,N));     %N-point complex DFT
% amplitude spectrum
df=fs/N;                            %frequency resolution
sampleIndex = -N/2:N/2-1;   %ordered index for FFT plot
f=sampleIndex*df;               %x-axis index converted to ordered frequencies
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initial plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
% time domain signal
subplot(2,2,1);
plot(t*1e3,u,'Color',[0,0,1],'linewidth',1);
xlim([0,1]);
xlabel('t [ms]','Fontsize',11);
ylabel('u(t)','Fontsize',11);
set(gca,'Fontsize',10,'linewidth',1);
% time domain signal - zoom
subplot(2,2,2);
plot(t*1e3,u,'Color',[0,0,1],'linewidth',1);
xlim([0.1,0.2]);
xlabel('t [ms]','Fontsize',11);
ylabel('u(t)','Fontsize',11);
set(gca,'Fontsize',10,'linewidth',1);
title('u(t) zoom');
% magnitude
subplot(2,2,3);
stem(f/1e3,2*abs(U),'r','filled','MarkerSize',3); %magnitudes vs frequencies
xlabel('f [kHz]','Fontsize',11); ylabel('|U(f)|','Fontsize',11);
xlim([-100 100]);
ylim([0 0.015]);
set(gca,'Fontsize',10,'linewidth',1);
% magnitude - zoom around 180-350 kHz
subplot(2,2,4);
stem(f/1e3,2*abs(U),'r','filled','MarkerSize',3); %magnitudes vs frequencies
xlabel('f [kHz]','Fontsize',11); ylabel('|U(f)|','Fontsize',11);
title('U(f) zoom');
xlim([180 350]);
set(gca,'Fontsize',10,'linewidth',1);
%%%%%%%%%%%%%%%
fig = gcf;set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
fig.PaperPositionMode   = 'auto';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%return;
%% signal filtering (basic low-pass filter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fc_low_pass = 200e3; % cut-off frequency for the low-pass filter (200 kHz)
[fc_low_pass_pos,Istart_pos]=min(abs(f-fc_low_pass));% find index of cut-off frequency (fc)
U(Istart_pos:end,1) = 0;% reject frequencies above cut-off frequency
% do the same for the negative side of the spectrum
[fc_low_pass_neg,Istart_neg]=min(abs(-f-fc_low_pass));% find index of negative cut-off frequency (-fc)
U(1:Istart_neg,1) = 0;
%U(1:N/2,1) = 0;% alternative approach - reject negative frequencies
u_recon=N*real(ifft(ifftshift(U),N)); % reconstructed signal
t =[0:1:length(u_recon)-1]/fs; %recompute time index 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% filtered plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
% time domain signal
subplot(2,2,1);
plot(t*1e3,u_recon,'Color',[0,0,1],'linewidth',1);
xlim([0,1]);
xlabel('t [ms]','Fontsize',11);
ylabel('u(t)','Fontsize',11);
set(gca,'Fontsize',10,'linewidth',1);
% time domain signal - zoom
subplot(2,2,2);
plot(t*1e3,u_recon,'Color',[0,0,1],'linewidth',1);
xlim([0.1,0.2]);
xlabel('t [ms]','Fontsize',11);
ylabel('u(t)','Fontsize',11);
set(gca,'Fontsize',10,'linewidth',1);
title('u(t) zoom');
% magnitude
subplot(2,2,3);
stem(f/1e3,2*abs(U),'r','filled','MarkerSize',3); %magnitudes vs frequencies
xlabel('f [kHz]','Fontsize',11); ylabel('|U(f)|','Fontsize',11);
xlim([-100 100]);
ylim([0 0.015]);
set(gca,'Fontsize',10,'linewidth',1);
% magnitude - zoom around 180-350 kHz
subplot(2,2,4);
stem(f/1e3,2*abs(U),'r','filled','MarkerSize',3); %magnitudes vs frequencies
xlabel('f [kHz]','Fontsize',11); ylabel('|U(f)|','Fontsize',11);
title('U(f) zoom');
xlim([180 350]);
set(gca,'Fontsize',10,'linewidth',1);
%%%%%%%%%%%%%%%
fig = gcf;set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[30 10 fig_width fig_height]); 
fig.PaperPositionMode   = 'auto';