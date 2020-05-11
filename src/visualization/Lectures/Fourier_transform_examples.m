clear all; close all;
% examples of Fourier transform

% load projectroot path
load project_paths projectroot src_path;
output_path = fullfile(projectroot,'reports','beamer_presentations','Lectures','Fourier_Transform_1','figs',filesep);
fig_width = 7; fig_height = 3; 
logoblue=[1,67,140]/255; % line color
darkblue=[0,0,139]/255;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Example 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = 0.5; %amplitude of the cosine wave
fc=10;%frequency of the cosine wave [Hz]
phase=30; %desired phase shift of the cosine in degrees
fs=32*fc;%sampling frequency with oversampling factor 32

t=0:1/fs:2-1/fs;%2 seconds duration
 
phi = phase*pi/180; %convert phase shift in degrees in radians
u=A*cos(2*pi*fc*t+phi);%time domain signal with phase shift
 
figure; plot(t,u,'Color',logoblue,'linewidth',1); %plot the signal
figfilename = 'FFT_example_cos_time';
xlabel('t [s]');
ylabel('u(t)');
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
%% 
% Fourier transform
N=256; %FFT size
U = 1/N*fftshift(fft(u,N));%N-point complex DFT
% amplitude spectrum
df=fs/N; %frequency resolution
sampleIndex = -N/2:N/2-1; %ordered index for FFT plot
f=sampleIndex*df; %x-axis index converted to ordered frequencies
figure;stem(f,abs(U),'r','filled','MarkerSize',3); %magnitudes vs frequencies
figfilename = 'FFT_example_cos_frequency';
xlabel('f [Hz]'); ylabel('|U(k)|');
xlim([-30 30]);
ylim([0,0.3]);
xticks([-20, -10,0,10,20]);
yticks([0,0.25]); % works in new matlab
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
% real
figure;stem(f,real(U),'r','filled','MarkerSize',3); %Real vs frequencies
figfilename = 'FFT_example_cos_real';
xlabel('f [Hz]'); ylabel('Re(U(k))');
xlim([-30 30]);
ylim([0,0.25]);
xticks([-20, -10,0,10,20]);
yticks([0,0.2165]); % works in new matlab
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
% imag
figure;stem(f,imag(U),'r','filled','MarkerSize',3); %Real vs frequencies
figfilename = 'FFT_example_cos_imag';
xlabel('f [Hz]'); ylabel('Im(U(k))');
xlim([-30 30]);
ylim([-0.25,0.25]);
xticks([-20, -10,0,10,20]);
yticks([-0.125,0,0.125]); % works in new matlab
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Example 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = 0.5; %amplitude of the cosine wave
fc=5;%frequency of the cosine wave [Hz]
phase=30; %desired phase shift of the cosine in degrees
fs=32*fc;%sampling frequency with oversampling factor 32

t=0:1/fs:2-1/fs;%2 seconds duration
 
phi = phase*pi/180; %convert phase shift in degrees in radians
u=A*sin(2*pi*fc*t+phi);%time domain signal with phase shift
 
figure; plot(t,u,'Color',logoblue,'linewidth',1); %plot the signal
figfilename = 'FFT_example_sin_time';
xlabel('t [s]');
ylabel('u(t)');
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
%% 
% Fourier transform
N=256; %FFT size
U = 1/N*fftshift(fft(u,N));%N-point complex DFT
% amplitude spectrum
df=fs/N; %frequency resolution
sampleIndex = -N/2:N/2-1; %ordered index for FFT plot
f=sampleIndex*df; %x-axis index converted to ordered frequencies
figure;stem(f,abs(U),'r','filled','MarkerSize',3); %magnitudes vs frequencies
figfilename = 'FFT_example_sin_frequency';
xlabel('f [Hz]'); ylabel('|U(k)|');
xlim([-30 30]);
ylim([0,0.3]);
xticks([-20, -5,0,5,20]);
yticks([0,0.25]); % works in new matlab
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
% real
figure;stem(f,real(U),'r','filled','MarkerSize',3); %Real vs frequencies
figfilename = 'FFT_example_sin_real';
xlabel('f [Hz]'); ylabel('Re(U(k))');
xlim([-30 30]);
ylim([0,0.25]);
xticks([-20, -5,0,5,20]);
yticks([0,0.125]); % works in new matlab
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
% imag
figure;stem(f,imag(U),'r','filled','MarkerSize',3); %Real vs frequencies
figfilename = 'FFT_example_sin_imag';
xlabel('f [Hz]'); ylabel('Im(U(k))');
xlim([-30 30]);
ylim([-0.25,0.25]);
xticks([-20, -5,0,5,20]);
yticks([-0.2165,0,0.2165]); % works in new matlab
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
%
% as in presentation
% j=1:N;
% dt=1/fs;
% t1 = 0+ (j-1)*dt;
% u1=A*cos(2*pi*fc*t1+phi);%time domain signal with phase shift
