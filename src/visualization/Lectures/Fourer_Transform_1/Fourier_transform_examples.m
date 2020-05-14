clear all; close all;
% examples of Fourier transform

% load projectroot path
load project_paths projectroot src_path;
output_path = fullfile(projectroot,'reports','beamer_presentations','Lectures','Fourier_Transform_1','figs',filesep);
fig_width = 7; fig_height = 3; 
logoblue=[1,67,140]/255; % line color
darkblue=[0,0,139]/255;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Example 1 - cosine
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = 0.5;                            %amplitude of the cosine wave
fc=10;                              %frequency of the cosine wave [Hz]
phase=30;                         %desired phase shift of the cosine in degrees
fs=32*fc;                           %sampling frequency with oversampling factor 32
t=0:1/fs:2-1/fs;                  %2 seconds duration
phi = phase*pi/180;             %convert phase shift in degrees in radians
u=A*cos(2*pi*fc*t+phi);     %time domain signal with phase shift
% Fourier transform
N=256;                              %FFT size
U = 1/N*fftshift(fft(u,N));     %N-point complex DFT
% amplitude spectrum
df=fs/N;                            %frequency resolution
sampleIndex = -N/2:N/2-1;   %ordered index for FFT plot
f=sampleIndex*df;               %x-axis index converted to ordered frequencies
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example 1 - plotting results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; plot(t,u,'Color',logoblue,'linewidth',1); %plot the signal
figfilename = 'FFT_example_cos_time';
xlabel('t [s]','Fontsize',11);
ylabel('u(t)','Fontsize',11);
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% magnitude
figure;stem(f,abs(U),'r','filled','MarkerSize',3); %magnitudes vs frequencies
figfilename = 'FFT_example_cos_frequency';
xlabel('f [Hz]','Fontsize',11); ylabel('|U(k)|','Fontsize',11);
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% real
figure;stem(f,real(U),'r','filled','MarkerSize',3); %Real vs frequencies
figfilename = 'FFT_example_cos_real';
xlabel('f [Hz]','Fontsize',11); ylabel('Re(U(k))','Fontsize',11);
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% imag
figure;stem(f,imag(U),'r','filled','MarkerSize',3); %Real vs frequencies
figfilename = 'FFT_example_cos_imag';
xlabel('f [Hz]','Fontsize',11); ylabel('Im(U(k))','Fontsize',11);
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
%% Example 2 - sine
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = 0.5;                            %amplitude of the sine wave
fc=5;                                 %frequency of the sine wave [Hz]
phase=30;                          %desired phase shift of the cosine in degrees
fs=32*fc;                           %sampling frequency with oversampling factor 32
t=0:1/fs:2-1/fs;                  %2 seconds duration
phi = phase*pi/180;             %convert phase shift in degrees in radians
u=A*sin(2*pi*fc*t+phi);       %time domain signal with phase shift
% Fourier transform
N=256;                              %FFT size
U = 1/N*fftshift(fft(u,N));     %N-point complex DFT
% amplitude spectrum
df=fs/N;                            %frequency resolution
sampleIndex = -N/2:N/2-1;   %ordered index for FFT plot
f=sampleIndex*df;               %x-axis index converted to ordered frequencies
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example 2 - plotting results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; plot(t,u,'Color',logoblue,'linewidth',1); %plot the signal
figfilename = 'FFT_example_sin_time';
xlabel('t [s]','Fontsize',11);
ylabel('u(t)','Fontsize',11);
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% magnitude
figure;stem(f,abs(U),'r','filled','MarkerSize',3); %magnitudes vs frequencies
figfilename = 'FFT_example_sin_frequency';
xlabel('f [Hz]','Fontsize',11); ylabel('|U(k)|','Fontsize',11);
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% real
figure;stem(f,real(U),'r','filled','MarkerSize',3); %Real vs frequencies
figfilename = 'FFT_example_sin_real';
xlabel('f [Hz]','Fontsize',11); ylabel('Re(U(k))','Fontsize',11);
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% imag
figure;stem(f,imag(U),'r','filled','MarkerSize',3); %Real vs frequencies
figfilename = 'FFT_example_sin_imag';
xlabel('f [Hz]','Fontsize',11); ylabel('Im(U(k))','Fontsize',11);
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Example 3 - cosine + sine
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = 0.5;                            %amplitude of the cosine wave
B = 0.3;                             %amplitude of the sine wave
fcA=10;                             %frequency of the cosine wave [Hz]
fcB=5;                               %frequency of the cosine wave [Hz]
phase=30;                           %desired phase shift of the cosine in degrees
fs=32*fcA;                          %sampling frequency with oversampling factor 32
t=0:1/fs:2-1/fs;                    %2 seconds duration
phi = phase*pi/180;               %convert phase shift in degrees in radians
u=A*cos(2*pi*fcA*t+phi)+B*sin(2*pi*fcB*t+phi);%time domain signal with phase shift
% Fourier transform
N=256;                              %FFT size
U = 1/N*fftshift(fft(u,N));     %N-point complex DFT
% amplitude spectrum
df=fs/N;                            %frequency resolution
sampleIndex = -N/2:N/2-1;   %ordered index for FFT plot
f=sampleIndex*df;               %x-axis index converted to ordered frequencies
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example 3 - plotting results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; plot(t,u,'Color',logoblue,'linewidth',1); %plot the signal
figfilename = 'FFT_example_sin_cos_time';
xlabel('t [s]','Fontsize',11);
ylabel('u(t)','Fontsize',11);
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% magnitude
figure;stem(f,abs(U),'r','filled','MarkerSize',3); %magnitudes vs frequencies
figfilename = 'FFT_example_sin_cos_frequency';
xlabel('f [Hz]','Fontsize',11); ylabel('|U(k)|','Fontsize',11);
xlim([-30 30]);
ylim([0,0.3]);
xticks([-20, -10,-5,0,5,10,20]);
yticks([0,0.15, 0.25]); % works in new matlab
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% real
figure;stem(f,real(U),'r','filled','MarkerSize',3); %Real vs frequencies
figfilename = 'FFT_example_sin_cos_real';
xlabel('f [Hz]','Fontsize',11); ylabel('Re(U(k))','Fontsize',11);
xlim([-30 30]);
ylim([0,0.25]);
xticks([-20, -10,-5,0,5,10,20]);
yticks([0,0.075,0.2165]); % works in new matlab
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% imag
figure;stem(f,imag(U),'r','filled','MarkerSize',3); %Real vs frequencies
figfilename = 'FFT_example_sin_cos_imag';
xlabel('f [Hz]','Fontsize',11); ylabel('Im(U(k))','Fontsize',11);
xlim([-30 30]);
ylim([-0.25,0.25]);
xticks([-20, -10,-5,0,5,10,20]);
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
%% Example 4 - Rectangular pulse (wide)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = 1;                            % amplitude of the rectangular pulse
fs=512;                          % sampling frequency [Hz]
t=-15:1/fs:15-1/fs;           % time range
T = 10;                            % rectangle pulse width [s]
half_T=T/2;                      % half of the rectangle pulse width [s]
N=2^nextpow2(length(t));   % number of FFT points
f=(-N/2:(N-1)/2)*fs/N;        % frequency range [Hz]
u=double(abs(t)<half_T);    % time domain rectangle pulse signal
U = 1/fs*fftshift(fft(u,N));    % N-point complex DFT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example 4 - plotting results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; plot(t,u,'Color',logoblue,'linewidth',1); %plot the signal
figfilename = 'FFT_example_rectangular_pulse_wide_time';
xlabel('t [s]','Fontsize',11);
ylabel('u(t)','Fontsize',11);
xlim([-10 10]);
ylim([0,1.2]);
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% amplitude spectrum
figure;stem(f,abs(U),'r','filled','MarkerSize',3); %magnitudes vs frequencies
figfilename = 'FFT_example_rectangular_pulse_wide_frequency';
xlabel('f [Hz]','Fontsize',11); ylabel('|U(k)|','Fontsize',11);
xlim([-2 2]);
xticks([-2, -1, 0, 1, 2]);
xticklabels({'-20/T','-10/T',0,'10/T','20/T'});
%ylim([-10,10]);
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% real
figure;stem(f,real(U),'r','filled','MarkerSize',3); %Real vs frequencies
figfilename = 'FFT_example_rectangular_pulse_wide_real';
xlabel('f [Hz]','Fontsize',11); ylabel('Re(U(k))','Fontsize',11);
xlim([-2 2]);
ylim([-10,10]);
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% imag
figure;stem(f,imag(U),'r','filled','MarkerSize',3); %Real vs frequencies
figfilename = 'FFT_example_rectangular_pulse_wide_imag';
xlabel('f [Hz]','Fontsize',11); ylabel('Im(U(k))','Fontsize',11);
xlim([-2 2]);
ylim([-10,10]);
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Example 5 - Rectangular pulse (narrow)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = 1;                            % amplitude of the rectangular pulse
fs=512;                          % sampling frequency [Hz]
t=-15:1/fs:15-1/fs;           % time range
T = 1;                            % rectangle pulse width [s]
half_T=T/2;                      % half of the rectangle pulse width [s]
N=2^nextpow2(length(t));   % number of FFT points
f=(-N/2:(N-1)/2)*fs/N;        % frequency range [Hz]
u=double(abs(t)<half_T);    % time domain rectangle pulse signal
U = 1/fs*fftshift(fft(u,N));    % N-point complex DFT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example 5 - plotting results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; plot(t,u,'Color',logoblue,'linewidth',1); %plot the signal
figfilename = 'FFT_example_rectangular_pulse_narrow_time';
xlabel('t [s]','Fontsize',11);
ylabel('u(t)','Fontsize',11);
xlim([-10 10]);
ylim([0,1.2]);
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% amplitude spectrum
figure;stem(f,abs(U),'r','filled','MarkerSize',3); %magnitudes vs frequencies
figfilename = 'FFT_example_rectangular_pulse_narrow_frequency';
xlabel('f [Hz]','Fontsize',11); ylabel('|U(k)|','Fontsize',11);
xlim([-2 2]);
%ylim([-10,10]);
xticks([-2, -1, 0, 1, 2]);
xticklabels({'-2/T','-1/T',0,'1/T','2/T'});
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% real
figure;stem(f,real(U),'r','filled','MarkerSize',3); %Real vs frequencies
figfilename = 'FFT_example_rectangular_pulse_narrow_real';
xlabel('f [Hz]','Fontsize',11); ylabel('Re(U(k))','Fontsize',11);
xlim([-2 2]);
%ylim([-10,10]);
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% imag
figure;stem(f,imag(U),'r','filled','MarkerSize',3); %Real vs frequencies
figfilename = 'FFT_example_rectangular_pulse_narrow_imag';
xlabel('f [Hz]','Fontsize',11); ylabel('Im(U(k))','Fontsize',11);
xlim([-2 2]);
%ylim([-2,2]);
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Example 6 - Gaussian pulse
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c = 3e+8;                 % Speed of light [m/sec]
lambda = 800e-9;       % Wavelength [nm]
fc = c/lambda;           % Actual Frequency of light [THz]
fs = fc*12;                % Sampling frequency with oversampling factor 12
dt = 1/fs;                  % Unit time [fs]
L = 400;                   % Length of signal
sigma = 8e-15;          % Pulse duration
t = (0:L-1)*dt;          % Time base
t0 = max(t)/2;           % Used to centering the pulse
% Electric field
Egauss = (exp(-2*log(2)*(t-t0).^2/(sigma)^2)).*cos(-2*pi*fc*(t-t0));
NFFT = 2^nextpow2(L); % number of FFT points
X = fft(Egauss,NFFT)/L;
%Pxx=X.*conj(X)/(NFFT*NFFT); %computing power with proper scaling
freq = 0.5*fs*linspace(0,1,NFFT/2+1);  % (full range) Frequency Vector - without fft shift
f=(-NFFT/2:(NFFT-1)/2)*fs/NFFT;        % frequency range [Hz] - with fft shift
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example 6 - plotting results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; plot(t/1e-15,real(Egauss),'Color',logoblue,'linewidth',1); %plot the signal
figfilename = 'FFT_example_gauss_time';
xlabel('t [fs]','Fontsize',11);
ylabel('u(t)','Fontsize',11);
xlim([0 80]);
ylim([-1.2,1.2]);
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% amplitude spectrum
figure;stem(freq/1e+12,2*abs(X(1:NFFT/2+1)),'r','filled','MarkerSize',3); %magnitudes vs frequencies
figfilename = 'FFT_example_gauss_frequency';
xlabel('f [THz]','Fontsize',11); ylabel('|U(k)|','Fontsize',11);
xlim([260 500]);
xticks([260, 375, 500]);
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
% real
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;stem(freq/1e+12,real(X(1:NFFT/2+1)),'r','filled','MarkerSize',3); %Real vs frequencies
figfilename = 'FFT_example_gauss_real';
xlabel('f [Hz]','Fontsize',11); ylabel('Re(U(k))','Fontsize',11);
xlim([260 500]);
xticks([260, 375, 500]);
%ylim([-0.4,0.6]);
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
% imag
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;stem(freq/1e+12,imag(X(1:NFFT/2+1)),'r','filled','MarkerSize',3); %Real vs frequencies
figfilename = 'FFT_example_gauss_imag';
xlabel('f [Hz]','Fontsize',11); ylabel('Im(U(k))','Fontsize',11);
xlim([260 500]);
xticks([260, 375, 500]);
%ylim([-2,2]);
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
