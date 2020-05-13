clear all; close all;
% Aliasing example

% load projectroot path
load project_paths projectroot src_path;
output_path = fullfile(projectroot,'reports','beamer_presentations','Lectures','Fourier_Transform_1','figs',filesep);
fig_width = 6; fig_height = 5; 
logoblue=[1,67,140]/255; % line color
darkblue=[0,0,139]/255;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Example 1 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fs=1;
t=0:0.01:8;
t3=0:1:8;
fs_hat = 1.75*fs;
fs_hat2 = 2.5*fs;
t4=0:1/(fs_hat):8;
t5=0:1/(fs_hat2):8;
s1=sin(2*pi*1/8*fs*t);      % signal alias 1
s2=sin(2*pi*(1/8-1)*fs*t); % signal alias 2
s3=sin(2*pi*1/8*fs*t3);     % % values at sampling interval
s4=sin(2*pi*(1/8-1)*fs*t4);     % values at Nyquist-Shannon sampling frequency
s5=sin(2*pi*(1/8-1)*fs*t5);     % values at proper sampling frequency
% Fourier transform
N=256;                              %FFT size
U1 = 1/N*fftshift(fft(s1,N));     %N-point complex DFT
U2 = 1/N*fftshift(fft(s2,N));     %N-point complex DFT
% amplitude spectrum
df=fs/N;                            %frequency resolution
sampleIndex = -N/2:N/2-1;   %ordered index for FFT plot
f=sampleIndex*df;               %x-axis index converted to ordered frequencies
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example 1 - plotting results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%figure; plot(t,s1,'Color','r','linewidth',1); %plot the signal
%hold on;
plot(t,s2,'Color',logoblue,'linewidth',1); %
%plot(t3,s3,'ko','markerfacecolor','k'); % values at sampling interval
figfilename = 'Aliasing_1';
xlim([0,8]);
xticks([0:8]);
%xlabel('t [s]','Fontsize',11);
%ylabel('u(t)','Fontsize',11);
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%figure; plot(t,s1,'Color','r','linewidth',1); %plot the signal
plot(t,s2,'Color',logoblue,'linewidth',1); %
hold on;
plot(t3,s3,'ko','markerfacecolor','k'); % values at sampling interval
figfilename = 'Aliasing_2';
xlim([0,8]);
xticks([0:8]);
%xlabel('t [s]','Fontsize',11);
%ylabel('u(t)','Fontsize',11);
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; plot(t,s1,'Color','r','linewidth',1); %plot the signal
hold on;
plot(t,s2,'Color',logoblue,'linewidth',1); %
plot(t3,s3,'ko','markerfacecolor','k'); % values at sampling interval
figfilename = 'Aliasing_3';
xlim([0,8]);
xticks([0:8]);
%xlabel('t [s]','Fontsize',11);
%ylabel('u(t)','Fontsize',11);
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; plot(t,s1,'Color','r','linewidth',1); %plot the signal
hold on;
plot(t,s2,'Color',logoblue,'linewidth',1); %
plot(t3,s3,'ko','markerfacecolor','k'); % values at sampling interval
plot(t4,s4,'d','Color','m'); % values at Nyquist-Shannon sampling frequency
figfilename = 'Aliasing_4';
xlim([0,8]);
xticks([0:8]);
%xlabel('t [s]','Fontsize',11);
%ylabel('u(t)','Fontsize',11);
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; plot(t,s1,'Color','r','linewidth',1); %plot the signal
hold on;
plot(t,s2,'Color',logoblue,'linewidth',1); %
plot(t3,s3,'ko','markerfacecolor','k'); % values at sampling interval
plot(t4,s4,'d','Color','m'); % values at Nyquist-Shannon sampling frequency
plot(t5,s5,'>','Color','g','markerfacecolor','g'); % values at proper sampling frequency
figfilename = 'Aliasing_5';
xlim([0,8]);
xticks([0:8]);
%xlabel('t [s]','Fontsize',11);
%ylabel('u(t)','Fontsize',11);
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600');
