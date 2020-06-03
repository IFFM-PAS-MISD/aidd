% this code comes from
% S. L. Brunton and J. N. Kutz, Data Driven Science \& Engineering: Machine Learning, 
% Dynamical Systems, and Control, Cambridge Textbook, 2019
% it is slightly modified
clear all, close all, clc
a = 1;        % Thermal diffusivity constant
L = 100;      % Length of domain
N = 1000;     % Number of discretization points
dx = L/N;
x = -L/2:dx:L/2-dx; % Define x domain

% Define discrete wavenumbers
kappa = (2*pi/L)*[-N/2:N/2-1];

% Initial condition
u0 = 0*x;
u0((L/2 - L/10)/dx:(L/2 + L/10)/dx) = 1;
%%
% Simulate in Fourier wavenumber (kappa) domain
t = [0:0.8:80]'; % integration time steps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% duhat(t,kappa)/dt = -a^2+kappa^2 uhat(t,kappa) -> uhat(t,kappa)
[t,uhat]=ode45(@(t,uhat) rhsHeat2(uhat,kappa,a),t,fftshift(fft(u0))); % for more details type: help ode45
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u=zeros(length(t),N);
for k = 1:length(t) % IFFT to return to spatial domain
    u(k,:) = ifft(ifftshift(uhat(k,:)));
end

%% FIGURES (PRODUCTION)
% Plot solution in time
figure; waterfall((u(1:10:end,:)));
figure; imagesc(flipud(u));
%%
figure
CC = colormap(jet(length(t)));
for k = 1:length(t)
    if(mod(k-1,10)==0)
        plot(x,u(k,:),'Color',CC(k,:),'LineWidth',1.5)
        hold on, grid on, drawnow
        % xlabel('Spatial variable, x')
        % ylabel('Temperature, u(x,t)')
        axis([-50 50 -.1 1.1])
        set(gca,'LineWidth',1.2,'FontSize',12);
        set(gcf,'Position',[100 100 550 220]);
        set(gcf,'PaperPositionMode','auto')
        % print('-depsc2', '-loose', '../../figures/FFTHeat1');
        pause(0.5);
    end   
end
%%
%
figure
subplot(1,2,1)
h=waterfall(u(1:10:end,:));
set(h,'LineWidth',2,'FaceAlpha',.5);
colormap(jet/1.5)
view(22,29)
set(gca,'LineWidth',1.5)
set(gca,'XTick',[0 500 1000],'XTickLabels',{})
set(gca,'ZTick',[0 .5 1],'ZTickLabels',{})
set(gca,'YTick',[0 5 10],'YTickLabels',{})

subplot(1,2,2)
imagesc(flipud(u));
set(gca,'LineWidth',1.5)
set(gca,'XTick',[0 500 1000],'XTickLabels',{})
set(gca,'YTick',[0 50 100],'YTickLabels',{})

colormap jet
set(gcf,'Position',[100 100 600 250])
set(gcf,'PaperPositionMode','auto')
% print('-depsc2', '-loose', '../../figures/FFTHeat2');