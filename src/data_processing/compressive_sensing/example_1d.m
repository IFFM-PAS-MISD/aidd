% example of Beating Nyquist with compressed sensing
% https://www.youtube.com/watch?v=A8W1I3mtjp8&list=PLMrJAkhIeNNRHP5UA-gIimsXLQyHXxRty&index=7

clear all; close all;
M=8;
RMSE=zeros(M,1);
RRMSE=zeros(M,1);
RMSE_half=zeros(M,1);
No_of_measurements=zeros(M,1);
for k=1:M
    %% Generate signal, PSD of signal
    n = 4096;
t = linspace(0, 1, n);
x = (cos(2 * 97 * pi * t) + cos(2 * 777 * pi * t));%+0.2*rand(n,1)')-0.1;
xt = fft(x); % Fourier trasnformed signal
%xd = dct(x); % Discrete cosine transformed signal
PSD = xt.*conj(xt)/n; % Power spectral density

%% Randomly sample signal
p= k*n/32; % num. random samples, p=n/32 CR=128/(2*777)*100 ~8.2% ?
No_of_measurements(k)=p;
perm = round(rand(p,1)*n);
perm(perm==0)=1;
y = x(perm); % compressed measurement
%y = x(n/p:n/p:end); % measurement on uniform mesh is bad!
%% Plot 1
time_window = [1024 1280]/4096;
figure
subplot(2,2,2)
freq = n/(n)*[0:n];
L = 1:floor(n/2);
plot(freq(L),PSD(L),'k','LineWidth',2);
%stem(freq(L),PSD(L),'k','LineWidth',2);
xlabel('Frequency, [Hz]');set(gca,'Fontsize',14);
subplot(2,2,1)
plot(t,x,'k','LineWidth',2);
hold on;
plot(t(perm),y,'cx','Linewidth',3);
%plot(t(n/p:n/p:end),y,'cx','Linewidth',3);
xlabel('Time [s]'); set(gca, 'Fontsize',14);
xlim([0.25 0.31]);
%% Solve compressed sensing problem
Psi = dct(eye(n,n)); % build Psi
Theta = Psi(perm, :); % random rows of Psi

%s = cosamp(Theta, y', 10, 1.e-10, 50); % CS via matching pursuit
% https://github.com/chenhaodev/matlab-simul/blob/master/uwb140617/cosamp.m
% Algorithm as described in "CoSaMP: Iterative signal recovery from 
% incomplete and inaccurate samples" by Deanna Needell and Joel Tropp.
% Programmed by Chenhao

%s = CoSaMP(Theta, y', k*10); % CS via matching pursuit
s = CoSaMP(Theta, y', 10); % CS via matching pursuit
% s(194)=s(194)*1.5;
% s(195)=s(195)*1.5;
% s=abs(s);
% https://uk.mathworks.com/matlabcentral/fileexchange/32402-cosamp-and-omp-for-sparse-recovery
% Orthogonal matching Pursuit (OMP) and Compressive Sampling Matched Pursuit (CoSaMP) algorithm (see Needell and Tropp's 2008 paper http://arxiv.org/abs/0803.2392 ).

xrecon = idct(s); % reconstruct full signal

%% Plot 2
subplot(2,2,3)
plot(t,xrecon, 'c', 'LineWidth',2);
ylim([-2,2]);
xlabel('Time [s]'); set(gca, 'Fontsize',14);
xlim([0.25 0.31]);
subplot(2,2,4)
xtrecon = fft(xrecon,n); % computes the fast discrete Fourier transform
PSDrecon = xtrecon.*conj(xtrecon)/n; % Power spectrum of reconstructed signal
plot(freq(L),PSDrecon(L),'c', 'LineWidth',2);
xlabel('Frequency [Hz]'); set(gca,'Fontsize',14);

%%
RMSE(k) = sqrt(mean((x' - xrecon).^2))  % Root Mean Squared Error
RMSE_half(k) = sqrt(mean((x(1:n/2)' - xrecon(1:n/2)).^2))  % Root Mean Squared Error

RRMSE(k)=RMSE(k)/sqrt(mean((x').^2)) % Relative Root Mean Squared Error
end

figure;
subplot(3,1,1);
plot(No_of_measurements,RMSE,'o-');
ylabel('RMSE');
xlabel('No of measurements');
subplot(3,1,2);
plot(No_of_measurements,RMSE_half,'o-');
xlabel('No of measurements');
title('RMSE half of the signal');
subplot(3,1,3);
plot(No_of_measurements,RRMSE,'o-');

xlabel('No of measurements');
ylabel('RRMSE');

figure;
plot(t,xrecon);hold on; plot(t,x,'r');