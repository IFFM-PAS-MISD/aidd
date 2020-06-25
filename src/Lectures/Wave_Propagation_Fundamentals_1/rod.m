%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% program rod
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% example 1 - undamped longitudinal wave (elementary rod theory)
close all; clear all;
fig_width = 16; fig_height = 8; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ke=zeros(2,2); % initialization of dynamic stiffness matrix
ke1=zeros(2,2);% initialization of stiffness matrix of Sthrow-off
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% material data (steel)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
e=210e9; % Young modulus (no damping)
nu=0.3; % Poisson ratio (not used in the elementary rod theory; Love theory take it into account)
ro=7860; % density
L=1.5; % rod Length [m]
b=0.01; % rod width [m]
h=0.01; % rod height [m]
a=b*h; % cross-section area
%J=1/6*(b*h*(b^2+h^2)); % polar moment of area; rod of rectangular cross-section 
%J=1/2*pi*r^4; % polar moment of area; rod of circular cross-section of radius r
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nx=1024; % number of points along the rod
dx=L/(Nx-1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tt=0.5e-3; % total time 
fm=2e4;     % moduLation frequency of the  signal [Hz]]
fc=5*fm;   % carrier frequency of the signal [Hz]
t_t=1/fm;   % totaL duration time of the excitation [s]
t_1=0e-4;    % excitation initiation time [s]
t_2=t_1+t_t; % excitation termination time [s]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fs=32*fc;              %sampling frequency with oversampling factor 32
dt=1/fs;                  %time step
t=0:dt:tt;
%excitation signal (force vector)
f=zeros(length(t),1);
for n=1:length(t)
  if (t(n) >= t_1) && (t(n) <= t_2)
    f(n)=0.5*(1-cos(2*pi*fm*(t(n)-t_2)))*sin(2*pi*fc*(t(n)-t_1));
  end
end
t1=t;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=2^nextpow2(length(f)); %number of points in FFT
t =[0:1:(N-1)]/fs; %recompute time index 
% amplitude spectrum
df=fs/N;                            %frequency resoLution
sampleIndex = -N/2:N/2-1;   %ordered index for FFT plot (double-seded spectrum)
freq=sampleIndex*df;               %t-axis index converted to ordered frequencies
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Q=zeros(2,N);
un=zeros(N,Nx);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
F = 1/N*fftshift(fft(f,N)); % force vector in the frequency domain
figure(1);
subplot(1,2,1);
plot(t1*1e3,f);
xlabel('time [ms]');
ylabel('Force [N]');
axis([0 (1/fm+t_1)*1e3 -1 1]);
subplot(1,2,2);
plot(freq/1e3,abs(F));
axis([0 2*fc/1e3 0 max(abs(F))]);
xlabel('Frequency [kHz]');
%%%%%%%%%%%%%%%
fig = gcf;set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[1 18 fig_width fig_height]); 
fig.PaperPositionMode   = 'auto';
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
om=zeros(N,1);
ne=zeros(1,2);
for m=[N/2+2:N]
      [m]
      om(m)=2*pi*freq(m);
      k=om(m)*sqrt(ro/e);% elementary rod theory
      %k=sqrt((om(m)^2*ro)/(e+0.1*om(m)^2));% artificial dispersive rod
      %k=sqrt( (om(m)^2*ro -1i*om(m) * eta)/e );% damped rod % eta - viscous damping
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % spectraL rod element - dynamic stiffness
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      ke(1,1)=k*a*e*cot(k*L);
      ke(1,2)=-k*a*e*csc(k*L);
      ke(2,1)=-k*a*e*csc(k*L);
      ke(2,2)=k*a*e*cot(k*L);
%           according to Doyle     
%           delta=1-exp(-1i*2*k*L);
%           C=e*a*1i*k*L/(L*delta);
%           ke(1,1)=C*(1+exp(-1i*2*k*L));
%           ke(1,2)=C*(-2)*exp(-1i*k*L);
%           ke(2,1)=C*(-2)*exp(-1i*k*L);
%           ke(2,2)=C*(1+exp(-1i*2*k*L));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         %ke1(1,1)=e*a*1i*k; % throw-off element on the left side
         ke1(2,2)=e*a*1i*k; % throw-off element on the right side
         kd=ke+ke1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %QQ=kd\(i*om(m)*[F(m);0]);  % velocity
      %QQ=kd\([F(m);0]);  % displacement (force at left end)
      %QQ=kd\([F(m);F(m)]);  % displacement (force at left and right end)
      QQ=kd\([0;F(m)]);  % displacement (force at left and right end)
      %QQ=-om(m)^2*(kd\([U(m);0])); % acceleration 
      Q(:,m)=Q(:,m)+QQ; % nodal displacements at the time step m
      cc=0;
        for x=0:dx:L
            cc=cc+1;
            % shape functions
            ne(1)=csc(k*L)*sin(k*(L-x)); 
            ne(2)=csc(k*L)*sin(k*x);
            un(m,cc)=ne*QQ;
        end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
yn=zeros(Nx,N);
for nx=1:Nx
  yn(nx,:)=N*real(ifft(ifftshift(un(:,nx)),N)); % reconstructed signal along the rod at discrete points
end
u1=N*real(ifft(ifftshift(Q(1,:)),N)); % reconstructed signal at the beginning of the rod (left side)
u2=N*real(ifft(ifftshift(Q(2,:)),N)); % reconstructed signal at the end of the rod (right side)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% signals at nodes
figure(2);
subplot(2,1,1); 
plot(t*1e3,real(u1));
xlabel('time [ms]');xlim([0 t(end)*1e3]);
subplot(2,1,2); 
plot(t*1e3,real(u2));
xlabel('time [ms]');xlim([0 t(end)*1e3]);
%%%%%%%%%%%%%%%
fig = gcf;set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[18 18 fig_width fig_height]); 
fig.PaperPositionMode   = 'auto';
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% animation
figure(3);
fig = gcf;set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[35 18 fig_width fig_height]); 
fig.PaperPositionMode   = 'auto';
xn=0:dx:L;
smax=max(max(abs(yn)));
for m =1:N
    plot(xn,real(yn(:,m)));
    axis([0 L -smax smax]);
    xlabel('x [m]');
    pause(0.01);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% example 2 - problem of time window - undamped longitudinal wave (elementary rod theory)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ke=zeros(2,2); % initialization of dynamic stiffness matrix
ke1=zeros(2,2);% initialization of stiffness matrix of throw-off
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% material data (steel)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
e=210e9; % Young modulus (no damping)
nu=0.3; % Poisson ratio (not used in the elementary rod theory; Love theory take it into account)
ro=7860; % density
L=1.5; % rod Length [m]
b=0.01; % rod width [m]
h=0.01; % rod height [m]
a=b*h; % cross-section area
%J=1/6*(b*h*(b^2+h^2)); % polar moment of area; rod of rectangular cross-section 
%J=1/2*pi*r^4; % polar moment of area; rod of circular cross-section of radius r
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nx=1024; % number of points along the rod
dx=L/(Nx-1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tt=0.3e-3; % total time (too short)
fm=2e4;     % moduLation frequency of the  signal [Hz]]
fc=5*fm;   % carrier frequency of the signal [Hz]
t_t=1/fm;   % totaL duration time of the excitation [s]
t_1=0e-4;    % excitation initiation time [s]
t_2=t_1+t_t; % excitation termination time [s]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fs=32*fc;              %sampling frequency with oversampling factor 32
dt=1/fs;                  %time step
t=0:dt:tt;
%excitation signal (force vector)
f=zeros(length(t),1);
for n=1:length(t)
  if (t(n) >= t_1) && (t(n) <= t_2)
    f(n)=0.5*(1-cos(2*pi*fm*(t(n)-t_2)))*sin(2*pi*fc*(t(n)-t_1));
  end
end
t1=t;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=2^nextpow2(length(f)); %number of points in FFT
t =[0:1:(N-1)]/fs; %recompute time index 
% amplitude spectrum
df=fs/N;                            %frequency resoLution
sampleIndex = -N/2:N/2-1;   %ordered index for FFT plot (double-seded spectrum)
freq=sampleIndex*df;               %t-axis index converted to ordered frequencies
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Q=zeros(2,N);
un=zeros(N,Nx);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
F = 1/N*fftshift(fft(f,N)); % force vector in the frequency domain
figure(4);
subplot(1,2,1);
plot(t1*1e3,f);
xlabel('time [ms]');
ylabel('Force [N]');
axis([0 (1/fm+t_1)*1e3 -1 1]);
subplot(1,2,2);
plot(freq/1e3,abs(F));
axis([0 2*fc/1e3 0 max(abs(F))]);
xlabel('Frequency [kHz]');
%%%%%%%%%%%%%%%
fig = gcf;set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[1 8 fig_width fig_height]); 
fig.PaperPositionMode   = 'auto';
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
om=zeros(N,1);
ne=zeros(1,2);
for m=[N/2+2:N]
      [m]
      om(m)=2*pi*freq(m);
      k=om(m)*sqrt(ro/e);% elementary rod theory
      %k=sqrt((om(m)^2*ro)/(e+0.1*om(m)^2));% artificial dispersive rod
      %k=sqrt( (om(m)^2*ro -1i*om(m) * eta)/e );% damped rod % eta - viscous damping
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % spectraL rod element - dynamic stiffness
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      ke(1,1)=k*a*e*cot(k*L);
      ke(1,2)=-k*a*e*csc(k*L);
      ke(2,1)=-k*a*e*csc(k*L);
      ke(2,2)=k*a*e*cot(k*L);
%           according to Doyle     
%           delta=1-exp(-1i*2*k*L);
%           C=e*a*1i*k*L/(L*delta);
%           ke(1,1)=C*(1+exp(-1i*2*k*L));
%           ke(1,2)=C*(-2)*exp(-1i*k*L);
%           ke(2,1)=C*(-2)*exp(-1i*k*L);
%           ke(2,2)=C*(1+exp(-1i*2*k*L));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         %ke1(1,1)=e*a*1i*k; % throw-off element on the left side
         ke1(2,2)=e*a*1i*k; % throw-off element on the right side
         kd=ke+ke1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %QQ=kd\(i*om(m)*[F(m);0]);  % velocity
      %QQ=kd\([F(m);0]);  % displacement (force at left end)
      %QQ=kd\([F(m);F(m)]);  % displacement (force at left and right end)
      QQ=kd\([0;F(m)]);  % displacement (force at left and right end)
      %QQ=-om(m)^2*(kd\([U(m);0])); % acceleration 
      Q(:,m)=Q(:,m)+QQ; % nodal displacements at the time step m
      cc=0;
        for x=0:dx:L
            cc=cc+1;
            % shape functions
            ne(1)=csc(k*L)*sin(k*(L-x)); 
            ne(2)=csc(k*L)*sin(k*x);
            un(m,cc)=ne*QQ;
        end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
yn=zeros(Nx,N);
for nx=1:Nx
  yn(nx,:)=N*real(ifft(ifftshift(un(:,nx)),N)); % reconstructed signal along the rod at discrete points
end
u1=N*real(ifft(ifftshift(Q(1,:)),N)); % reconstructed signal at the beginning of the rod (left side)
u2=N*real(ifft(ifftshift(Q(2,:)),N)); % reconstructed signal at the end of the rod (right side)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% signals at nodes
figure(5);
subplot(2,1,1); 
plot(t*1e3,real(u1));
xlabel('time [ms]');xlim([0 t(end)*1e3]);
subplot(2,1,2); 
plot(t*1e3,real(u2));
xlabel('time [ms]');xlim([0 t(end)*1e3]);
%%%%%%%%%%%%%%%
fig = gcf;set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[18 8 fig_width fig_height]); 
fig.PaperPositionMode   = 'auto';
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% animation
figure(6);
fig = gcf;set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[35 8 fig_width fig_height]); 
fig.PaperPositionMode   = 'auto';
xn=0:dx:L;
smax=max(max(abs(yn)));
for m =1:N
    plot(xn,real(yn(:,m)));
    axis([0 L -smax smax]);
    xlabel('x [m]');
    pause(0.01);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% example 3 - damped longitudinal wave (elementary rod theory)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ke=zeros(2,2); % initialization of dynamic stiffness matrix
ke1=zeros(2,2);% initialization of stiffness matrix of throw-off
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% material data (steel)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
e=210e9; % Young modulus (no damping)
eta=3e7; % damping coefficient
%e=210e9*(1+0.01i); % Young modulus (damping by imaginary part of Young's modulus)
%e=210e9*(1+0.0001i); % Young modulus (slightly damped)
nu=0.3; % Poisson ratio (not used in the elementary rod theory; Love theory take it into account)
ro=7860; % density
L=1.5; % rod Length [m]
b=0.01; % rod width [m]
h=0.01; % rod height [m]
a=b*h; % cross-section area
%J=1/6*(b*h*(b^2+h^2)); % polar moment of area; rod of rectangular cross-section 
%J=1/2*pi*r^4; % polar moment of area; rod of circular cross-section of radius r
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nx=1024; % number of points along the rod
dx=L/(Nx-1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tt=0.5e-3; % total time 
fm=2e4;     % moduLation frequency of the  signal [Hz]]
fc=5*fm;   % carrier frequency of the signal [Hz]
t_t=1/fm;   % totaL duration time of the excitation [s]
t_1=0e-4;    % excitation initiation time [s]
t_2=t_1+t_t; % excitation termination time [s]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fs=32*fc;              %sampling frequency with oversampling factor 32
dt=1/fs;                  %time step
t=0:dt:tt;
%excitation signal (force vector)
f=zeros(length(t),1);
for n=1:length(t)
  if (t(n) >= t_1) && (t(n) <= t_2)
    f(n)=0.5*(1-cos(2*pi*fm*(t(n)-t_2)))*sin(2*pi*fc*(t(n)-t_1));
  end
end
t1=t;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=2^nextpow2(length(f)); %number of points in FFT
t =[0:1:(N-1)]/fs; %recompute time index 
% amplitude spectrum
df=fs/N;                            %frequency resoLution
sampleIndex = -N/2:N/2-1;   %ordered index for FFT plot (double-seded spectrum)
freq=sampleIndex*df;               %t-axis index converted to ordered frequencies
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Q=zeros(2,N);
un=zeros(N,Nx);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
F = 1/N*fftshift(fft(f,N)); % force vector in the frequency domain
figure(1);
subplot(1,2,1);
plot(t1*1e3,f);
xlabel('time [ms]');
ylabel('Force [N]');
axis([0 (1/fm+t_1)*1e3 -1 1]);
subplot(1,2,2);
plot(freq/1e3,abs(F));
axis([0 2*fc/1e3 0 max(abs(F))]);
xlabel('Frequency [kHz]');
%%%%%%%%%%%%%%%
fig = gcf;set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[1 1 fig_width fig_height]); 
fig.PaperPositionMode   = 'auto';
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
om=zeros(N,1);
ne=zeros(1,2);
for m=[N/2+2:N]
      [m]
      om(m)=2*pi*freq(m);
      %k=om(m)*sqrt(ro/e);% elementary rod theory
      %k=sqrt((om(m)^2*ro)/(e+0.1*om(m)^2));% artificial dispersive rod
      k=sqrt( (om(m)^2*ro -1i*om(m) * eta)/e );% damped rod % eta - viscous damping
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % spectraL rod element - dynamic stiffness
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      ke(1,1)=k*a*e*cot(k*L);
      ke(1,2)=-k*a*e*csc(k*L);
      ke(2,1)=-k*a*e*csc(k*L);
      ke(2,2)=k*a*e*cot(k*L);
%           according to Doyle     
%           delta=1-exp(-1i*2*k*L);
%           C=e*a*1i*k*L/(L*delta);
%           ke(1,1)=C*(1+exp(-1i*2*k*L));
%           ke(1,2)=C*(-2)*exp(-1i*k*L);
%           ke(2,1)=C*(-2)*exp(-1i*k*L);
%           ke(2,2)=C*(1+exp(-1i*2*k*L));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         %ke1(1,1)=e*a*1i*k; % throw-off element on the left side
         ke1(2,2)=e*a*1i*k; % throw-off element on the right side
         kd=ke+ke1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %QQ=kd\(i*om(m)*[F(m);0]);  % velocity
      %QQ=kd\([F(m);0]);  % displacement (force at left end)
      %QQ=kd\([F(m);F(m)]);  % displacement (force at left and right end)
      QQ=kd\([0;F(m)]);  % displacement (force at left and right end)
      %QQ=-om(m)^2*(kd\([U(m);0])); % acceleration 
      Q(:,m)=Q(:,m)+QQ; % nodal displacements at the time step m
      cc=0;
        for x=0:dx:L
            cc=cc+1;
            % shape functions
            ne(1)=csc(k*L)*sin(k*(L-x)); 
            ne(2)=csc(k*L)*sin(k*x);
            un(m,cc)=ne*QQ;
        end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
yn=zeros(Nx,N);
for nx=1:Nx
  yn(nx,:)=N*real(ifft(ifftshift(un(:,nx)),N)); % reconstructed signal along the rod at discrete points
end
u1=N*real(ifft(ifftshift(Q(1,:)),N)); % reconstructed signal at the beginning of the rod (left side)
u2=N*real(ifft(ifftshift(Q(2,:)),N)); % reconstructed signal at the end of the rod (right side)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% signals at nodes
figure(2);
subplot(2,1,1); 
plot(t*1e3,real(u1));
xlabel('time [ms]');xlim([0 t(end)*1e3]);
subplot(2,1,2); 
plot(t*1e3,real(u2));
xlabel('time [ms]');xlim([0 t(end)*1e3]);
%%%%%%%%%%%%%%%
fig = gcf;set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[18 1 fig_width fig_height]); 
fig.PaperPositionMode   = 'auto';
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% animation
figure(3);
fig = gcf;set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[35 1 fig_width fig_height]); 
fig.PaperPositionMode   = 'auto';
xn=0:dx:L;
smax=max(max(abs(yn)));
for m =1:N
    plot(xn,real(yn(:,m)));
    axis([0 L -smax smax]);
    xlabel('x [m]');
    pause(0.01);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% example 4 - undamped dispersive longitudinal wave (artificial rod theory)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
ke=zeros(2,2); % initialization of dynamic stiffness matrix
ke1=zeros(2,2);% initialization of stiffness matrix of throw-off
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% material data (steel)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
e=210e9; % Young modulus (no damping)
nu=0.3; % Poisson ratio (not used in the elementary rod theory; Love theory take it into account)
ro=7860; % density
L=1.5; % rod Length [m]
b=0.01; % rod width [m]
h=0.01; % rod height [m]
a=b*h; % cross-section area
%J=1/6*(b*h*(b^2+h^2)); % polar moment of area; rod of rectangular cross-section 
%J=1/2*pi*r^4; % polar moment of area; rod of circular cross-section of radius r
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nx=1024; % number of points along the rod
dx=L/(Nx-1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tt=0.5e-3; % total time 
fm=2e4;     % moduLation frequency of the  signal [Hz]]
fc=5*fm;   % carrier frequency of the signal [Hz]
t_t=1/fm;   % totaL duration time of the excitation [s]
t_1=0e-4;    % excitation initiation time [s]
t_2=t_1+t_t; % excitation termination time [s]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fs=32*fc;              %sampling frequency with oversampling factor 32
dt=1/fs;                  %time step
t=0:dt:tt;
%excitation signal (force vector)
f=zeros(length(t),1);
for n=1:length(t)
  if (t(n) >= t_1) && (t(n) <= t_2)
    f(n)=0.5*(1-cos(2*pi*fm*(t(n)-t_2)))*sin(2*pi*fc*(t(n)-t_1));
  end
end
t1=t;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=2^nextpow2(length(f)); %number of points in FFT
t =[0:1:(N-1)]/fs; %recompute time index 
% amplitude spectrum
df=fs/N;                            %frequency resoLution
sampleIndex = -N/2:N/2-1;   %ordered index for FFT plot (double-seded spectrum)
freq=sampleIndex*df;               %t-axis index converted to ordered frequencies
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Q=zeros(2,N);
un=zeros(N,Nx);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
F = 1/N*fftshift(fft(f,N)); % force vector in the frequency domain
figure(1);
subplot(1,2,1);
plot(t1*1e3,f);
xlabel('time [ms]');
ylabel('Force [N]');
axis([0 (1/fm+t_1)*1e3 -1 1]);
subplot(1,2,2);
plot(freq/1e3,abs(F));
axis([0 2*fc/1e3 0 max(abs(F))]);
xlabel('Frequency [kHz]');
%%%%%%%%%%%%%%%
fig = gcf;set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[1 18 fig_width fig_height]); 
fig.PaperPositionMode   = 'auto';
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
om=zeros(N,1);
ne=zeros(1,2);
for m=[N/2+2:N]
      [m]
      om(m)=2*pi*freq(m);
      %k=om(m)*sqrt(ro/e);% elementary rod theory
      k=sqrt((om(m)^2*ro)/(e+0.1*om(m)^2));% artificial dispersive rod
      %k=sqrt( (om(m)^2*ro -1i*om(m) * eta)/e );% damped rod % eta - viscous damping
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % spectraL rod element - dynamic stiffness
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      ke(1,1)=k*a*e*cot(k*L);
      ke(1,2)=-k*a*e*csc(k*L);
      ke(2,1)=-k*a*e*csc(k*L);
      ke(2,2)=k*a*e*cot(k*L);
%           according to Doyle     
%           delta=1-exp(-1i*2*k*L);
%           C=e*a*1i*k*L/(L*delta);
%           ke(1,1)=C*(1+exp(-1i*2*k*L));
%           ke(1,2)=C*(-2)*exp(-1i*k*L);
%           ke(2,1)=C*(-2)*exp(-1i*k*L);
%           ke(2,2)=C*(1+exp(-1i*2*k*L));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         %ke1(1,1)=e*a*1i*k; % throw-off element on the left side
         ke1(2,2)=e*a*1i*k; % throw-off element on the right side
         kd=ke+ke1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %QQ=kd\(i*om(m)*[F(m);0]);  % velocity
      %QQ=kd\([F(m);0]);  % displacement (force at left end)
      %QQ=kd\([F(m);F(m)]);  % displacement (force at left and right end)
      QQ=kd\([0;F(m)]);  % displacement (force at left and right end)
      %QQ=-om(m)^2*(kd\([U(m);0])); % acceleration 
      Q(:,m)=Q(:,m)+QQ; % nodal displacements at the time step m
      cc=0;
        for x=0:dx:L
            cc=cc+1;
            % shape functions
            ne(1)=csc(k*L)*sin(k*(L-x)); 
            ne(2)=csc(k*L)*sin(k*x);
            un(m,cc)=ne*QQ;
        end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
yn=zeros(Nx,N);
for nx=1:Nx
  yn(nx,:)=N*real(ifft(ifftshift(un(:,nx)),N)); % reconstructed signal along the rod at discrete points
end
u1=N*real(ifft(ifftshift(Q(1,:)),N)); % reconstructed signal at the beginning of the rod (left side)
u2=N*real(ifft(ifftshift(Q(2,:)),N)); % reconstructed signal at the end of the rod (right side)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% signals at nodes
figure(2);
subplot(2,1,1); 
plot(t*1e3,real(u1));
xlabel('time [ms]');xlim([0 t(end)*1e3]);
subplot(2,1,2); 
plot(t*1e3,real(u2));
xlabel('time [ms]');xlim([0 t(end)*1e3]);
%%%%%%%%%%%%%%%
fig = gcf;set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[18 18 fig_width fig_height]); 
fig.PaperPositionMode   = 'auto';
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% animation
figure(3);
fig = gcf;set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[35 18 fig_width fig_height]); 
fig.PaperPositionMode   = 'auto';
xn=0:dx:L;
smax=max(max(abs(yn)));
for m =1:N
    plot(xn,real(yn(:,m)));
    axis([0 L -smax smax]);
    xlabel('x [m]');
    pause(0.01);
end
