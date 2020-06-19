%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% program rod
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clear all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ke=zeros(2,2);
ke1=zeros(2,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% materiaL data (steeL)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
e=210e9; % Young modulus (no damping)
%e=210e9*(1+0.01i); % Young modulus (damping)
%e=210e9*(1+0.0001i); % Young modulus (slightly damped)
nu=0.3; % Poisson ratio
ro=7860; % density
%L=1.5; % rod Length [m]
L=30; % rod Length [m]
b=0.01; % rod width [m]
h=0.01; % rod height [m]
a=b*h; % cross-section area
%J=1/6*(b*h*(b^2+h^2)); % polar moment of area; rod of rectangular cross-section 
%J=1/2*pi*r^4; % polar moment of area; rod of circular cross-section of radius r
eta=3e5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nx=1024; % number of points along the rod
dx=L/(Nx-1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tt=50e-3; % total time (semi-infinite rod)
%tt=0.5e-3; % total time (semi-infinite rod)
%tt=0.3e-3; % too short total time
%f_1=2e4;     % moduLation frequency of the  signal [Hz]
f_1=1e3;     % moduLation frequency of the  signal [Hz]
f_2=3*f_1;   % carrier frequency of the signal [Hz]
%f_2=5*f_1;   % carrier frequency of the signal [Hz]
t_t=1/f_1;   % totaL duration time of the excitation [s]
t_1=0e-4;    % excitation initiation time [s]
t_2=t_1+t_t; % excitation termination time [s]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fs=32*f_2;              %sampling frequency with oversampling factor 32
dt=1/fs;                  %time step
t=0:dt:tt;
%excitation signal (force vector)
f=zeros(length(t),1);
for n=1:length(t)
  if (t(n) >= t_1) && (t(n) <= t_2)
    f(n)=0.5*(1-cos(2*pi*f_1*(t(n)-t_2)))*sin(2*pi*f_2*(t(n)-t_1));
  end
end
figure(1);
plot(t,f);
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=2^nextpow2(length(f)); %number of points in FFT
t =[0:1:(N-1)]/fs; %recompute time index 
% amplitude spectrum
df=fs/N;                            %frequency resoLution
sampLeIndex = -N/2:N/2-1;   %ordered index for FFT plot (double-seded spectrum)
freq=sampLeIndex*df;               %t-axis index converted to ordered frequencies
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Q=zeros(2,N);
un=zeros(N,Nx);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
F = 1/N*fftshift(fft(f,N)); % force vector in the frequency domain
%figure;plot(freq,abs(F));
%return;
%u_recon = N*ifft(ifftshift(F),N); %reconstructed signal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%om=zeros(1:N/2+1,1);
om=zeros(N,1);
ne=zeros(1,2);
%for m=[1:N/2,N/2+2:N]
for m=[N/2+2:N]
      [m]
      om(m)=2*pi*freq(m);
      %k=om(m)*sqrt(ro/e);% elementary rod theory
      %k=sqrt((om(m)^2*ro)/(e+0.1*om(m)^2));% artificial dispersive rod
      k=sqrt( (om(m)^2*ro -i*om(m) * eta)/e );% damped rod % eta - viscous damping
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % spectraL rod eLement - dynamic stiffness
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      ke(1,1)=k*e*a*cot(k*L);
      ke(1,2)=-k*e*a*csc(k*L);
      ke(2,1)=-k*e*a*csc(k*L);
      ke(2,2)=k*e*a*cot(k*L);
%           delta=1-exp(-1i*2*k*L);
%           C=e*a*1i*k*L/(L*delta);
%           ke(1,1)=C*(1+exp(-1i*2*k*L));
%           ke(1,2)=C*(-2)*exp(-1i*k*L);
%           ke(2,1)=C*(-2)*exp(-1i*k*L);
%           ke(2,2)=C*(1+exp(-1i*2*k*L));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         %ke1(1,1)=e*a*1i*k; % throw-off eLement on the left side
         ke1(2,2)=e*a*1i*k; % throw-off eLement on the right side
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
            % semi infinite rod
            ne(1)=csc(k*L)*sin(k*(L-x)); % shape functions
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
figure(2);
subplot(2,1,1); plot(t,real(u1));
subplot(2,1,2); plot(t,real(u2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
figure(3);
xn=0:dx:L;
smax=max(max(abs(yn)));
for m =1:N
    m
    plot(xn,real(yn(:,m)));
    axis([0 L -smax smax]);
    pause(0.01);
end

