% Initialize constants and variables
rng('default');                 % set RNG seed
N = 4096;                % length of signal
P = 2;                  % number of sinusoids
K = 1024;                 % number of measurements to take (N < L)

% Generate signal with P randomly spread sinosoids
% Note that a real-valued sinusoid has two peaks in the frequency domain
freq = randperm(N/2)-1;
freq = freq(1:P).';
n = 0:N-1;
t = linspace(0, 1, N);
x = (cos(2 * 97 * pi * t) + cos(2 * 777 * pi * t))';%

% Orthonormal basis matrix
Psi = dftmtx(N);
%Psi_inv = conj(Psi)/N;
Psi_inv = conj(Psi);
X = Psi*x;              % FFT of x(t)

% Plot signals
amp = 1.2*max(abs(x));
figure; subplot(5,1,1); plot(x); xlim([1 N]); ylim([-amp amp]);
title('$\mathbf{x(t)}$', 'Interpreter', 'latex')
subplot(5,1,2); plot(abs(X)); xlim([1 N]);
title('$|\mathbf{X(f)}|$', 'Interpreter', 'latex');

% Obtain K measurements
x_m = zeros(N,1);
q = randperm(N);
q = q(1:K);
x_m(q) = x(q);
subplot(5,1,3); plot(real(x_m)); xlim([1 N]);
title('Measured samples of $\mathbf{x(t)}$', 'Interpreter', 'latex');

A = Psi_inv(q, :);      % sensing matrix
%A = Psi(q, :);      % sensing matrix
y = A*X;                % measured values (see real(y))
% y1 = x(q, :);

% Perform Compressed Sensing recovery
x0 = A.'*y;
X_hat = l1eq_pd(x0, A, [], y, 1e-5);

subplot(5,1,4); plot(abs(X_hat)); xlim([1 N]);
title('$|\mathbf{\hat{X}(f)}|$', 'Interpreter', 'latex');

x_hat = real(Psi_inv*X_hat);    % IFFT of X_hat(f)

subplot(5,1,5); plot(x_hat*max(x)/max(x_hat)); xlim([1 N]);  ylim([-amp amp]);
title('$\mathbf{\hat{x}(t)}$', 'Interpreter', 'latex');

figure;
plot(t,x);
hold on;
plot(t,x_hat*max(x)/max(x_hat),'r');
legend('original','recovered');