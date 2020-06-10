function duhatdt = rhsHeat2(uhat,kappa,a)
% right hand side of heat equation
% a - Thermal diffusivity constant
% kappa - wavenumbers; size [1,N]
% uhat - solution in the wavenumber domain at the time step t; size [N,1]
% duhatdt - derivative in respect to the time step t; size [N,1]

duhatdt = -a^2*(kappa.^2)'.*uhat;  % Linear and diagonal
