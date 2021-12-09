function z = cosamp(Phi,u,K,tol,maxiterations)
%------------------------------------------------------------------------------
%                          CoSaMP algorithm
% Algorithm as described in "CoSaMP: Iterative signal recovery from 
% incomplete and inaccurate samples" by Deanna Needell and Joel Tropp.
% Programmed by Chenhao
% version 1.0
% -  z: Solution found by the algorithm
% -  K : sparsity of z 
% -  Phi : measurement matrix
% -  u: measured vector
% -  tol : tolerance for approximation between successive solutions. 
% -  maxiterations: maximum number of iterations
% e.g. [A,y] = randmodu(x,20)
% e.g. z = cosamp(A,y,1,1e-5,20)
%------------------------------------------------------------------------------

% Initialization
z = zeros(size(Phi,2),1); %size(z) = N*1
v = u;
t = 1; 
numericalprecision = 1e-12;
T = [];
while (t <= maxiterations) && (norm(v)/norm(u) > tol) 
    y = abs(Phi'*v);
    [vals,z] = sort(y,'descend');
    Omega = find(y >= vals(2*K) & y > numericalprecision);
    T = union(Omega,T);
    b = pinv(Phi(:,T))*u;
    [vals,z] = sort(abs(b),'descend');
    K_indices = (abs(b) >= vals(K) & abs(b) > numericalprecision);
    T = T(K_indices);
    z = zeros(size(Phi,2),1);
    b = b(K_indices);
    z(T) = b;
    v = u - Phi(:,T)*b;
    t = t+1;
end