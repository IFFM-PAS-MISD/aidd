% https://www.mathworks.com/matlabcentral/fileexchange/50584-orthogonal-matching-pursuit-algorithm-omp

function [x] = OMP (K,y,A)
Res = y.' ;
[m,n] = size (A) ;
Q = zeros (m,K) ;
R = zeros (K,K) ;
Rinv = zeros (K,K) ;
w = zeros (m,K) ;
x = zeros (1,n) ;
for J = 1 : K
    
    %Index Search
    [V ,kkk] = max(abs(A'*Res)) ;
    kk (J) = kkk ;
    
    %Residual Update
    w (:,J) = A (:,kk (J)) ;
    for I = 1 : J-1
        if (J-1 ~= 0)
            R (I,J) = Q (:,I)' * w (:,J) ;
            w (:,J) = w (:,J) - R (I,J) * Q (:,I) ;
        end
    end
    R (J,J) = norm (w (:,J)) ;
    Q (:,J) = w (:,J) / R (J,J) ;
    Res = Res - (Q (:,J) * Q (:,J)' * Res) ;
  
end
%Least Squares
for J = 1 : K
    Rinv (J,J) = 1 / R (J,J) ;
    if (J-1 ~= 0)
        for I = 1 : J-1
            Rinv (I,J) = -Rinv (J,J) * (Rinv (I,1:J-1) * R (1:J-1,J)) ;
        end
    end
end
xx = Rinv * Q' * y.' ;
for I = 1 : K
    x (kk (I)) = xx (I) ;
end