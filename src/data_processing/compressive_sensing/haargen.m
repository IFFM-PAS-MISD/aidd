function [h]=haargen(N)
    % Generating Haar Matrix
    h=zeros(N,N); 
    h(1,1:N)=ones(1,N)/sqrt(N);
    for k=1:N-1 
        p=fix(log(k)/log(2)); 
        q=k-(2^p); 
        k1=2^p; t1=N/k1; 
        k2=2^(p+1); t2=N/k2; 
        for i=1:t2 
            h(k+1,i+q*t1)   = (2^(p/2))/sqrt(N); 
            h(k+1,i+q*t1+t2)=-(2^(p/2))/sqrt(N); 
        end 
    end