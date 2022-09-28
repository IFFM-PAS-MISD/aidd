function [ mx] = mean2 ( x )

%   sx=sum(x,'all'); 
%   [m,n]=size(x);
%   nsum=m*n;
%   mx=sx./nsum;
mx=mean(mean(x));