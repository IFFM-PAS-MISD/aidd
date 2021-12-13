function y = cconv(x,h,d)

% cconv - spatial domain circular convolution
%
%   y = cconv(x,h,d);
%
%   Circular convolution along dimension |d|.
%
%   If p=length(h) is odd then h((p+1)/2) corresponds to the zero position.
%   If p is even, h(p/2) corresponds to the zero position.
%
%   Copyright (c) 2009 Gabriel Peyre

error('cconv has been renamed to cconvol');

if nargin<3
    d = 1;
end

if d==2
    y = permute( cconv( permute(x,[2 1 3]),h) ,[2 1 3]);
    return;
elseif d==3
    y = permute( cconv( permute(x,[3 2 1]),h) ,[3 2 1]);
    return;    
end

p = length(h);
if mod(p,2)==0
    pc = p/2;
else
    pc = (p+1)/2;
end

y = zeros(size(x));
for i=1:length(h)
    y = y + h(i)*circshift(x,i-pc);  
end