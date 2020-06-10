function [Mask] = LowPassMaskGauss(Data,sigma)
% sigma - standard deviation

[rows,cols] = size(Data);
Mask = zeros(rows,cols);

% ideal filter mask 
a = 1;
b = rows/cols;

    for x = 1:cols
       for y = 1:rows
           d = sqrt( (((cols/2)-x)^2)*a + (((rows/2)-y)^2)*b);            
           Mask(y,x) = exp(-d^2/(2*sigma^2));
        end
    end
end