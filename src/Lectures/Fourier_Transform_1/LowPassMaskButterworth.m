function [Mask] = LowPassMaskButterworth(Data,R,n)
% R - radius
% n - filtr order

[rows,cols] = size(Data);
Mask = zeros(rows,cols);

% ideal filter mask 
a = 1;
b = rows/cols;

    for x = 1:cols
       for y = 1:rows
           d = sqrt( (((cols/2)-x)^2)*a + (((rows/2)-y)^2)*b);            
            %Mask(y,x) = 1/(1+ ( -d/(2*R) )^(2*n) );
            Mask(y,x) = 1/(1+ ( -d/(R) )^(2*n) );
        end
    end
end