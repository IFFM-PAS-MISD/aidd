function [Mask] = HighPassMask(Data,R)

[rows,cols] = size(Data);
Mask = ones(rows,cols);

% ideal filter mask 
a = 1;
b = rows/cols;
 
    for x = 1:cols
       for y = 1:rows
            if sqrt( (((cols/2)-x)^2)*a + (((rows/2)-y)^2)*b) < R                
                Mask(y,x) = 0;
            end
        end
    end
end