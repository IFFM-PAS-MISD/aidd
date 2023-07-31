function [h] = Gauss(hsize,sigma)

    siz   = (hsize-1)/2;

    [x,y] = meshgrid(-siz(1):siz(1),-siz(1):siz(1));
    arg   = -(x.*x + y.*y)/(2*sigma*sigma);

    h     = exp(arg);
    h(h<eps*max(h(:))) = 0;

    sumh = sum(h(:));
    if sumh ~= 0,
           h  = h/sumh;
    end
end