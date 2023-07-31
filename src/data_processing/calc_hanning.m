function w = calc_hanning(m,n)
w = .5*(1 - cos(2*pi*(1:m)'/(n+1))); 
end