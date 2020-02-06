function A=frame2image(frame, filename)

% function converts frame of propagating waves (double values)
% to grayscale image: values [0, 255] and saves it to disc

% image is flipped due to coordinates in the left upper corner
% whereas in the frame coordinates are in the left lower corner
% normalization 1
 A=uint8(((1+ frame / max(max(abs(frame))) )/2 )*255); 
 % normalization 2
%  f_mean=mean(mean(frame));
%  f_std=std(std(frame));
%  frame=(frame-f_mean)/f_std; % data centering at 0 is not necessary because displacement field is centered at zero
%  A=frame/max(max(abs(frame)));
%  A=uint8(((1+A)/2)*255);
 % correct flipping
 A=flipud(A);
 imwrite(A,[filename,'.png'],'png');