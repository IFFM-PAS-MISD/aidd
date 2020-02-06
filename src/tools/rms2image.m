function A=frame2image(frame, filename)

% function converts frame of propagating waves (double values)
% to grayscale image: values [0, 255] and saves it to disc

% image is flipped due to coordinates in the left upper corner
% whereas in the frame coordinates are in the left lower corner
% normalization 1
maxx = max(max(frame));
minn = min(min(frame));
A = uint8(255/(maxx-minn)*(frame-minn)); 

 % correct flipping
 A=flipud(A);
 imwrite(A,[filename,'.png'],'png');