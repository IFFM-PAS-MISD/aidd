function delam_image_label(N,xCenter,yCenter,a,b,rotAngle,filename)
% DELAM_IMAGE_LABEL   create logic matrix showing delaminated region and convert it to image 
%    image can be used as label for machine learning 
% 
% Syntax: delam_image_label(N,xCenter,yCenter,a,b,rotAngle,filename)
% 
% Inputs: 
%    N - numer of points in adjacent grid, integer
%    xCenter -  delamination x coordinate in pixel scale, integer
%    yCenter -  delamination y coordinate in pixel scale, integer
%    rotAngle - delamination rotation angle [0:180), Units: deg
%    a - semi-major axis
%    b - semi-minor axis
%    filename - name of image label file, string
% 
% Outputs: 
% 
% Example: 
%    delam_image_label(N,xCenter,yCenter,a,b,rotAngle,filename)
% 
% Other m-files required: none 
% Subfunctions: none 
% MAT-files required: none 
% See also: 
% 

% Author: Pawel Kudela, D.Sc., Ph.D., Eng. 
% Institute of Fluid Flow Machinery Polish Academy of Sciences 
% Mechanics of Intelligent Structures Department 
% email address: pk@imp.gda.pl 
% Website: https://www.imp.gda.pl/en/research-centres/o4/o4z1/people/ 

%---------------------- BEGIN CODE---------------------- 

    % image/labels grid
    [X,Y]=meshgrid(0:N/(N-1):N,0:N/(N-1):N);
    %plot(X,Y,'k.');
    theta=rotAngle*pi/180;
    B=((X-xCenter)*cos(theta)+(Y-yCenter)*sin(theta)).^2/a^2 + ((X-xCenter)*sin(theta)-(Y-yCenter)*cos(theta)).^2/b^2;
    C=B<=1;
    X1=C.*X;
    X1((C<1))=NaN;
    Y1=C.*Y;
    Y1((C<1))=NaN;
    D=uint8(C)*255;
    D=flipud(D);
    imwrite(D,[filename,'.png'],'png');


%---------------------- END OF CODE---------------------- 

% ================ [delam_image_label.m] ================  
