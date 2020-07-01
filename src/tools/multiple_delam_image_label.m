function multiple_delam_image_label(N,xCenter,yCenter,a,b,rotAngle,type,filename)
% DELAM_IMAGE_LABEL   create logic matrix showing delaminated region and convert it to image 
%    image can be used as label for machine learning 
% 
% Syntax: delam_image_label(N,xCenter,yCenter,a,b,rotAngle,filename)
% 
% Inputs: 
%    N - numer of points in adjacent grid, integer
%    xCenter -  delamination x coordinate in pixel scale, vector of integers
%    yCenter -  delamination y coordinate in pixel scale, vector of integers
%    rotAngle - delamination rotation angle [0:180), Units: deg
%    a - semi-major axis or rectangle length, vector of doubles
%    b - semi-minor axis or rectangle width, vector of doubles
%    type = 'ellipse' or 'rectangle'
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
    C=zeros(N,N);
    for k=1:length(xCenter)
        switch type(k)
            case 'ellipse'
                theta=rotAngle(k)*pi/180;
                B=((X-xCenter(k))*cos(theta)+(Y-yCenter(k))*sin(theta)).^2/a(k)^2 + ((X-xCenter(k))*sin(theta)-(Y-yCenter(k))*cos(theta)).^2/b(k)^2;
                I=B<=1;
                C(I)=1;
%                 X1=C.*X;
%                 X1((C<1))=NaN;
%                 Y1=C.*Y;
%                 Y1((C<1))=NaN;
            case 'rectangle'
                
                xv=[xCenter(k)-a(k)/2,xCenter(k)+a(k)/2,xCenter(k)+a(k)/2,xCenter(k)-a(k)/2,xCenter(k)-a(k)/2];
                yv=[yCenter(k)-b(k)/2,yCenter(k)-b(k)/2,yCenter(k)+b(k)/2,yCenter(k)+b(k)/2,yCenter(k)-b(k)/2];
                [in,on] = inpolygon(X,Y,xv,yv);
                C(in)=1;
                
        end
    end
    D=uint8(C)*255;
    D=flipud(D);
    imwrite(D,[filename,'.png'],'png');


%---------------------- END OF CODE---------------------- 

% ================ [delam_image_label.m] ================  
