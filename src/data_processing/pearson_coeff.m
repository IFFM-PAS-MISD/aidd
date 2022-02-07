function [r] = pearson_coeff(x,y)
% PEARSON_COEFF   Calculate Pearson correlation coefficient 
% The correlation coefficient has the value of r=1 if the two wavefield 
% images are identical, and r=0 if they are completely uncorrelated
% 
% Syntax: [r] = pearson_coeff(x,y) 
% 
% Inputs: 
%    x - experimental wavefield, double, dimensions [m, n], Units: m/s 
%    y - numerical wavefield, double, dimensions [m, n], Units: m/s 
% 
% Outputs: 
%    r - Pearson correlation coefficient, double, Units: - 
% 
% Example: 
%    [r] = pearson_coeff(x,y) 
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

x=reshape(x,1,[]);
y=reshape(y,1,[]);

xm=mean(x);
ym=mean(y);

r=sum((x-xm).*(y-ym))/(sqrt(sum((x-xm).^2))*sqrt(sum((y-ym).^2)));

%---------------------- END OF CODE---------------------- 

% ================ [pearson_coeff.m] ================  
