function [ind_jitter,x_jitter_mask]=my_jitter_mask(m,n1)
% MY_JITTER_MASK   Compute jitter indices and mask for square matrix 
% 
% Syntax: [ind_jitter,x_jitter_mask]=my_jitter_mask(m,n1) 
% 
% Inputs: 
%    m - number of measurement points (in compressed measuremetns), integer
%    n1 - number of points in targeted reconstructed mesh (n1xn1), integer
%    m < n1*n1
%
% Outputs: 
%    ind_jitter - indices of jitter mask, dimensions [m, 1]
%    x_jitter_mask - jitter mask in a vector form filled with ones
%                    representing measurement points, dimensions [n1*n1,1]
%                    mask can be reshaped to matrix and plotted
%    imagesc(reshape(x_jitter_mask,n1,n1)); axis equal; colormap gray;
% 
% Example: 
%    [ind_jitter,x_jitter_mask]=my_jitter_mask(4096,128)  
% 
% Other m-files required: none
% Subfunctions: none 
% MAT-files required:  none
% See also: 
% 
% Author: Pawel Kudela, D.Sc., Ph.D., Eng. 
% Institute of Fluid Flow Machinery Polish Academy of Sciences 
% Mechanics of Intelligent Structures Department 
% email address: pk@imp.gda.pl 
% Website: https://www.imp.gda.pl/en/research-centres/o4/o4z1/people/ 

%---------------------- BEGIN CODE---------------------- 

ind_jitter=zeros(m,1)+1;
x_jitter_mask=zeros(n1*n1,1);
if(m >= n1*n1)
    disp('error, wrong parameters m < n1*n1');  
    return;
end

a = -n1/sqrt(m)/2;
b = n1/sqrt(m)/2;
c=0;
for j=1:n1/sqrt(m):n1+n1/sqrt(m)
    for i=1:n1/sqrt(m):n1+n1/sqrt(m)
        c=c+1;
        eps1= a + (b-a)*rand(1);
        eps2= a + (b-a)*rand(1);
        ind_x = round(i + eps1);
        ind_y = round(j + eps2);
        if(ind_x <= 0)
            ind_x = 1;
        end
        if(ind_y <= 0)
            ind_y = 1;
        end
        if(ind_x > n1 )
            ind_x = n1;
        end
        if(ind_y > n1)
            ind_y = n1;
        end
        ind_jitter(c) = ind_x+(ind_y-1)*n1;
    end
end
ind_jitter=unique(ind_jitter);
I1=setdiff(1:n1*n1,ind_jitter)';
% add randomly the rest of points
l1=length(unique(ind_jitter));
if(l1<m)
    l2=m-l1;
    P = randperm(length(I1),l2);
    ind_jitter=[ind_jitter;I1(P)];
else
    % or remove randomly points
    l2=l1-m;
    P = sort(randperm(l1,l2),'descend');
    ind_jitter(P) =[];
end

x_jitter_mask(ind_jitter)=1;
%---------------------- END OF CODE---------------------- 

% ================ [my_jitter_mask.m] ================ 