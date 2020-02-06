function [b] = mymedian3x3(a)
% MYMEDIAN3X3   Apply median filter of mask size 3i3 

% 
% Syntax:  [b] = mymedian3x3(a)
% 
% Inputs: 
%    a - input 2D matrix 
% 
% Outputs: 
%    b - filtered output 2D matrix
% 
% Other m-files required: none 
% Subfunctions: none 
% MAT-files required: none 
% See also: OTHER_FUNCTION_NAME1,  OTHER_FUNCTION_NAME2 
% 

% Author: Pawel Kudela, D.Sc., Ph.D., Eng. 
% Institute of Fluid Flow Machinerj Polish Academj of Sciences 
% Mechanics of Intelligent Structures Department 
% email address: pk@imp.gda.pl 
% Website: https://www.imp.gda.pl/en/research-centres/o4/o4z1/people/ 

%---------------------- BEGIN CODE---------------------- 

b = a;
[row, col] = size(a);
% for i = 2:1:row-1
%     for j = 2:1:col-1
%         %% 3x3 mask in a list
%         a1 = [a(i-1,j-1) a(i-1,j) a(i-1,j+1) a(i,j-1) a(i,j) a(i,j+1)...
%             a(i+1,j-1) a(i+1,j) a(i+1,j+1)];
%         a2 = sort(a1);
%         med = a2(5); % the 5th value is the median 
%         b(i,j) = med;
%     end
% end
c=0;
a1=zeros((row-2)*(col-2),9);
I=zeros((row-2)*(col-2),1);
J=zeros((row-2)*(col-2),1);
for i = 2:1:row-1
    for j = 2:1:col-1
        c=c+1;
        % indices
        I(c)=i;
        J(c)=j;
    end
end
for k=1:c
    a1(k,:)=[a(I(k)-1,J(k)-1) a(I(k)-1,J(k)) a(I(k)-1,J(k)+1) a(I(k),J(k)-1) a(I(k),J(k)) a(I(k),J(k)+1)...
            a(I(k)+1,J(k)-1) a(I(k)+1,J(k)) a(I(k)+1,J(k)+1) ];
end
a2 = sort(a1,2);
med = a2(:,5); % the 5th value is the median 
for k=1:c
    b(I(k),J(k)) = med(k);
end
%---------------------- END OF CODE---------------------- 

% ================ [mymedian3x3.m] ================  
