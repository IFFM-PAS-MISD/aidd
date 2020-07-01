function IoU=intersect_over_union_fun(A,B)
% IoU=intersect_over_union_fun(A,B)
% calculates intersection over union IoU from binary matrices A,B of equal dimensions

% Intersection
I=A.*B;
% Union
U=A+B-A.*B;
% Intersection over Union
if(sum(sum(U))~=0)
     IoU=sum(sum(I))/sum(sum(U));
else
    IoU=0;
end