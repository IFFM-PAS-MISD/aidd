function plot_graph(A,vertex1, str)

// plot_graph - plot a graph
//
// plot_graph(A,pos);
//
//  Copyright (c) 2010 Gabriel Peyre

if size(vertex1,1)>size(vertex1,2)
    vertex1 = vertex1';
end

if argn(2)<3
    str = 'k.-';
end

ij = spget(A>0);
i = ij(:,1);
j = ij(:,2);

I = find(i<=j); 
i = i(I); j = j(I);

x = [vertex1(1,i);vertex1(1,j)];
y = [vertex1(2,i);vertex1(2,j)];
plot(x,y,str);
h = gce();

endfunction