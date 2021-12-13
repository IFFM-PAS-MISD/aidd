function x = perform_cg(A,y,options)

// perform_cg - conjugate gradient
//  
//  x = perform_cg(A,y,options);
//
//  Copyright (c) 2009 Gabriel Peyre

options.null = 0;
tol = getoptions(options, 'tol', 1d-6);
maxit = getoptions(options, 'maxit', 100);

if norm(A-A', 'fro')<=1e-9
    //  symetric
    [x, fail, err, iter, res] = pcg(A,y,maxIter=maxit,tol=tol);
else
    // non symetric
    rstr = 10;
    [x,flag,err,iter,res] = gmres(A,y,rstr,tol,maxit);
end

endfunction