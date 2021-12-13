% https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/9554/versions/5/previews/numerical-tour/cs_fourier/index.html

% getd = @(p)path(path,p); 

% getd('toolbox_signal/');
% getd('toolbox_general/');

%We load an image.

n = 256;
%M = load_image('cameraman', n);
Orig_Image = im2double(imread('/home/pkudela/work/projects/nawa-bekker/ma-shm/data/processed/num/wavefield_dataset2_bottom_out/41_output/111_flat_shell_Vz_41_500x500bottom.png'));

%imshow(Orig_Image)
Orig_Image = imresize(Orig_Image,[n n]);
M = Orig_Image;
M = rescale(M);
% We compute a random mask that select a subset of frequencies. We enforce a few low frequency measurements that are not random.

% ratio of measurements
rho = .4;
% total number of measures
Q = round(rho*n^2);
% randomized values
A = rand(n,n);
[Y,X] = meshgrid(-n/2+1:n/2, -n/2+1:n/2);
A = A + (sqrt(X.^2+Y.^2)<20);
[v,I] = sort(A(:));
if v(1)<v(n*n)
    I = fliplr(I);
end
% mask
mask = zeros(n,n);
mask(I(1:Q)) = 1;
% We define centered, normalized, Fourier transforms. With Matlab, we use inlined callback function, and with Scilab, the functions are implemented in a separated file.

if using_matlab()
    Fourier = @(M)fftshift(fft2(M)/n);
    iFourier = @(F)real( ifft2( fftshift(F)*n ) );
end
% Build the compressed sensing acquisition function, CS(M) computes the CS measurements from M.

if using_matlab()
    S.type = '()'; S.subs = {find(mask==1)};
    CS = @(M)subsref(Fourier(M),S);
else
    global Ics;
    Ics = find(mask==1);
end
% The dual operator maps the measures to an image, by zero padding.

if using_matlab()
    iCS = @(y)iFourier(subsasgn(zeros(n,n),S,y));
end
% Set the noise level that perturbates the measures.

sigma = .01;
% Partial measurements with noise added.

y = CS(M) + sigma*randn(Q,1);
% Pseudo inverse reconstruction. Since the measurements is an orthogonal projection, its pseudo inverse is simply the dual operator.

% Mpseudo = real( ifft2( fftshift(Mpseudo)*n ) );
Mpseudo = iCS(y);
% We display the mask and the pseudo inverse reconstruction.

clf;
imageplot(mask, 'Fourier mask', 1,2,1);
imageplot(Mpseudo, 'Pseudo inverse reconstruction', 1,2,2);
