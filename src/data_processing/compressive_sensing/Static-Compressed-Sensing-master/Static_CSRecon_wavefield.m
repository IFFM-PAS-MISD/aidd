%Sam Rothstein
%June 2020
% https://github.com/srothst1/Static-Compressed-Sensing

%Read Input
Orig_Image = im2double(imread('/home/pkudela/work/projects/nawa-bekker/ma-shm/data/processed/num/wavefield_dataset2_bottom_out/41_output/111_flat_shell_Vz_41_500x500bottom.png'));

imshow(Orig_Image)
Orig_Image = imresize(Orig_Image,[100 100]);
imshow(Orig_Image)

F_Transform_M = dctmtx(size(Orig_Image,1));
%F_Transform_M = dct2(size(Orig_Image,1));
Sparse_Image = F_Transform_M*Orig_Image*F_Transform_M';
imshow(Sparse_Image)

%Get Sparse Vector
S = size(Orig_Image);
N = 1000;
%N=25000;
%reshape from matrix into vector, stacking columns under one another from
%left to right
k_w=1;  
Sparse_Vector = zeros(S(1,1)*S(1,2),1);
for j = 1:S(1,2)
    for i = 1:S(1,1)
        Sparse_Vector(k_w,1)=Sparse_Image(i,j);
        k_w = k_w+1;
    end
end

%now we create a random gaussian sampling matrix
num_samples = 3500;
Random_G_Sam_m = rand(num_samples,10*N);
for x = 1:num_samples
    for y =1:10*N
        if (Random_G_Sam_m(x,y)+ 0.001 >= 1)
            Random_G_Sam_m(x,y) = 1;
        else
            Random_G_Sam_m(x,y) = 0;
        end
    end
end


Random_Sample_Y = Random_G_Sam_m * Sparse_Vector; 

%There are infinite solutions to (random matrix * transform) * solution =
%sample
%we must pick the best one

theta = Random_G_Sam_m * dctmtx(size(10*N,1));

%To pick the correct solution, we will use the OMP algorithm

solution = (cs_omp(Random_Sample_Y,theta,10*N)');

%convert image vector back to pixel matrix
%S = size(solution);                     
Solution_Pixel_Matrix = zeros(S(1,1),S(1,2)); %pixel matrix
k=1;
j=1;
for i = 1 : N
    if(k > 100)
        k=1;
        j=j+1;
    end
    Solution_Pixel_Matrix(k,j) = solution(i);
    k=k+1;
end

%Invert 2d discrete cosine transform
Solution_Pixel_Matrix = idct2(Solution_Pixel_Matrix); %use line 34 -> faster

imshow(Solution_Pixel_Matrix)

static_reconstruct_error = norm(Solution_Pixel_Matrix - Orig_Image) / norm(Orig_Image)
