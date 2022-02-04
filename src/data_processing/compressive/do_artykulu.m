%% initialization
clear; clc; close all
load 389286p.mat

%% parameters
x_points = 128;
y_points = 128;
p_frame = 110;
points = 3000;
%cmap = 'default';
cmap = 'jet';
for points = 6000%1000:1000:7000
    
%% Data preparation
reg_Frame = regInterp(Data,XYZ,x_points,y_points,p_frame);
ref_Frame = regInterp(Data,XYZ,512,512,p_frame);
xmax = max(max(ref_Frame));

% if points were selected use the same points for the next frame
exist rand_XY
if ans == 1
   if size(rand_XY,1) == points
       OneDData = reshape(reg_Frame,[],1);
       rand_Frame = OneDData(perm,:);  
   else
   [perm rand_Frame, rand_XY] = point_random_selection(reg_Frame,x_points,y_points,points);   
   end
else
    [perm rand_Frame, rand_XY] = point_random_selection(reg_Frame,x_points,y_points,points);
end

%% CS meaurement and recovery
% ------------CS measurement-------------
n = x_points*y_points;
perm = pointindexsearch(x_points,y_points,rand_XY(:,1:2)); % nearest point index serach
individual_perm = unique(perm);
%y1 = Data(:,p_frame);                               % compressed measurement
%-------------- reconstruct with orthogonal matching pursuit -----------

Th=1e-4;% residual threshold 
Psi = dftmtx(n);% Fourier sparse basis
%Psi = dctmtx(n);% DCT sparse basis
Theta = Psi(perm, :); % random rows of Psi

opts = spgSetParms('optTol',1e-4);
[xSparse,r,g,info] = spg_bpdn(Theta,rand_Frame,Th,opts);% reconstruction the sparse signal
xRec = real(Psi' * xSparse); % calculate the original signal
recon_image =  reshape(flipud(xRec), x_points,y_points); % reconstructed image\\

%% plot data
close all;
%jitter mask
% figure 
% jMask = zeros(x_points*y_points,1);
% jMask(perm) = 1;
% jMask = reshape(jMask, x_points,y_points);
% imagesc(jMask); colormap gray;axis equal;axis off;
% print('-djpeg','-r600',[num2str(x_points), 'x', num2str(y_points),'p','_siatka_',num2str(points),'_jMask.jpg']); 

% scatterinterpolant data
int_method = 'natural' ; %linear, nearest
ext_method = 'linear'; %none, nearest
F = scatteredInterpolant(rand_XY(:,1),rand_XY(:,2),rand_Frame,int_method,ext_method);
tx_min = min(rand_XY(:,1));
tx_max = max(rand_XY(:,1));
ty_min = min(rand_XY(:,2));
ty_max = max(rand_XY(:,2));
dx = (tx_max-tx_min)/(x_points-1);
dy = (ty_max-ty_min)/(y_points-1);
tx = tx_min:dx:tx_max;
ty = ty_min:dy:ty_max;
[qx,qy] = meshgrid(tx,ty);
qz = F(qx,qy);

% 6 in 1 plot
figure('Position',[1 1 1920 1000])   
set(gca,'LooseInset', max(get(gca,'TightInset'), 0));  

subplot(2,3,1)
scatter(rand_XY(:,1),rand_XY(:,2),5,rand_Frame);colormap(cmap);axis equal;axis off;
caxis([-xmax xmax])
title(['Input data: ',num2str(x_points),'x', num2str(y_points),' unique points: ',num2str(size(individual_perm,1))])
    
subplot(2,3,2)
imagesc(recon_image);colormap(cmap);axis equal;axis off;
caxis([-xmax xmax])
title(['Reconstructed to ',num2str(x_points),'x', num2str(y_points), ' points'])

subplot(2,3,3)
imagesc(tx,ty,qz);colormap(cmap);axis equal;axis off;
caxis([-xmax xmax])
title(['ScatteredInterpolant to ',num2str(x_points),'x', num2str(y_points), ' points'])

subplot(2,3,4)
imagesc(ref_Frame);colormap(cmap);axis equal;axis off;
caxis([-xmax xmax])
title(['Reference'])

subplot(2,3,5)
int_recon_image = imresize(recon_image, [512 512], 'bicubic');
imagesc(int_recon_image);colormap(cmap);axis equal;axis off;
caxis([-xmax xmax])
title(['Interp. to 512x512 from rec. to ',num2str(x_points),'x', num2str(y_points), ' points'])

subplot(2,3,6)
intsct = imresize(reg_Frame, [512 512], 'bicubic');
imagesc(intsct);colormap(cmap);axis equal;axis off;
caxis([-xmax xmax])
title(['Interp. to 512x512 from regular grid of ',num2str(x_points),'x', num2str(y_points), ' points'])

print('-djpeg','-r200',[num2str(x_points), 'x', num2str(y_points),'p','_siatka_',num2str(points),'_klatka_',num2str(p_frame),'.jpg']);

%close all;
%clear XYZ Data filename dx dy tx ty qz F recon_image Theta y1 Th opts xRec xSparse r g info Psi Psi_inv
end

%% FUNCTIONS
function perm = pointindexsearch(x_points,y_points,PQ)
% nearest point search
% perm - idexes of nearest points
% x_points, y_points - define number of points in search grid
% PQ - imput coordinates pairs

tx_min = min(PQ(:,1));
tx_max = max(PQ(:,1));
ty_min = min(PQ(:,2));
ty_max = max(PQ(:,2));
dx = (tx_max-tx_min)/(x_points-1);
dy = (ty_max-ty_min)/(y_points-1);
tx = tx_min:dx:tx_max;
ty = ty_min:dy:ty_max;
[qx,qy] = meshgrid(tx,ty);

P = [reshape(qx,[],1),reshape(qy,[],1)];
[perm,dist] = dsearchn(P,PQ);
D = mean(dist);

% figure
% plot(P(:,1),P(:,2),'ko')
% hold on
% plot(PQ(:,1),PQ(:,2),'*g')
% hold on
% plot(P(perm,1),P(perm,2),'*r')
% legend('Grid','Input','Nearest','Location','sw')
% title([num2str(x_points),'x',num2str(y_points),' points, avarage distance ',num2str(D)])
% xlim([0 0.05])
% ylim([0 0.05])
end

%function int = interpolation(recon_image)
%    int = imresize(recon_image, [512 512], 'bicubic');
%end

%% Regular grid interpolation
function reg_Data = regInterp(Data,XYZ,x_points,y_points,p_frame)
int_method = 'natural' ; %linear, nearest
ext_method = 'linear'; %none, nearest
            
F = scatteredInterpolant(XYZ(:,1),XYZ(:,2),Data(:,p_frame),int_method,ext_method);

rangex = abs(min(XYZ(:,1)))+abs(max(XYZ(:,1)));

tx_min = min(XYZ(:,1))+rangex*0.01;
tx_max = max(XYZ(:,1))-rangex*0.01;

rangey = abs(min(XYZ(:,2)))+abs(max(XYZ(:,2)));

ty_min = min(XYZ(:,2))+rangey*0.01;
ty_max = max(XYZ(:,2))-rangey*0.01;
            
dx = (tx_max-tx_min)/(x_points-1);
dy = (ty_max-ty_min)/(y_points-1);

tx = tx_min:dx:tx_max;
ty = ty_min:dy:ty_max;

[qx,qy] = meshgrid(tx,ty);
reg_Data = F(qx,qy);

% figure     
% fig = gcf;
% width = 2*8;
% height = 2*8;
% set(fig, 'Units','centimeters', 'Position',[10 10 width height]); % size 12cm by 8cm (1-column text)      
% fig.PaperPositionMode  = 'auto';
% set(gca,'LooseInset', max(get(gca,'TightInset'), 0));    
% imagesc(reg_Data)
end

%%
function [perm rand_Frame rand_XY] = point_random_selection(regFrame,x_points,y_points,points);
tx_min = 0;
tx_max = 0.5;
ty_min = 0;
ty_max = 0.5;
dx = (tx_max-tx_min)/(x_points-1);
dy = (ty_max-ty_min)/(y_points-1);
tx = tx_min:dx:tx_max;
ty = ty_min:dy:ty_max;
[qx,qy] = meshgrid(tx,ty);

OneDData = reshape(regFrame,[],1);
XY = [reshape(qx,[],1),reshape(qy,[],1)];
n = x_points*y_points;
perm = randperm(n,points)'; %returns a row vector containing k unique integers selected randomly from 1 to n.

rand_Frame = OneDData(perm,:);
rand_XY = XY(perm,:);
end
