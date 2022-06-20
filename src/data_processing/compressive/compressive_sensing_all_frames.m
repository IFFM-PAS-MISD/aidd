%% compressive sensing applied to all frames
clear; clc; close all
load 389286p.mat; % Data, XYZ, time from SLDV measurements

%% Prepare output directories
% allow overwriting existing results if true
overwrite=true;
% retrieve model name based on running file and folder
currentFile = mfilename('fullpath');
[pathstr,name,ext] = fileparts( currentFile );
idx = strfind( pathstr,filesep );
foldername = pathstr(idx(end)+1:end); % name of folder
modelname = name; 
% prepare model output path
model_output_path = prepare_data_processing_paths('processed','exp',foldername,modelname);
%figure_output_path = prepare_figure_paths(foldername,modelname);

%% parameters
caxis_cut = 0.6;
fig_width =5; % figure widht in cm
fig_height=5; % figure height in cm
zoom_y = 300:400;
zoom_x = 256-50:256+50;
x_points = 128;
y_points = 128;
Lx=0.5; % plate length
Ly=0.5; % plate width
n = x_points*y_points;
No_of_measurement_points = 4000;
%No_of_measurement_points = [1024,3000,4000];
%No_of_measurement_points = [3000,4000];
cmap = 'parula'; % default matlab map
%cmap = 'jet';
mask_type = 1;
% mask_type = 1; % random mask
% mask_type = 2; % jitter mask
for points=No_of_measurement_points
figure_output_path = prepare_figure_paths(foldername,modelname);
% create random or jitter mask
switch mask_type
    case 1
        mask_name = 'random';
        if exist(['rand_XY_',num2str(x_points), 'x', num2str(y_points),'p','_siatka_',num2str(points)],'file')
            load(['rand_XY_',num2str(x_points), 'x', num2str(y_points),'p','_siatka_',num2str(points)]);
        else
            perm = randperm(n,points)'; %returns a row vector containing k unique integers selected randomly from 1 to n.
            x_mask=zeros(n,1);
            x_mask(perm)=1;
            save(['rand_XY_',num2str(x_points), 'x', num2str(y_points),'p','_siatka_',num2str(points)],'x_mask','perm');
        end
    case 2
        mask_name = 'jitter';
        if exist(['jitter_XY_',num2str(x_points), 'x', num2str(y_points),'p','_siatka_',num2str(points)],'file')
            load(['jitter_XY_',num2str(x_points), 'x', num2str(y_points),'p','_siatka_',num2str(points)]);
        else
            % my jitter mask 
            [perm,x_mask]=my_jitter_mask(points,x_points);
            save(['jitter_XY_',num2str(x_points), 'x', num2str(y_points),'p','_siatka_',num2str(points)],'x_mask','perm');
        end
end
% figure output path subfolders
% check if folder exist, if not create it
if ~exist([figure_output_path,filesep,mask_name], 'dir')
    mkdir([figure_output_path,filesep,mask_name]);
end
if ~exist([figure_output_path,filesep,mask_name,filesep,num2str(points),'p'], 'dir')
    mkdir([figure_output_path,filesep,mask_name,filesep,num2str(points),'p']);
end
figure_output_path = [figure_output_path,filesep,mask_name,filesep,num2str(points),'p',filesep];
% compute corresponding random coordinates
[rand_XY] = random_coordinates(perm,x_points,y_points,Lx,Ly);

% initialization
PSNR_metric = zeros(512,1);
SSIM_metric = zeros(512,1);
PEARSON_metric = zeros(512,1);
MSE_metric = zeros(512,1);
PSNR_metric_delam = zeros(512,1);
SSIM_metric_delam = zeros(512,1);
PEARSON_metric_delam = zeros(512,1);
MSE_metric_delam = zeros(512,1);
parameter_frames = zeros(512,1);
c = 0;
%% CS meaurement and recovery
for  p_frame = 1:512 % loop over frames
    
    c=c+1; 

    % ------------CS measurement-------------
    % interpolate random measurement points on uniform meshes
    reg_Frame = regInterp(Data,XYZ,x_points,y_points,p_frame); 
    ref_Frame = regInterp(Data,XYZ,512,512,p_frame);
    xmax = max(max(ref_Frame));
    s_zoom_max = max(max(ref_Frame(zoom_y,zoom_x)));
    
    % frame composed of random or jittered points for scatter plot
    OneDData = reshape(reg_Frame,[],1); 
    rand_Frame = OneDData(perm,:);
    Th=1e-4;% residual threshold 
    %Psi2 = dftmtx(n);% Fourier sparse basis
    Psi = zeros(n,n);
    tx_min = min(rand_XY(:,1));
    tx_max = max(rand_XY(:,1));
    Lx= tx_max - tx_min; 
    ty_min = min(rand_XY(:,2));
    ty_max = max(rand_XY(:,2));
    Ly = ty_max-ty_min;
    dx = (tx_max-tx_min)/(x_points-1);
    dy = (ty_max-ty_min)/(y_points-1);
    tx = tx_min:dx:tx_max;
    ty = ty_min:dy:ty_max;
    [qx,qy] = meshgrid(tx,ty);
    kx = linspace(-210,210,x_points);
    ky = linspace(-210,210,y_points);
    [Kx,Ky] = meshgrid(kx,ky);
    [Xi,Yi] = meshgrid(linspace(-0.25,0.25,x_points),linspace(-0.25,0.25,y_points));
    Qx1d = reshape(Xi,[n,1]);
    Qy1d = reshape(Yi,[n,1]);
    Kx1d = reshape(Kx,[n,1]);
    Ky1d = reshape(Ky,[n,1]);
    disp('calculation of Fourier basis');

    Psi = exp(-1i*( (Kx1d*Qx1d')/Lx +  (Ky1d*Qy1d')/Ly) );
    x = reshape(reg_Frame,[n,1]);
    X = (Psi)*x; % FFT of x
    %X2 = Psi2*x; % FFT of x
%     figure;
%     surf(fftshift(abs(reshape(X,[x_points,y_points])))); shading interp; view(2); axis square;
%     
%     figure;
%     surf(abs(reshape(X,[x_points,y_points]))); shading interp; view(2);axis square;
%     drawnow;
    
%      XF= fftshift(fft2(reg_Frame));
%      figure;
%      surf(abs(reshape(XF,[x_points,y_points]))); shading interp; view(2);axis square;
     
    %Psi = dctmtx(n);% DCT sparse basis
    Theta = Psi(perm, :); % random rows of Psi
    sigma = var(reshape(reg_Frame,x_points*y_points,1)-mean(reshape(reg_Frame,x_points*y_points,1)))*0.5;
    Th = sigma;
    opts = spgSetParms('optTol',1e-4,'iterations',1000);
    [xSparse,r,g,info] = spg_bpdn(Theta,rand_Frame,Th,opts);% reconstruction the sparse signal


    xRec = real(Psi' * xSparse); % calculate the original signal
    recon_image =  reshape(flipud(xRec), x_points,y_points); % reconstructed image\\

    %% plot data
    close all;
    
    % scatterinterpolant data
    int_method = 'natural' ; %linear, nearest
    ext_method = 'linear'; %none, nearest
    F = scatteredInterpolant(rand_XY(:,1),rand_XY(:,2),rand_Frame,int_method,ext_method);
    qz = F(qx,qy);
    
    % 6 in 1 plot
    figure('Position',[1 1 1920 1000])   
    set(gca,'LooseInset', max(get(gca,'TightInset'), 0));  

    subplot(2,3,1)
    scatter(rand_XY(:,1),rand_XY(:,2),5,rand_Frame);colormap(cmap);axis square;axis off;
    caxis([-xmax xmax])
    title(['Input data: ',num2str(x_points),'x', num2str(y_points),' unique points: ',num2str(size(perm,1))])

    subplot(2,3,2)
    imagesc(recon_image);colormap(cmap);axis square;axis off;
    caxis([-xmax xmax])
    title(['Reconstructed to ',num2str(x_points),'x', num2str(y_points)])

    subplot(2,3,3)
    imagesc(tx,ty,qz);
    colormap(cmap);axis square;axis off;
    caxis([-xmax xmax])
    title(['ScatteredInterpolant to ',num2str(x_points),'x', num2str(y_points)])

    subplot(2,3,4)
    imagesc(ref_Frame);colormap(cmap);axis square;axis off;
    caxis([-xmax xmax])
    title(['Reference'])

    subplot(2,3,5)
    int_recon_image = imresize(recon_image, [512 512], 'bicubic');
    imagesc(int_recon_image);colormap(cmap);axis square;axis off;
    caxis([-xmax xmax])
    title(['Interp. to 512x512 from rec. to ',num2str(x_points),'x', num2str(y_points)])

    subplot(2,3,6)
    intsct = imresize(reg_Frame, [512 512], 'bicubic');
    imagesc(intsct);colormap(cmap);axis square;axis off;
    caxis([-xmax xmax])
    title(['Interp. to 512x512 from regular grid of ',num2str(x_points),'x', num2str(y_points)])
    drawnow;
    
    print([figure_output_path,num2str(x_points), 'x', num2str(y_points),'p','_siatka_',num2str(points),'_klatka_',num2str(p_frame),'_',cmap,'_',mask_name,'.png'],'-dpng','-r300');
    
    % single plot - reference
    figure;
    imagesc(intsct);colormap(cmap);
    %caxis([-xmax xmax]); 
    run fig_param;
    caxis([caxis_cut*Smin,caxis_cut*Smax]);
    print([figure_output_path,'ref_',num2str(x_points), 'x', num2str(y_points),'p','_siatka_',num2str(points),'_klatka_',num2str(p_frame),'_',cmap,'_',mask_name,'.png'],'-dpng','-r600');
    
    % single plot - reconstructed
    figure;
    imagesc(int_recon_image);colormap(cmap);
    %caxis([-xmax xmax]);
    run fig_param;
    caxis([caxis_cut*Smin,caxis_cut*Smax]);
    print([figure_output_path,'recon_',num2str(x_points), 'x', num2str(y_points),'p','_siatka_',num2str(points),'_klatka_',num2str(p_frame),'_',cmap,'_',mask_name,'.png'],'-dpng','-r600');
    
    % close up at delamination reflection
    if (p_frame==110)
        figure('Position',[1 1 1920 1000])   
        set(gca,'LooseInset', max(get(gca,'TightInset'), 0));  
        subplot(1,2,1)
        imagesc(ref_Frame(zoom_y,zoom_x));colormap(cmap);axis square;axis off;
        title('Reference at delamination');

        subplot(1,2,2)
        imagesc(int_recon_image(zoom_y,zoom_x));colormap(cmap);axis square;axis off;
        title(['Reconstr. at delamination, ',num2str(points),' points']);
        drawnow;

        print([figure_output_path,'delam_',num2str(x_points), 'x', num2str(y_points),'p','_siatka_',num2str(points),'_klatka_',num2str(p_frame),'_',cmap,'_',mask_name,'.png'],'-dpng','-r600');
        
        % single figures for delamination region
        % reference
        figure;
        imagesc(ref_Frame(zoom_y,zoom_x));colormap(cmap);
        run fig_param;
        caxis([-s_zoom_max s_zoom_max]);
        drawnow;
        print([figure_output_path,'ref_delam_',num2str(x_points), 'x', num2str(y_points),'p','_siatka_',num2str(points),'_klatka_',num2str(p_frame),'_',cmap,'_',mask_name,'.png'],'-dpng','-r600');
        % reconstructed
        figure;
        imagesc(int_recon_image(zoom_y,zoom_x));colormap(cmap);
        run fig_param;
        caxis([-s_zoom_max s_zoom_max]);
        drawnow;
        print([figure_output_path,'recon_delam_',num2str(x_points), 'x', num2str(y_points),'p','_siatka_',num2str(points),'_klatka_',num2str(p_frame),'_',cmap,'_',mask_name,'.png'],'-dpng','-r600');
        
        % frame with rectangle showing delamination region
        % reference
        figure;
        imagesc(ref_Frame);colormap(cmap);
        rectangle('Position',[206 300 101 101]);
        run fig_param;
        caxis([caxis_cut*Smin,caxis_cut*Smax]);
        
        print([figure_output_path,'ref_rect_',num2str(x_points), 'x', num2str(y_points),'p','_siatka_',num2str(points),'_klatka_',num2str(p_frame),'_',cmap,'_',mask_name,'.png'],'-dpng','-r600');
        % reconstructed
        figure;
        imagesc(int_recon_image);colormap(cmap);
        rectangle('Position',[206 300 101 101]);
        run fig_param;
        caxis([caxis_cut*Smin,caxis_cut*Smax]);
        print([figure_output_path,'recon_rect_',num2str(x_points), 'x', num2str(y_points),'p','_siatka_',num2str(points),'_klatka_',num2str(p_frame),'_',cmap,'_',mask_name,'.png'],'-dpng','-r600');
        
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% quality metrics
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % peak signal-to-noise ratio
    [peaksnr] = psnr(int_recon_image, ref_Frame); 
    fprintf('The Peak-SNR value is %0.4f\n', peaksnr);
    PSNR_metric(c,1) = peaksnr;

    % Structural Similarity Index 
    [ssimval, ssimmap] = ssim(int_recon_image,ref_Frame); 
    fprintf('The ssim value is %0.4f.\n',ssimval);
    SSIM_metric(c,1) = ssimval;
    
    % Pearson correlation coefficient
    pcc=pearson_coeff(int_recon_image,ref_Frame);
    PEARSON_metric(c,1) = pcc;
    
    % mean-squared error (MSE) 
    mse=immse(int_recon_image,ref_Frame);
    MSE_metric(c,1) = mse;
    
    % Metrics on delamination reflection area

    % peak signal-to-noise ratio
    [peaksnr_delam] = psnr(int_recon_image(zoom_y,zoom_x), ref_Frame(zoom_y,zoom_x)); 
    PSNR_metric_delam(c,1) = peaksnr_delam;

    % Structural Similarity Index 
    [ssimval_delam, ssimmap] = ssim(int_recon_image(zoom_y,zoom_x),ref_Frame(zoom_y,zoom_x)); 
    SSIM_metric_delam(c,1) = ssimval_delam;
    
    % Pearson correlation coefficient
    pcc_delam=pearson_coeff(int_recon_image(zoom_y,zoom_x),ref_Frame(zoom_y,zoom_x));
    PEARSON_metric_delam(c,1) = pcc_delam;
    
    % mean-squared error (MSE) 
    mse_delam=immse(int_recon_image(zoom_y,zoom_x),ref_Frame(zoom_y,zoom_x));
    MSE_metric_delam(c,1) = mse_delam;
    
    parameter_frames(c,1) = c;
    
end
%%{
save([model_output_path,filesep,'frame_metrics_',num2str(x_points), 'x', num2str(y_points),'_points_',num2str(points),'_',mask_name],'PSNR_metric','SSIM_metric','PEARSON_metric','MSE_metric','PSNR_metric_delam','SSIM_metric_delam','PEARSON_metric_delam','MSE_metric_delam','parameter_frames');

figure;
plot(parameter_frames,MSE_metric,'m','LineWidth',1);
run font_param;
legend('MSE','Location','east','Fontsize',legend_font_size,'FontName','Times');
title(['grid ', num2str(x_points), 'x', num2str(y_points)]);
xlabel({'$N_f$'},'Fontsize',label_font_size,'interpreter','latex');
run fig_param2;
print([figure_output_path,'frame_MSE_',num2str(x_points), 'x', num2str(y_points),'_points_',num2str(points),'_',mask_name],'-dpng','-r300');

figure;
yyaxis left;
plot(parameter_frames,PSNR_metric,'LineWidth',1);
yyaxis right;
plot(parameter_frames,PEARSON_metric,'LineWidth',1);
run font_param;
legend('PSNR','PEARSON CC','Location','east','Fontsize',legend_font_size,'FontName','Times');
title(['CS: ', num2str(points),' points'],'Fontsize',title_font_size,'FontName','Times');
xlabel({'$N_f$'},'Fontsize',label_font_size,'interpreter','latex');
set(gcf,'color','white');
run fig_param2;
print([figure_output_path,'frame_metrics_',num2str(x_points), 'x', num2str(y_points),'_points_',num2str(points),'_',mask_name],'-dpng','-r600');

% metrics at delamination

figure;
plot(parameter_frames,SSIM_metric_delam,'b','LineWidth',1);
run font_param;
legend('SSIM','Location','east','Fontsize',legend_font_size);
title(['grid ', num2str(x_points), 'x', num2str(y_points),' delam'],'FontName','Times');
xlabel({'$N_f$'},'Fontsize',label_font_size,'interpreter','latex');
set(gcf,'color','white');
run fig_param2;
print([figure_output_path,'frame_SSIM_delam_',num2str(x_points), 'x', num2str(y_points),'_points_',num2str(points),'_',mask_name],'-dpng','-r300');

figure;
plot(parameter_frames,MSE_metric_delam,'m','LineWidth',1);
run font_param;
legend('MSE','Location','east','Fontsize',legend_font_size,'FontName','Times');
%title(['grid ', num2str(x_points), 'x', num2str(y_points),' delam']);
xlabel({'$N_f$'},'Fontsize',label_font_size,'interpreter','latex');
run fig_param2;
print([figure_output_path,'frame_MSE_delam_',num2str(x_points), 'x', num2str(y_points),'_klatka_',num2str(points),'_',mask_name],'-dpng','-r300');

figure;
yyaxis left;
plot(parameter_frames,PSNR_metric_delam,'LineWidth',1);
yyaxis right;
plot(parameter_frames,PEARSON_metric_delam,'LineWidth',1);
run font_param;
legend('PSNR','PEARSON CC','Location','southeast','Fontsize',legend_font_size,'FontName','Times');
title(['CS: ', num2str(points),' points (delam)'],'Fontsize',title_font_size,'FontName','Times');
xlabel({'$N_f$'},'Fontsize',label_font_size,'interpreter','latex');
run fig_param2;
print([figure_output_path,'frame_metrics_delam_',num2str(x_points), 'x', num2str(y_points),'_points_',num2str(points),'_',mask_name],'-dpng','-r600');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mask
%{
figure('Position',[1 1 1920 1000])   
    %set(gca,'LooseInset', max(get(gca,'TightInset'), 0));  
ax1=subplot(1,2,1);
    scatter(rand_XY(:,1),rand_XY(:,2),5,rand_Frame);colormap(ax1,cmap);axis square;axis off;
    caxis([-xmax xmax])
    title(['Input data: ',num2str(x_points),'x', num2str(y_points),' unique points: ',num2str(size(perm,1))])
ax2=subplot(1,2,2);    
    imagesc( reshape(x_mask,x_points,y_points)); axis equal;axis off; colormap(ax2,'gray');
    ax2.YDir = 'normal';
    switch mask_type
        case 1
            title('random mask');
        case 2
            title('jitter mask');
    end
%}    
figure;   
imagesc( reshape(x_mask,x_points,y_points)); axis equal;axis off; colormap('gray');
set(gca,'YDir','normal');
run font_param;
switch mask_type
    case 1
        title('random mask','FontName','Times','Fontsize',title_font_size);
    case 2
        title('jitter mask','FontName','Times','Fontsize',title_font_size);
end
run fig_param3;
print([figure_output_path,'mask_',mask_name,'_',num2str(x_points), 'x', num2str(y_points),'_points_',num2str(points)],'-dpng','-r600'); 
%%}
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%reg_Data = griddata(XYZ(:,1),XYZ(:,2),Data(:,p_frame),qx,qy,'cubic');

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
function [perm rand_Frame rand_XY] = point_random_selection(regFrame,x_points,y_points,points)
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

%%
function [rand_XY] = random_coordinates(perm,x_points,y_points,Lx,Ly)

tx_min = 0;
tx_max = Lx;
ty_min = 0;
ty_max = Ly;
dx = (tx_max-tx_min)/(x_points-1);
dy = (ty_max-ty_min)/(y_points-1);
tx = tx_min:dx:tx_max;
ty = ty_min:dy:ty_max;
[qx,qy] = meshgrid(tx,ty);
XY = [reshape(qx,[],1),reshape(qy,[],1)];
rand_XY = XY(perm,:);
end


