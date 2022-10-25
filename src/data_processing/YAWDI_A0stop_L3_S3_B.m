%% Yet Another Wavenumber Damage Imaging (YAWDI)
% variant with A0 mode stop band filter 
% (better for amplitude tracking)
% additionally RMS of stop band filtered wavefield is calculated
% Algorithms needs at least two matrices as an input:
% 1. chirp wavefield
% 2. wavefield pass-banded around frequency fc
% Dimensions of matrices should be 512x512x512
% which corresponds to spatial dimensions (x,y) and time (t)
% Other inputs:
% time vector as "time" variable
% Plate dimensions as "WL" variable (two-element column vector)

% Author: Pawel Kudela, D.Sc., Ph.D., Eng. 
% Institute of Fluid Flow Machinery Polish Academy of Sciences 
% Mechanics of Intelligent Structures Department 
% email address: pk@imp.gda.pl 
% Website: https://www.imp.gda.pl/en/research-centres/o4/o4z1/people/ 
clear all;close all;   warning off;clc;
tic
load project_paths projectroot src_path;
%% Prepare output directories
% allow overwriting existing results if true
%overwrite=false;
overwrite=true;
interim_figs=true;
% retrieve model name based on running file and folder
currentFile = mfilename('fullpath');
[pathstr,name,ext] = fileparts( currentFile );
idx = strfind( pathstr,filesep );
modelfolder = pathstr(idx(end)+1:end); % name of folder
modelname = name; 
% prepare output paths
dataset_output_path = prepare_data_processing_paths('processed','exp',modelname);
figure_output_path = prepare_figure_paths(modelname);

radians_flag = false; % if true units of wanumbers [rad/m] if false [1/m]
%test_case=[2:5]; % select file numbers for processing
test_case=[2:7]; % select file numbers for processing
scaling_factor=0.5;
%% input for figures
Cmap = jet(256); 
Cmap2 = turbo; 
caxis_cut = 0.8;
fig_width =6; % figure widht in cm
fig_height=6; % figure height in cm
damage_outline=false;
%% Damage outline - ellipse
%    N - numer of points in adjacent grid, integer
%    xCenter -  delamination x coordinate 
%    yCenter -  delamination y coordinate 
%    rotAngle - delamination rotation angle [0:180), Units: deg
%    a - semi-major axis
%    b - semi-minor axis
rotAngle=90;
xCenter = 0.125;
yCenter = 0.125;
b=2/3*0.03792835351/2;
a=2/3*0.05689253027/2;
alpha=rotAngle*pi/180;
te=linspace(-pi,pi,50);
x=a*cos(te);
y=b*sin(te);
R  = [cos(alpha) -sin(alpha); ...
      sin(alpha)  cos(alpha)];
rCoords = R*[x ; y];   
xr = rCoords(1,:)';      
yr = rCoords(2,:)';     
delam1= [xr+xCenter-0.25,yr+yCenter-0.25];
%plot(delam1(:,1),delam1(:,2),'k:','LineWidth',0.5); axis square; xlim([0 0.5]);ylim([0 0.5]);
%% Input for signal processing
base_thickness = 3.9; % [mm] reference thicknes of the plate
WL = [0.5 0.5];
%mask_thr = 5;% percentage of points removed by filter mask, should be in range 0.5 - 5 
%mask_thr = 1;
%mask_thr = 0.5;
mask_thr = 0.7;
%PLT = 0.5;% if PLT = 0 do no create plots; 0<PLT<=0.5 only ERMSF plot ;0.5<PLT<=1 - RMS plots;  1<PLT - all plots
PLT = 0.6;
%threshold = 0.0018; % threshold for binarization
%% Processing parameters
Nx = 512;   % number of points after interpolation in X direction
Ny = 512;   % number of points after interpolation in Y direction
Nmed = 3;   % median filtering window size e.g. Nmed = 2 gives 2 by 2 points window size
selected_frames=240:4:440; % selected frames for Hilbert transform
selected_frames2= 200:450; % selected frames for RMS of A0 stop band filtered wavefield
N = 1024;% for zero padding

%% input for mask
if(radians_flag)
    mask_width_A0_1=200/2; % half wavenumber band width
    mask_width_A0_2=300/2;
else
    mask_width_A0_1=200/2/(2*pi); % half wavenumber band width
    mask_width_A0_2=300/2/(2*pi);
end
%%
% create path to the experimental raw data folder

raw_data_path = ['/pkudela_odroid_laser/aidd/data/raw/exp/L3_S3_B/'];

% create path to the numerical interim data folder
interim_data_path = fullfile( projectroot, 'data','interim','num', filesep );

% create path to the numerical processed data folder
processed_data_path = fullfile( projectroot, 'data','processed','num', filesep );

% full field measurements
list = {'chirp_interp','Data50','Data75','Data100','Data150','Data200','Data225'};
freq_list =[50,75,100,150,200,225]; % frequency list in kHz according to files above

if(~exist([dataset_output_path,filesep,'cart_mask_A0.mat'], 'file'))
    % load chirp signal
    disp('loading chirp signal');
    filename = list{1};
    load([raw_data_path,filename]); % Data, time, WL
    %Data=rot90(Data);
    [m,n,nft]=size(Data);
    Width=WL(1);
    Length=WL(2);

    disp('Transform to wavenumber-wavenumber-frequency domain');
    [KXKYF,kx_vec,ky_vec,f_vec] = spatial_to_wavenumber_wavefield_full(Data,Length,Width,time); % full size data (-kx:+kx,-ky:+ky,-f:+f)
    [m1,n1,nft1] = size(KXKYF);
    % filter 3D wavefield for mode separation (A0 mode extraction)
    [kx_grid,ky_grid]=ndgrid(kx_vec,ky_vec);
    %figure;
    %surf(kx_grid,ky_grid,squeeze(abs(KXKYF(m+1:end,n+1:end,100))));shading interp; view(2);axis square
    [f_grid,k_grid]=ndgrid(f_vec,kx_vec);
    %figure; surf(f_grid,k_grid,squeeze(abs(KXKYF(m+1:end,m+1,nft+1:end)))');shading interp; view(2)
    %figure; surf(squeeze(abs(KXKYF(:,m+1,nft+1:end))));shading interp; view(2)
    fmax=f_vec(end);
    kxmax=kx_vec(end);
    kymax=ky_vec(end);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% KX-KY-F slices
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if(interim_figs)
        if(radians_flag)   
            [mkx,mky,mf] = meshgrid(kx_vec,ky_vec,f_vec/1e3);
        else
            [mkx,mky,mf] = meshgrid(kx_vec/(2*pi),ky_vec/(2*pi),f_vec/1e3);
        end
        % maxkx = 1000/(2*pi);
        % maxky = 1000/(2*pi);
        maxkx = 200;
        maxky = 200;
        maxf = 300;
 
        for f = 1:length(freq_list)
                freq_slice = freq_list(f); % [kHz]
                xslice1 = []; yslice1 = []; zslice1 = freq_slice;
                xslice2 = 0; yslice2 = 0; zslice2 = [];
                figure;
                t=tiledlayout(2,1);
                %t.TileSpacing = 'tight';
                t.TileSpacing = 'none';
                t.Padding = 'tight';
                % Top plot
                ax1 = nexttile;
                h1 = slice(ax1,mkx,mky,mf,abs(KXKYF(:,:,end/2+1:end)),xslice2,yslice2,zslice2);
                set(h1,'FaceColor','interp','EdgeColor','none'); set(gcf,'Renderer','zbuffer');
                hold on;
                %ylabel({'$k_y$ [1/m]'},'Rotation',-37,'Fontsize',10,'interpreter','latex');% for 8cm figure
                ylabel({'$k_y$ [1/m]'},'Rotation',-38,'Fontsize',10,'interpreter','latex');% for 7.5cm figure
                xlabel({'$k_x$ [1/m]'},'Rotation', 15,'Fontsize',10,'interpreter','latex');
                zlabel({'$f$ [kHz]'},'Fontsize',10,'interpreter','latex')
                set(gca,'Fontsize',8,'linewidth',1);
                set(gca,'FontName','Times');
                grid(ax1,'off');
                view(3);
                lightangle(ax1,-45,45)
                lightangle(ax1,-45,45)
                colormap(Cmap2);
                %colormap turbo;
                line([0,0],[0,0],[0,max(f_vec)],'Color','y','LineWidth',1);
                line([-maxkx -maxkx],[-maxky  maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                line([-maxkx  maxkx],[ maxky  maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                line([ maxkx  maxkx],[ maxky -maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                line([ maxkx -maxkx],[-maxky -maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                xlim([-maxkx maxkx])
                ylim([-maxky maxky])
                zlim([0 maxf])
                box on; ax = gca; ax.BoxStyle = 'full';
                %view(-20,20)
                %view(-40,15)
                view(-30,50)
                Smax=max(max(max(abs(KXKYF(3:end,3:end,end/2+10:end)))));
                %caxis([0 0.5*Smax]);
                %caxis([0 0.4*Smax]);
                caxis([0 0.3*Smax]);
                
                % bottom plot
                ax2 = nexttile;
                h2 = slice(ax2,mkx,mky,mf,abs(KXKYF(:,:,end/2+1:end)),xslice1,yslice1,zslice1);
                set(h2,'FaceColor','interp','EdgeColor','none'); set(gcf,'Renderer','zbuffer');
                hold on;
                %ylabel({'$k_y$ [1/m]'},'Rotation',-37,'Fontsize',10,'interpreter','latex');% for 8cm figure
                ylabel({'$k_y$ [1/m]'},'Rotation',-38,'Fontsize',10,'interpreter','latex');% for 7.5cm figure
                xlabel({'$k_x$ [1/m]'},'Rotation', 15,'Fontsize',10,'interpreter','latex');
                zlabel({'$f$ [kHz]'},'Fontsize',10,'interpreter','latex')
                set(gca,'Fontsize',8,'linewidth',1);
                set(gca,'FontName','Times');
                view(3);
                
                colormap(Cmap2);
                %colormap turbo;
                lightangle(ax2,-45,45)
                lightangle(ax2,-45,45)
                line([0,0],[0,0],[0,max(f_vec)],'Color','y','LineWidth',1);
                line([-maxkx -maxkx],[-maxky  maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                line([-maxkx  maxkx],[ maxky  maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                line([ maxkx  maxkx],[ maxky -maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                line([ maxkx -maxkx],[-maxky -maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                xlim([-maxkx maxkx])
                ylim([-maxky maxky])
                zlim([freq_slice-0.01*freq_slice freq_slice+0.01*freq_slice]);
                % box off; ax = gca; ax.BoxStyle = 'full';
                %axis on;
                axis off;
                grid(ax2,'off');
                %title([num2str(freq_slice),' kHz'],'Fontsize',10,'interpreter','latex');
                text(-maxkx,maxky,freq_slice+0.01*freq_slice,[num2str(freq_slice),' kHz'],'HorizontalAlignment','left','Fontsize',10,'interpreter','latex');
                %view(-20,20)
                %view(-40,50)
                view(-30,50)
                [~,I]=min(abs(freq_slice-f_vec/1e3));
                Smax=max(max(max(abs(KXKYF(3:end,3:end,end/2+I)))));
                %caxis([0 0.8*Smax]);
                %caxis([0 0.7*Smax]);
                %caxis([0 0.6*Smax]);
                caxis([0 0.3*Smax]);
                set(gcf,'color','white');set(gca,'TickDir','out');
                %set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
                set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
                %set(gcf, 'Units','centimeters', 'Position',[10 10 8 10]);
                set(gcf, 'Units','centimeters', 'Position',[10 10 7.5 10]);
                % remove unnecessary white space
                set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));

                set(gcf,'PaperPositionMode','auto');
                drawnow;
                processed_filename = ['A0stop_KXKYF_chirp_',num2str(freq_slice),'_kHz']; % filename of processed .mat data
                print([figure_output_path,processed_filename],'-dpng', '-r600'); 
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % cartesian to polar            
    % 0-360deg
    dbeta = 360/(m1-1);
    beta = (dbeta:dbeta:(m1)*dbeta)-dbeta; 
    if(radians_flag)   
        [KXKYF_polar,~,k_radius] = cartesian_to_polar_wavefield_2pi_gridded2(KXKYF(:,:,nft+1:end),2*kxmax,2*kymax,beta);%fast - for data on regural 
    else
        [KXKYF_polar,~,k_radius] = cartesian_to_polar_wavefield_2pi_gridded2(KXKYF(:,:,nft+1:end),2*kxmax/(2*pi),2*kymax/(2*pi),beta);%fast - for data on regural 
    end
    %clear KXKYF;
    [m2,n2,nft2]=size(KXKYF_polar);
    %figure;surf(abs(squeeze(KXKYF_polar(:,:,100))));shading interp; view(2);      
    %figure;surf(abs(squeeze(KXKYF_polar(100,:,:))));shading interp; view(2);  


    %[TH,Radius] = meshgrid(beta*pi/180,linspace(0,k_radius,n2));
    % [Xk,Yk,Zk] = pol2cart(TH,Radius,KXKYF_polar(:,:,100));
    % figure;surf(Xk,Yk,abs(Zk)');shading interp;view(2);

    %figure;surf(abs(squeeze(KXKYF_polar_smooth(:,:,280))));shading interp; view(2);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% ridge picking
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % frequency range 15-130 kHz
    f_start = 15000;
    f_end = 130000;
    [~,f_start_ind] = min(abs(f_vec-f_start));
    [~,f_end_ind] = min(abs(f_vec-f_end));
    Ind = zeros(f_end_ind - f_start_ind+1,length(beta)/2);
    k_vec = linspace(0,k_radius,n2);
    k_A0_smooth = zeros(length(beta),length(f_vec));
    k_A0_f_selected = zeros(length(beta)/2,f_end_ind-f_start_ind+1);
    n_outliers = round(0.1*(f_end_ind - f_start_ind+1)); % number of data points to remove
    weighting = linspace(1,20,n2); % promote higher wavenumbers - for better extraction of A0 mode
    
    disp('Extracting A0 mode');
    
    %figure;hold on
     c=0;
    for n_freq = f_start_ind:f_end_ind
       c=c+1;
        for n_angle = 1:length(beta)/2 % 0:pi
            [kmax1,J1]=max(weighting.*abs(squeeze(KXKYF_polar(n_angle,:,n_freq))));
            [kmax2,J2]=max(weighting.*abs(squeeze(KXKYF_polar(n_angle+length(beta)/2,:,n_freq))));
%             cumsum1 = cumsum(abs(squeeze(KXKYF_polar(n_angle,:,n_freq))));
%             cumsum2 = cumsum(abs(squeeze(KXKYF_polar(n_angle+length(beta)/2,:,n_freq)))); % pi:2pi
%             cd1 = diff(cumsum1)/(k_vec(2)-k_vec(1));
%             cd2 = diff(cumsum2)/(k_vec(2)-k_vec(1));   
%             [cdmax1,I1] = max(cd1);
%             [cdmax2,I2] = max(cd2); % pi:2pi
%             I1=I1+1;
%             I2=I2+1;
            
            if(abs(J1-J2)< 8 ) 
                Ind(c,n_angle)=ceil((J1+J2)/2);
            else
                if(J1 > J2) % symmetrization (pick higher amplitude)
                    Ind(c,n_angle)=J1;
                    %plot(J1,cd1(J1),'bo');
                else
                    Ind(c,n_angle)=J2;
                    %plot(J2,cd2(J2),'ro');
                end
            end
        end
    end

    c=0;
    for n_freq = f_start_ind:f_end_ind
       c=c+1;    
        % fit polynomial
        [p3]=polyfit( beta(1:length(beta)/2)'*pi/180,k_vec(Ind(c,:)),8);
        yfit3 = polyval(p3,beta(1:length(beta)/2)'*pi/180);

        % remove outliers
        n_outliers = round(0.2*(length(yfit3))); % number of data points to remove
        [kd]=abs(yfit3'-k_vec(Ind(c,:)));
        [~,J]=sort(kd,'ascend');
        correct_data_ind=(sort(J(1:end-n_outliers)));

        % interpolation second pass
        [p4]=polyfit( beta(correct_data_ind)'*pi/180,k_vec(Ind(c,correct_data_ind)),8);
        yfit4 = polyval(p4,beta(1:length(beta)/2)'*pi/180);
        k_A0_f_selected(:,c) = yfit4;
       
%         figure;
%         polarplot(beta(1:length(beta)/2)'*pi/180,k_vec(Ind(81,:)),'ko');
%         hold on;
%         polarplot(beta(1:length(beta)/2)'*pi/180,yfit3,'r');
%         polarplot(beta(correct_data_ind)'*pi/180,k_vec(Ind(81,correct_data_ind)),'gx');
%         polarplot(beta(1:length(beta)/2)'*pi/180,yfit4,'c');
        
    end
%     k_A0_smooth = movmean([k_A0;k_A0(1:50,:)],50,1); % wrap around data for improved smoothing
%     k_A0_smooth = k_A0_smooth(1:length(beta),:);
%     figure;polarplot(beta(1:length(beta)/2)'*pi/180,k_A0_f_selected(:,92));
%     hold on;
%     polarplot(beta(1:length(beta)/2)'*pi/180,k_A0_f_selected(:,80),'r');
%     polarplot(beta(1:length(beta)/2)'*pi/180,k_A0_f_selected(:,70),'g');
%     polarplot(beta(1:length(beta)/2)'*pi/180,k_A0_f_selected(:,50),'c');
%     polarplot(beta(1:length(beta)/2)'*pi/180,k_A0_f_selected(:,5),'m');
%     polarplot(beta(1:length(beta)/2)'*pi/180,k_A0_f_selected(:,1),'k');
    
    %%%%%%%%%%%%%%%
    for n_angle = 1:length(beta)/2 % 0:pi
        % fit a curve
        [p5]=polyfit(f_vec(f_start_ind:f_end_ind),k_A0_f_selected(n_angle,:),2);
        yfit5 = polyval(p5,f_vec(f_start_ind:f_end_ind));
        [p6]=polyfit(f_vec(f_end_ind-20:f_end_ind),yfit5(end-20:end),1);
        yfit6 = polyval(p6,f_vec(f_end_ind-20:f_end_ind));
        %vq = interp1([0,f_vec(f_start_ind:f_end_ind)],[0,yfit5] ,f_vec,'spline','extrap');
        vq = interp1([0,f_vec(f_start_ind:f_end_ind)],[0,yfit5(1:end-20),yfit6(end-19:end)],f_vec,'spline','extrap');
        k_A0_smooth(n_angle,:) = vq; 
    end
    % symmetry
    k_A0_smooth(length(beta)/2+1:length(beta),:) = k_A0_smooth(1:length(beta)/2,:);
%     figure;
%     polarplot(beta(1:length(beta))'*pi/180,k_A0_smooth(:,f_start_ind+81));
%     hold on;
%     polarplot(beta(1:length(beta)/2)'*pi/180,k_A0_smooth_f_selected(:,81));
%     figure;
%     plot(f_vec/1e3,k_A0_smooth(100,:))
%     hold on;
%     plot(f_vec(f_start_ind:f_end_ind)/1e3,k_A0_f_selected(100,:),'r');
%     figure;
%     for i=1:4:50
%     xx = k_A0_f_selected(:,i).* cos(beta(1:length(beta)/2)'*pi/180); 
%     yy = k_A0_f_selected(:,i).* sin(beta(1:length(beta)/2)*pi/180)';
%     plot3(xx,yy,repmat(f_vec(f_start_ind+i)/1e3,[length(beta)/2 1]),'ko'); hold on;
%     axis equal;
%     end
%     hold on;
%     for i=1:4:50
%     xx = k_A0_smooth(1:length(beta)/2,f_start_ind+i).* cos(beta(1:length(beta)/2)'*pi/180); 
%     yy = k_A0_smooth(1:length(beta)/2,f_start_ind+i).* sin(beta(1:length(beta)/2)*pi/180)';
%     plot3(xx,yy,repmat(f_vec(f_start_ind+i)/1e3,[length(beta)/2 1]),'rd'); hold on;
%     axis equal;
%     end
    clear k_A0_f_selected;
    clear KXKYF_polar;
%     figure;
%     for n_angle = 1:length(beta)
%         plot(f_vec/1e3,k_A0_smooth(n_angle,:),'r');hold on;
%     end
%     figure;
%     for j = 1:length(f_vec)
%         plot3( k_A0_smooth(:,j).*cos(beta'*pi/180),k_A0_smooth(:,j).*sin(beta'*pi/180), repmat(f_vec(j),[length(beta),1]),'r.'); hold on;
%     end
%     figure;
%     for j = 1:length(f_vec)
%         plot3( k_A0_smooth(1:10:end,j).*cos(beta(1:10:end)'*pi/180),k_A0_smooth(1:10:end,j).*sin(beta(1:10:end)'*pi/180), repmat(f_vec(j),[length(beta(1:10:end)),1]),'r.'); hold on;
%     end
%     figure;surf(k_A0_smooth);shading interp;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% thickness sensitivity curves
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    thickness_range = linspace(0.1,3,128); %0.1:3; % percentage of base_thickness, gives 0.2:6 mm for 2 mm thick plate
    thickness_sensitivities = zeros(length(beta),length(thickness_range),length(freq_list));
    for n_angle = 1:length(beta)
        kq=zeros(length(freq_list),length(thickness_range));
        for f = 1:length(freq_list)

            freq_slice = freq_list(f); % [kHz]

            [~,f_ind] = min(abs(f_vec/1e3 - freq_slice));
            thickness = thickness_range*base_thickness; 
            for j=1:length(thickness_range)
                kq(f,j)=interp1(f_vec*thickness_range(j),k_A0_smooth(1,:),freq_slice*1e3,'linear','extrap');
            end
        end
        thickness_sensitivities(n_angle,:,:)=kq';
    end
    if(interim_figs)
        figure;
        plot(thickness(1:4:end),kq(1,1:4:end),'kd-','MarkerFaceColor','k','MarkerSize',2);hold on;
        plot(thickness(1:4:end),kq(2,1:4:end),'rv-','MarkerFaceColor','r','MarkerSize',2);
        plot(thickness(1:4:end),kq(3,1:4:end),'go-','MarkerFaceColor','g','MarkerSize',2);
        plot(thickness(1:4:end),kq(4,1:4:end),'bd-','MarkerFaceColor','b','MarkerSize',2);
        xlabel('h [mm]','FontSize',10,'FontName','Times New Roman');
        ylabel('k [1/m]','FontSize',10,'FontName','Times New Roman');
        legend('50 kHz', '75 kHz','100 kHz','150 kHz','FontSize',10);
        set(gca,'FontSize',10);
        set(gcf,'color','white');set(gca,'TickDir','out');
        %set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
        set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
        %set(gcf, 'Units','centimeters', 'Position',[10 10 8 10]);
        set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]);
        % remove unnecessary white space
        set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));

        set(gcf,'PaperPositionMode','auto');
        drawnow;
        processed_filename = ['A0stop_Thickness_sensitivities']; % filename of processed .mat data
        print([figure_output_path,processed_filename],'-dpng', '-r600'); 
    end
    save([dataset_output_path,filesep,'thickness_sensitivities'],'thickness','thickness_sensitivities');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% mask A0 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    disp('creating mask for A0 mode extraction');
    mask_width = [linspace(0,mask_width_A0_1,5),linspace(mask_width_A0_1,mask_width_A0_2,length(f_vec)-5)];
    wavenumber_lower_bound = (k_A0_smooth' - repmat(mask_width',1,length(beta)))'; 
    wavenumber_upper_bound = (k_A0_smooth' + repmat(mask_width',1,length(beta)))';
    if(interim_figs)
        figure;
        surf(k_A0_smooth);shading interp; hold on;
        surf(wavenumber_lower_bound);shading interp; 
        surf(wavenumber_upper_bound);shading interp; 
        drawnow;
    end

    polar_mask_A0 = zeros(length(beta),n2,length(f_vec));
    ka=linspace(0,k_radius,n2);
    for n_angle = 1:length(beta)
        for n_freq = 1:length(f_vec)
            J3=(ka>=wavenumber_lower_bound(n_angle,n_freq));
            ind3 = find(J3);
            J4=(ka<=wavenumber_upper_bound(n_angle,n_freq));
            J=J3.*J4;
            ind = find(J);
            polar_mask_A0(n_angle,ind,n_freq) = hann(length(ind));
            clear ind;
        end
    end
%     figure;
%     surf(squeeze(polar_mask_A0(end,:,:)));shading interp;
    disp('converting mask from cylindrical to cartesian coordinates');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % convert back to cartesian coordinates by using linear interpolation
    % scattered interpolation - slow
    [TH,Radius] = meshgrid(beta*pi/180,ka);
    [Xk,Yk,Zk] = pol2cart(TH',Radius',polar_mask_A0);
    %figure;surf(Xk,Yk,Zk(:,:,100));shading interp;view(2);
    
     % 
    if(radians_flag)   
        [XI,YI] = meshgrid(linspace(-kxmax,kxmax,m1),linspace(-kymax,kymax,n1));
    else
        [XI,YI] = meshgrid(linspace(-kxmax/(2*pi),kxmax/(2*pi),m1),linspace(-kymax/(2*pi),kymax/(2*pi),n1));
    end
    %{   
    cart_mask_A0 = zeros(n1,m1,2*length(f_vec));
    for n_freq = 1:length(f_vec)
        [n_freq length(f_vec)]
        F = scatteredInterpolant(reshape(Xk,[],1),reshape(Yk,[],1),reshape(squeeze(Zk(:,:,n_freq)),[],1),'linear','none'); % requires ndgrid format; no extrapolation
        %F = scatteredInterpolant(reshape(Xk,[],1),reshape(Yk,[],1),reshape(squeeze(Zk(:,:,n_freq)),[],1),'nearest','none');    
        cart_mask_A0(:,:,length(f_vec)+n_freq)=F(XI,YI);
    end
    cart_mask_A0(:,:,1:length(f_vec)) = flip(cart_mask_A0(:,:,length(f_vec)+1:end),3); % flip for neagative frequencies
    cart_mask_A0(isnan(cart_mask_A0))=0;
    save([dataset_output_path,'cart_mask_A0'],'cart_mask_A0');
    %}          
    %% convert back to cartesian coordinates 
    % Delaunay triangulation approach (nearest) - fastest
    %{
    cart_mask_A0_ = zeros(n1,m1,2*length(f_vec));
    DT = delaunayTriangulation(reshape(Xk,[],1), reshape(Yk,[],1));
    vi = nearestNeighbor(DT,reshape(XI,[],1), reshape(YI,[],1));
    for n_freq = 1:length(f_vec)
        Zkf=reshape(squeeze(Zk(:,:,n_freq)),[],1);
        Vq = Zkf(vi);
        cart_mask_A0_(:,:,length(f_vec)+n_freq) = reshape(Vq,n1,m1);
    end
    cart_mask_A0_(:,:,1:length(f_vec)) = flip(cart_mask_A0_(:,:,length(f_vec)+1:end),3); % flip for neagative frequencies
    cart_mask_A0_(isnan(cart_mask_A0_))=0;
    %}
    % Delaunay triangulation approach (linear interpolation) - fast
    % General shape is ok but there are some large errors
    
    cart_mask_A0 = zeros(n1,m1,2*length(f_vec));
    DT = delaunayTriangulation(reshape(Xk,[],1), reshape(Yk,[],1));
    [ti,bc] = pointLocation(DT,reshape(XI,[],1), reshape(YI,[],1));
    % problem with corner nodes
    nan_ind = find(isnan(ti));
    ti(nan_ind) =1;
    bc(nan_ind,:) = 0;
    for n_freq = 1:length(f_vec)
        %[n_freq, length(f_vec)]
        Zkf=reshape(squeeze(Zk(:,:,n_freq)),[],1); 
        triVals = Zkf(DT(ti,:));
        Vq = dot(bc',triVals')';
        cart_mask_A0(:,:,length(f_vec)+n_freq) = reshape(Vq,n1,m1);
    end
    cart_mask_A0(:,:,1:length(f_vec)) = flip(cart_mask_A0(:,:,length(f_vec)+1:end),3); % flip for neagative frequencies
    cart_mask_A0(isnan(cart_mask_A0))=0;
    cart_mask_A0(:,:,length(f_vec))=0;
    cart_mask_A0(:,:,length(f_vec)+1)=0;
    cart_mask_A0=1-cart_mask_A0;
    save([dataset_output_path,filesep,'cart_mask_A0'],'cart_mask_A0','kx_vec','ky_vec','f_vec','-v7.3');
    % figure;
    % surf(reshape(Vq,n1,m1));shading interp; 
    % figure;
    % surf(squeeze(cart_mask_A0(:,:,100)));shading interp;
    % figure;
    % surf(squeeze(cart_mask_A0_(:,:,100)));shading interp;
else
    disp('Loading cartesian mask for A0 mode');
    load([dataset_output_path,filesep,'cart_mask_A0']);
    load([dataset_output_path,filesep,'thickness_sensitivities']);
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% mask slices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(radians_flag) 
    [mkx,mky,mf] = meshgrid(kx_vec,ky_vec,f_vec/1e3);
else
    [mkx,mky,mf] = meshgrid(kx_vec/(2*pi),ky_vec/(2*pi),f_vec/1e3);
end
% maxkx = 1000/(2*pi);
% maxky = 1000/(2*pi);
maxkx = 200;
maxky = 200;
maxf = 300;
if(interim_figs)
    for f = 1:length(freq_list)

        freq_slice = freq_list(f); % [kHz]
        xslice1 = []; yslice1 = []; zslice1 = freq_slice;
        xslice2 = 0; yslice2 = 0; zslice2 = [];

        figure;
        t=tiledlayout(2,1);
        %t.TileSpacing = 'tight';
        t.TileSpacing = 'none';
        t.Padding = 'tight';
        % Top plot
        ax1 = nexttile;

        h1 = slice(ax1,mkx,mky,mf,cart_mask_A0(:,:,end/2+1:end),xslice2,yslice2,zslice2);

        set(h1,'FaceColor','interp','EdgeColor','none'); set(gcf,'Renderer','zbuffer');
        hold on;
        %ylabel({'$k_y$ [1/m]'},'Rotation',-37,'Fontsize',10,'interpreter','latex');% for 8cm figure
        ylabel({'$k_y$ [1/m]'},'Rotation',-38,'Fontsize',10,'interpreter','latex');% for 7.5cm figure
        xlabel({'$k_x$ [1/m]'},'Rotation', 15,'Fontsize',10,'interpreter','latex');
        zlabel({'$f$ [kHz]'},'Fontsize',10,'interpreter','latex')
        set(gca,'Fontsize',8,'linewidth',1);
        set(gca,'FontName','Times');
        grid(ax1,'off');
        view(3);
        lightangle(ax1,-45,45)
        lightangle(ax1,-45,45)
        colormap (gray)
        line([0,0],[0,0],[0,max(f_vec)],'Color','y','LineWidth',1);
        line([-maxkx -maxkx],[-maxky  maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
        line([-maxkx  maxkx],[ maxky  maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
        line([ maxkx  maxkx],[ maxky -maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
        line([ maxkx -maxkx],[-maxky -maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');

        xlim([-maxkx maxkx])
        ylim([-maxky maxky])
        zlim([0 maxf])
        box on; ax = gca; ax.BoxStyle = 'full';
        %view(-20,20)
        %view(-40,15)
        view(-30,50)

        % bottom plot
        ax2 = nexttile;

        h2 = slice(ax2,mkx,mky,mf,cart_mask_A0(:,:,end/2+1:end),xslice1,yslice1,zslice1);

        set(h2,'FaceColor','interp','EdgeColor','none'); set(gcf,'Renderer','zbuffer');
        hold on;
        %ylabel({'$k_y$ [1/m]'},'Rotation',-37,'Fontsize',10,'interpreter','latex');% for 8cm figure
        ylabel({'$k_y$ [1/m]'},'Rotation',-38,'Fontsize',10,'interpreter','latex');% for 7.5cm figure
        xlabel({'$k_x$ [1/m]'},'Rotation', 15,'Fontsize',10,'interpreter','latex');
        zlabel({'$f$ [kHz]'},'Fontsize',10,'interpreter','latex')
        set(gca,'Fontsize',8,'linewidth',1);
        set(gca,'FontName','Times');

        view(3);
%         lightangle(-45,45)
%         lightangle(-45,45)
        colormap(gray)
        line([0,0],[0,0],[0,max(f_vec)],'Color','y','LineWidth',1);
        line([-maxkx -maxkx],[-maxky  maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
        line([-maxkx  maxkx],[ maxky  maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
        line([ maxkx  maxkx],[ maxky -maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
        line([ maxkx -maxkx],[-maxky -maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');

        xlim([-maxkx maxkx])
        ylim([-maxky maxky])
        zlim([freq_slice-0.01*freq_slice freq_slice+0.01*freq_slice]);
        % box off; ax = gca; ax.BoxStyle = 'full';
        %axis on;
        axis off;
        grid(ax2,'off');
        %title([num2str(freq_slice),' kHz'],'Fontsize',10,'interpreter','latex');
        text(-maxkx,maxky,freq_slice+0.01*freq_slice,[num2str(freq_slice),' kHz'],'HorizontalAlignment','left','Fontsize',10,'interpreter','latex');
        %view(-20,20)
        %view(-40,50)
        view(-30,50)


        set(gcf,'color','white');set(gca,'TickDir','out');
        %set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
        set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
        %set(gcf, 'Units','centimeters', 'Position',[10 10 8 10]);
        set(gcf, 'Units','centimeters', 'Position',[10 10 7.5 10]);
        % remove unnecessary white space
        set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));

        set(gcf,'PaperPositionMode','auto');
        drawnow;
        processed_filename = ['A0stop_mask_A0_',num2str(freq_slice),'_kHz']; % filename of processed .mat data
        print([figure_output_path,processed_filename],'-dpng', '-r600'); 
    end
end
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Yet another wavenumber damage imaging (YAWDI)
disp('Yet another wavenumber damage imaging (YAWDI)');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Thickness_map_avg = zeros(Nx,Ny); % for average thickness map over frequencies
RMS_A0stop_avg = zeros(Nx,Ny); 
folder  = raw_data_path;
nFile   = length(test_case);
success = false(1, nFile);
for k = test_case
    filename = list{k};
    processed_filename = ['A0stop_RMS_wavenumbers_selected_',filename]; 
    % check if already exist
    if(overwrite||(~overwrite && ~exist([figure_output_path,processed_filename,'.png'], 'file')))
       
            % load raw experimental data file
            disp('loading data');
            load([raw_data_path,filename]); % Data, time, WL
            [nx,ny,nft] = size(Data);
            Width=WL(1);
            Length=WL(2);
            % beta range for cylindrical coordinates
            % 0-360deg
            dbeta = 360/(4*nx-1);
            beta = (dbeta:dbeta:(4*nx)*dbeta)-dbeta;  
            %             0-360deg-dbeta
            %              dbeta = 360/(4*nx);
            %             beta = (dbeta:dbeta:(4*nx)*dbeta)-dbeta;
            Rmax = radius_max(beta*pi/180,Length,Width);
            %nx=512;ny=512;nft=512;
            %% PROCESS DATA
            fprintf('Processing:\n%s\n',filename);
            disp('3D FFT filtering - A0 mode separation');
            disp('Transform to wavenumber-wavenumber-frequency domain');
            [KXKYF,kx_vec,ky_vec,f_vec] = spatial_to_wavenumber_wavefield_full(Data,Length,Width,time); % full size data (-kx:+kx,-ky:+ky,-f:+f)
            clear Data;
            KXKYF_A0 = KXKYF.*cart_mask_A0;
            % inverse Fourier transform for pure A0 wavefield
            Data = ifftn(ifftshift(KXKYF_A0),'symmetric'); % wavefield A0
            Data = Data(1:nx,1:ny,1:nft);
            %% plot intermediate results for checking masks
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% KX-KY-F slices
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if(interim_figs)
                if(radians_flag) 
                    [mkx,mky,mf] = meshgrid(kx_vec,ky_vec,f_vec/1e3);
                else
                   [mkx,mky,mf] = meshgrid(kx_vec/(2*pi),ky_vec/(2*pi),f_vec/1e3);  
                end
                freq_slice = freq_list(k-1); % [kHz]
                % maxkx = 1000/(2*pi);
                % maxky = 1000/(2*pi);
                maxkx = 200;
                maxky = 200;
                maxf = 300;
                xslice1 = []; yslice1 = []; zslice1 = freq_slice;
                xslice2 = 0; yslice2 = 0; zslice2 = [];

                figure;
                t=tiledlayout(2,1);
                %t.TileSpacing = 'tight';
                t.TileSpacing = 'none';
                t.Padding = 'tight';
                % Top plot
                ax1 = nexttile;
                h1 = slice(ax1,mkx,mky,mf,abs(KXKYF(:,:,end/2+1:end)),xslice2,yslice2,zslice2);
                set(h1,'FaceColor','interp','EdgeColor','none'); set(gcf,'Renderer','zbuffer');
                hold on;
                %ylabel({'$k_y$ [1/m]'},'Rotation',-37,'Fontsize',10,'interpreter','latex');% for 8cm figure
                ylabel({'$k_y$ [1/m]'},'Rotation',-38,'Fontsize',10,'interpreter','latex');% for 7.5cm figure
                xlabel({'$k_x$ [1/m]'},'Rotation', 15,'Fontsize',10,'interpreter','latex');
                zlabel({'$f$ [kHz]'},'Fontsize',10,'interpreter','latex')
                set(gca,'Fontsize',8,'linewidth',1);
                set(gca,'FontName','Times');
                grid(ax1,'off');
                view(3);
                lightangle(ax1,-45,45)
                lightangle(ax1,-45,45)
                colormap(Cmap2);
                %colormap turbo;
                line([0,0],[0,0],[0,max(f_vec)],'Color','y','LineWidth',1);
                line([-maxkx -maxkx],[-maxky  maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                line([-maxkx  maxkx],[ maxky  maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                line([ maxkx  maxkx],[ maxky -maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                line([ maxkx -maxkx],[-maxky -maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                xlim([-maxkx maxkx])
                ylim([-maxky maxky])
                zlim([0 maxf])
                box on; ax = gca; ax.BoxStyle = 'full';
                %view(-20,20)
                %view(-40,15)
                view(-30,50)
                Smax=max(max(max(abs(KXKYF(3:end,3:end,end/2+10:end)))));
                %caxis([0 0.4*Smax]);
                caxis([0 0.3*Smax]);
                
                % bottom plot
                ax2 = nexttile;
                h2 = slice(ax2,mkx,mky,mf,abs(KXKYF(:,:,end/2+1:end)),xslice1,yslice1,zslice1);
                set(h2,'FaceColor','interp','EdgeColor','none'); set(gcf,'Renderer','zbuffer');
                hold on;
                %ylabel({'$k_y$ [1/m]'},'Rotation',-37,'Fontsize',10,'interpreter','latex');% for 8cm figure
                ylabel({'$k_y$ [1/m]'},'Rotation',-38,'Fontsize',10,'interpreter','latex');% for 7.5cm figure
                xlabel({'$k_x$ [1/m]'},'Rotation', 15,'Fontsize',10,'interpreter','latex');
                zlabel({'$f$ [kHz]'},'Fontsize',10,'interpreter','latex')
                set(gca,'Fontsize',8,'linewidth',1);
                set(gca,'FontName','Times');
                view(3);
                
                colormap(Cmap2);
                %colormap turbo;
                lightangle(ax2,-45,45)
                lightangle(ax2,-45,45)
                line([0,0],[0,0],[0,max(f_vec)],'Color','y','LineWidth',1);
                line([-maxkx -maxkx],[-maxky  maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                line([-maxkx  maxkx],[ maxky  maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                line([ maxkx  maxkx],[ maxky -maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                line([ maxkx -maxkx],[-maxky -maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                xlim([-maxkx maxkx])
                ylim([-maxky maxky])
                zlim([freq_slice-0.01*freq_slice freq_slice+0.01*freq_slice]);
                % box off; ax = gca; ax.BoxStyle = 'full';
                %axis on;
                axis off;
                grid(ax2,'off');
                %title([num2str(freq_slice),' kHz'],'Fontsize',10,'interpreter','latex');
                text(-maxkx,maxky,freq_slice+0.01*freq_slice,[num2str(freq_slice),' kHz'],'HorizontalAlignment','left','Fontsize',10,'interpreter','latex');
                %view(-20,20)
                %view(-40,50)
                view(-30,50)
                [~,I]=min(abs(freq_slice-f_vec/1e3));
                Smax=max(max(max(abs(KXKYF(3:end,3:end,end/2+I)))));
                %caxis([0 0.8*Smax]);
                %caxis([0 0.7*Smax]);
                %caxis([0 0.6*Smax]);
                caxis([0 0.3*Smax]);
                set(gcf,'color','white');set(gca,'TickDir','out');
                %set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
                set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
                %set(gcf, 'Units','centimeters', 'Position',[10 10 8 10]);
                set(gcf, 'Units','centimeters', 'Position',[10 10 7.5 10]);
                % remove unnecessary white space
                set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));

                set(gcf,'PaperPositionMode','auto');
                drawnow;
                processed_filename = ['A0stop_KXKYF_',num2str(freq_slice),'_kHz']; % filename of processed .mat data
                print([figure_output_path,processed_filename],'-dpng', '-r600'); 
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %% KX-KY-F slices with applied mask
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                xslice1 = []; yslice1 = []; zslice1 = freq_slice;
                xslice2 = 0; yslice2 = 0; zslice2 = [];

                figure;
                t=tiledlayout(2,1);
                %t.TileSpacing = 'tight';
                t.TileSpacing = 'none';
                t.Padding = 'tight';
                % Top plot
                ax1 = nexttile;
                h1 = slice(ax1,mkx,mky,mf,abs(KXKYF_A0(:,:,end/2+1:end)),xslice2,yslice2,zslice2);
                set(h1,'FaceColor','interp','EdgeColor','none'); set(gcf,'Renderer','zbuffer');
                hold on;
                %ylabel({'$k_y$ [1/m]'},'Rotation',-37,'Fontsize',10,'interpreter','latex');% for 8cm figure
                ylabel({'$k_y$ [1/m]'},'Rotation',-38,'Fontsize',10,'interpreter','latex');% for 7.5cm figure
                xlabel({'$k_x$ [1/m]'},'Rotation', 15,'Fontsize',10,'interpreter','latex');
                zlabel({'$f$ [kHz]'},'Fontsize',10,'interpreter','latex')
                set(gca,'Fontsize',8,'linewidth',1);
                set(gca,'FontName','Times');
                grid(ax1,'off');
                view(3);
                lightangle(ax1,-45,45)
                lightangle(ax1,-45,45)
                colormap(Cmap2);
                %colormap turbo;
                line([0,0],[0,0],[0,max(f_vec)],'Color','y','LineWidth',1);
                line([-maxkx -maxkx],[-maxky  maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                line([-maxkx  maxkx],[ maxky  maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                line([ maxkx  maxkx],[ maxky -maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                line([ maxkx -maxkx],[-maxky -maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                xlim([-maxkx maxkx])
                ylim([-maxky maxky])
                zlim([0 maxf])
                box on; ax = gca; ax.BoxStyle = 'full';
                %view(-20,20)
                %view(-40,15)
                view(-30,50)
                Smax=max(max(max(abs(KXKYF_A0(3:end,3:end,end/2+10:end)))));
                %caxis([0 0.7*Smax]);
                caxis([0 0.6*Smax]);
                % bottom plot
                ax2 = nexttile;
                h2 = slice(ax2,mkx,mky,mf,abs(KXKYF_A0(:,:,end/2+1:end)),xslice1,yslice1,zslice1);
                set(h2,'FaceColor','interp','EdgeColor','none'); set(gcf,'Renderer','zbuffer');
                hold on;
                %ylabel({'$k_y$ [1/m]'},'Rotation',-37,'Fontsize',10,'interpreter','latex');% for 8cm figure
                ylabel({'$k_y$ [1/m]'},'Rotation',-38,'Fontsize',10,'interpreter','latex');% for 7.5cm figure
                xlabel({'$k_x$ [1/m]'},'Rotation', 15,'Fontsize',10,'interpreter','latex');
                zlabel({'$f$ [kHz]'},'Fontsize',10,'interpreter','latex')
                set(gca,'Fontsize',8,'linewidth',1);
                set(gca,'FontName','Times');
                view(3);
                
                colormap(Cmap2);
                %colormap turbo;
                lightangle(ax2,-45,45)
                lightangle(ax2,-45,45)
                line([0,0],[0,0],[0,max(f_vec)],'Color','y','LineWidth',1);
                line([-maxkx -maxkx],[-maxky  maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                line([-maxkx  maxkx],[ maxky  maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                line([ maxkx  maxkx],[ maxky -maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                line([ maxkx -maxkx],[-maxky -maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                xlim([-maxkx maxkx])
                ylim([-maxky maxky])
                zlim([freq_slice-0.01*freq_slice freq_slice+0.01*freq_slice]);
                % box off; ax = gca; ax.BoxStyle = 'full';
                %axis on;
                axis off;
                grid(ax2,'off');
                %title([num2str(freq_slice),' kHz'],'Fontsize',10,'interpreter','latex');
                text(-maxkx,maxky,freq_slice+0.01*freq_slice,[num2str(freq_slice),' kHz'],'HorizontalAlignment','left','Fontsize',10,'interpreter','latex');
                %view(-20,20)
                %view(-40,50)
                view(-30,50)
                [~,I]=min(abs(freq_slice-f_vec/1e3));
                Smax=max(max(max(abs(KXKYF_A0(3:end,3:end,end/2+I)))));
                %caxis([0 0.7*Smax]);
                caxis([0 0.6*Smax]);
                set(gcf,'color','white');set(gca,'TickDir','out');
                %set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
                set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
                %set(gcf, 'Units','centimeters', 'Position',[10 10 8 10]);
                set(gcf, 'Units','centimeters', 'Position',[10 10 7.5 10]);
                % remove unnecessary white space
                set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));

                set(gcf,'PaperPositionMode','auto');
                drawnow;
                processed_filename = ['A0stop_KXKYF_A0_',num2str(freq_slice),'_kHz']; % filename of processed .mat data
                print([figure_output_path,processed_filename],'-dpng', '-r600'); 
            end
            %% cylindrical coordinate  
            %[Data_polar,number_of_points,radius] =
            %cartesian_to_polar_wavefield_2pi2(Data,WL(1),WL(2),beta);%slow - for scattered data
            [Data_polar,number_of_points,radius] = cartesian_to_polar_wavefield_2pi_gridded2(Data,WL(1),WL(2),beta);%fast - for data on regural 
            %    dimensions [number_of_angles,number_of_points,number_of_time_steps]
            %save('Data_polar','Data_polar','beta','radius','-v7.3');
%             disp('loading polar data');
%             load('Data_polar');
            [number_of_angles,number_of_points,number_of_time_steps]=size(Data_polar);
            %% spatial signal at selected angle and time
            %N = 2^nextpow2(number_of_points);
            N = number_of_points;
            wavenumbers = zeros(number_of_angles,N-1,length(selected_frames));
            Amplitude = zeros(number_of_angles,N,length(selected_frames));
            
            
            x=zeros(number_of_angles,N);
            y=zeros(number_of_angles,N);
            b = beta*pi/180;
            dr=radius/(number_of_points-1);
            for ka=1:number_of_angles 
                R=0:dr:(N-1)*dr;
                x(ka,:) = R*cos(b(ka));
                y(ka,:) = R*sin(b(ka));
            end
            % add tapering window at the beginning and end of signal
            usable_sig_length = floor(Rmax/dr);
            Hn = hann(21);
            Windowing_mask=zeros(number_of_angles,number_of_points);
            for n_angle=1:number_of_angles
                Windowing_mask(n_angle,1:usable_sig_length(n_angle)) = 1;      
                Windowing_mask(n_angle,1:11) = Hn(1:11);
                %Windowing_mask(n_angle,usable_sig_length(n_angle)-10:usable_sig_length(n_angle)) = Hn(11:21);
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% main algorithm
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            xr=R; % the same radius for each angle
            % Numerical approximation of the Hilbert transform in the FFT domain:
            W = (-floor(N/2):ceil(N/2)-1)/N; % wavenumber coordinates normalized to interval -0.5:0.5
            H = ifftshift(  -1i * sign(W)  ); % sampled Fourier response
            c=0;
            for frame = selected_frames
                [frame]
                c=c+1;
                wavenumbers_by_angle = zeros(number_of_angles,N-1);
                amplitude_by_angle = zeros(number_of_angles,N);
                s = zeros(number_of_angles,N);
                s(:,1:number_of_points) = squeeze(Data_polar(:,1:number_of_points,frame)).*Windowing_mask;
                %s(:,1:number_of_points) = squeeze(Data_polar(:,1:number_of_points,frame));
                parfor n_angle=1:number_of_angles
 
                    % FFT-domain Hilbert transform of the input signal 's':
                    hilb = real(ifft(  fft(s(n_angle,:)) .* H  )); 
                    
                    sa = s(n_angle,:) + 1i*hilb; % complex valued analytic signal associated to input signal
                    amp = abs(sa);    % instantaneous amplitude envelope
                    phz = angle(sa);  % instantaneous phase
                    amp_smoothed = movmean(amp,20); % moving mean for smoothing amplitude
                    
                    %
                    unwrapped_phase = unwrap(phz)*scaling_factor; % unwrapped phase
                    if(~radians_flag) 
                        unwrapped_phase = unwrapped_phase/(2*pi);
                    end
                    
                    [p]=polyfit(xr(20:round(nx/2)),unwrapped_phase(20:round(nx/2)),1);
                    yfit = polyval(p,xr);
                    unwrapped_phase_flat = unwrapped_phase-yfit;
                    %unwrapped_phase_flat_smooth = movmean(unwrapped_phase_flat,10);
                    unwrapped_phase_flat_smooth1 = movmean(unwrapped_phase_flat(1:nx),10);
                    unwrapped_phase_flat_smooth2 = movmean(unwrapped_phase_flat(nx+1-40:end),40); % larger window at the end
                    dh=unwrapped_phase_flat_smooth2(19)-unwrapped_phase_flat_smooth1(nx-20);
                    unwrapped_phase_flat_smooth = [unwrapped_phase_flat_smooth1(1:nx-20),unwrapped_phase_flat_smooth2(19:end-2)-dh];
                    % smoothing second pass
                    unwrapped_phase_flat_smooth = movmean(unwrapped_phase_flat_smooth,10);
                    
                    hd = diff(unwrapped_phase_flat_smooth+yfit)/dr; % first derivative
                    wavenumbers_by_angle(n_angle,:) = movmean(hd,5); % unwrapped phase
                    amplitude_by_angle(n_angle,:) = amp_smoothed;
                end
                
                % smoothing over angle
                wavenumbers_by_angle_smoothed = movmean(wavenumbers_by_angle,10,1);
                amplitude_by_angle_smoothed = movmean(amplitude_by_angle,3,1);
                for n_angle=1:number_of_angles
                    Amplitude(n_angle,:,c) = amplitude_by_angle_smoothed(n_angle,:);    
                    wavenumbers(n_angle,:,c)  = wavenumbers_by_angle_smoothed(n_angle,:); 
                end
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% end of main algorithm
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % remove central point
            Amplitude(:,1,:)=0;
            wavenumbers(:,1,:)=0;
            radius_cut_wavenumbers = 40;
            radius_cut_amplitude = 0;
%%{
            figure('Position',[1 1 1920 1000])         
            c=0;
            for frame = selected_frames  
                c=c+1;
                subplot(1,2,1);
                surf(x(:,1:end-radius_cut_wavenumbers-1),y(:,1:end-radius_cut_wavenumbers-1),squeeze(wavenumbers(:,1:end-radius_cut_wavenumbers,c)));shading interp; view(2);  colorbar; colormap(Cmap);       
                Smax=max(max(squeeze(wavenumbers(:,1:end-radius_cut_wavenumbers,c))));Smin=min(min(squeeze(wavenumbers(:,1:end-radius_cut_wavenumbers,c))));
                set(gcf,'Renderer','zbuffer');
                xlim([-0.25 0.25]);
                ylim([-0.25, 0.25]);
                %axis equal;
                axis square;
                %caxis([caxis_cut*Smin,caxis_cut*Smax]);   
                %caxis([Smin,Smax]); 
                
%                 switch k
%                     case 1           
%                         caxis([200 1000]); % chirp
%                     case 2
%                         caxis([300 500]); % 50 kHz
%                     case 3
%                         caxis([350 650]); % 75 kHz
%                     case 4
%                         caxis([400 700]); % 100 kHz
%                     case 5
%                         caxis([550 850]); % 150 kHz
%                 end
                %caxis([200 800]); 
                %caxis([-100 500]); 
                title(['Wavenumbers f= ',num2str(frame)]);
                
                subplot(1,2,2);
                surf(x(:,1:end-radius_cut_amplitude),y(:,1:end-radius_cut_amplitude),squeeze(Amplitude(:,1:end-radius_cut_amplitude,c)));shading interp; view(2); colorbar; colormap(Cmap);       
                Smax=max(max(squeeze(Amplitude(:,1:end-radius_cut_amplitude,c))));Smin=0;
                set(gcf,'Renderer','zbuffer');
                xlim([-0.25 0.25]);
                ylim([-0.25, 0.25]);
                axis square;  
                %caxis([caxis_cut*Smin,caxis_cut*Smax]);    
                
%                 switch k
%                     case 1           
%                         caxis([0 8e-3]);  % chirp
%                     case 2
%                         caxis([0 7e-3]);  % 50 kHz
%                     case 3
%                         caxis([0 7e-3]); % 75 kHz
%                     case 4
%                         caxis([0 7e-3]);  % 100 kHz
%                     case 5
%                         caxis([0 7e-3]);  % 150 kHz
%                 end
                title(['Amplitude f= ',num2str(frame)]);
                pause(0.1);
            end
%%}            
            % selected frames
            RMS_wavenumbers_selected = sqrt(sum(wavenumbers.^2,3))/length(selected_frames);
            Mean_wavenumbers_selected = mean(wavenumbers,3);
            RMS_amplitude_selected = sqrt(sum(Amplitude.^2,3))/length(selected_frames);
            Mean_amplitude_selected = mean(Amplitude,3);
            
            % further refine selection of nt frames with lowest standard deviation
            % refine frame selection
            Kstd = zeros(length(selected_frames),1);
            c=0;
            for frame = selected_frames 
                c=c+1;
                Kstd(c) = std(wavenumbers(:,:,c),1,'all'); 
            end
            nt=16;
            [B,I]=sort(Kstd,'ascend');
            
            % refined
            RMS_wavenumbers_refined = sqrt(sum(wavenumbers(:,:,I(1:nt)).^2,3))/length(selected_frames);
            Mean_wavenumbers_refined = mean(wavenumbers(:,:,I(1:nt)),3);
            RMS_amplitude_refined = sqrt(sum(Amplitude(:,:,I(1:nt)).^2,3))/length(selected_frames);
            Mean_amplitude_refined = mean(Amplitude(:,:,I(1:nt)),3);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % convert back to cartesian coordinates by using linear interpolation
            [TH,Radius] = meshgrid(beta*pi/180,R(1:end-1));
            [Xk,Yk,Zk] = pol2cart(TH,Radius,Mean_wavenumbers_refined');
            %figure;surf(Xk,Yk,Zk);shading interp;view(2);
            [XI,YI] = meshgrid(linspace(-WL(1)/2,WL(1)/2,Nx),linspace(-WL(2)/2,WL(2)/2,Ny)); % 
            F = scatteredInterpolant(reshape(Xk,[],1),reshape(Yk,[],1),reshape(Zk,[],1),'linear','none'); % requires ndgrid format; no extrapolation
            Data_cart=F(XI,YI);Data_cart(isnan(Data_cart))=0;
            %figure;surf(XI,YI,Data_cart);shading interp;view(2);xlim([-0.25,0.25]);ylim([-0.25 0.25]);axis square;
            Mean_wavenumbers_refined_smooth = medfilt2(Data_cart,[16,16]);
            
            [Xk,Yk,Zk] = pol2cart(TH,Radius,Mean_wavenumbers_selected');
            %figure;surf(Xk,Yk,Zk);shading interp;view(2);
            F = scatteredInterpolant(reshape(Xk,[],1),reshape(Yk,[],1),reshape(Zk,[],1),'linear','none'); % requires ndgrid format; no extrapolation
            Data_cart=F(XI,YI);Data_cart(isnan(Data_cart))=0;
            %figure;surf(XI,YI,Data_cart);shading interp;view(2);xlim([-0.25,0.25]);ylim([-0.25 0.25]);axis square;
            Mean_wavenumbers_selected_smooth = medfilt2(Data_cart,[16,16]);
            %figure;surf(XI,YI,Mean_wavenumbers_selected_smooth);shading interp;view(2);xlim([-0.25,0.25]);ylim([-0.25 0.25]);axis square;
           
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% Figures selected
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % RMS wavenumbers
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            radius_cut_wavenumbers=0;
            figure;
            surf(x(:,1:end-radius_cut_wavenumbers-1),y(:,1:end-radius_cut_wavenumbers-1),squeeze(RMS_wavenumbers_selected(:,1:end-radius_cut_wavenumbers)));shading interp; view(2);  colorbar; colormap(Cmap);       
            Smax=max(max(squeeze(RMS_wavenumbers_selected(:,1:end-radius_cut_wavenumbers))));Smin=0;
            set(gcf,'Renderer','zbuffer');
            xlim([-0.25 0.25]);
            ylim([-0.25, 0.25]);
            set(gca,'Fontsize',10);
            axis square;
            hold on;
            if(damage_outline) plot3(delam1(:,1),delam1(:,2),repmat(Smax,[length(delam1),1]),'k:','LineWidth',0.5); end
            %caxis([caxis_cut*Smin,caxis_cut*Smax]);  
            %caxis([Smin,Smax]); 
            %caxis([0 70]); 
            %title(['RMS wavenumbers']);
            set(gcf,'color','white');set(gca,'TickDir','out');
            %set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
            set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
            set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
            % remove unnecessary white space
            set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
            
            set(gcf,'PaperPositionMode','auto');
            drawnow;
            processed_filename = ['A0stop_RMS_wavenumbers_selected_',filename]; % filename of processed .mat data
            print([figure_output_path,processed_filename],'-dpng', '-r600'); 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Mean wavenumbers
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            figure;
            surf(x(:,1:end-radius_cut_wavenumbers-1),y(:,1:end-radius_cut_wavenumbers-1),squeeze(Mean_wavenumbers_selected (:,1:end-radius_cut_wavenumbers)));shading interp; view(2); colorbar; colormap(Cmap);       
            Smax=max(max(squeeze(Mean_wavenumbers_selected (:,1:end-radius_cut_wavenumbers))));Smin=min(min(squeeze(Mean_wavenumbers_selected (:,1:end-radius_cut_wavenumbers))));
            set(gcf,'Renderer','zbuffer');
            xlim([-0.25 0.25]);
            ylim([-0.25, 0.25]);
            set(gca,'Fontsize',10);
%             switch k
%                 case 1   
%                     if(radians_flag) 
%                         caxis([200 1000]*scaling_factor); % chirp
%                     else
%                         caxis([200 1000]/(2*pi)*scaling_factor); % chirp
%                     end
%                 case 2
%                     if(radians_flag) 
%                         caxis([300 500]*scaling_factor); % 50 kHz
%                     else
%                         caxis([300 500]/(2*pi)*scaling_factor); % 50 kHz
%                     end
%                 case 3
%                     if(radians_flag) 
%                         caxis([350 650]*scaling_factor); % 75 kHz
%                     else
%                         caxis([350 650]/(2*pi)*scaling_factor); % 75 kHz
%                     end
%                 case 4
%                     if(radians_flag) 
%                         caxis([400 700]*scaling_factor); % 100 kHz
%                     else
%                         caxis([400 700]/(2*pi)*scaling_factor); % 100 kHz
%                     end
%                 case 5
%                     if(radians_flag) 
%                         caxis([550 850]*scaling_factor); % 150 kHz
%                     else
%                         caxis([550 850]/(2*pi)*scaling_factor); % 150 kHz
%                     end
%             end
            axis square;
            hold on;
            if(damage_outline) plot3(delam1(:,1),delam1(:,2),repmat(Smax,[length(delam1),1]),'k:','LineWidth',0.5); end
            
            %caxis([caxis_cut*Smin,caxis_cut*Smax]);    
            %caxis([Smin,Smax]);  
            %caxis([0 500]); 
            %title(['Mean wavenumbers']);
            set(gcf,'color','white');set(gca,'TickDir','out');
            %set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
            set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
            set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
            % remove unnecessary white space
            set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
            
            set(gcf,'PaperPositionMode','auto');
            drawnow;
            processed_filename = ['A0stop_Mean_wavenumbers_selected_',filename]; % filename of processed .mat data
            print([figure_output_path,processed_filename],'-dpng', '-r600'); 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Mean wavenumbers smoothed
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            figure;
            surf(XI,YI,Mean_wavenumbers_selected_smooth);shading interp; view(2); colorbar; colormap(Cmap);       
            Smax=max(max(Mean_wavenumbers_selected_smooth));Smin=min(min(Mean_wavenumbers_selected_smooth));
            set(gcf,'Renderer','zbuffer');
            xlim([-0.25 0.25]);
            ylim([-0.25, 0.25]);
            set(gca,'Fontsize',10);
%             switch k
%                 case 1   
%                     if(radians_flag) 
%                         caxis([200 1000]*scaling_factor); % chirp
%                     else
%                         caxis([200 1000]/(2*pi)*scaling_factor); % chirp
%                     end
%                 case 2
%                     if(radians_flag) 
%                         caxis([300 500]*scaling_factor); % 50 kHz
%                     else
%                         caxis([300 500]/(2*pi)*scaling_factor); % 50 kHz
%                     end
%                 case 3
%                     if(radians_flag) 
%                         caxis([350 650]*scaling_factor); % 75 kHz
%                     else
%                         caxis([350 650]/(2*pi)*scaling_factor); % 75 kHz
%                     end
%                 case 4
%                     if(radians_flag) 
%                         caxis([400 700]*scaling_factor); % 100 kHz
%                     else
%                         caxis([400 700]/(2*pi)*scaling_factor); % 100 kHz
%                     end
%                 case 5
%                     if(radians_flag) 
%                         caxis([550 850]*scaling_factor); % 150 kHz
%                     else
%                         caxis([550 850]/(2*pi)*scaling_factor); % 150 kHz
%                     end
%             end
            axis square;
            hold on;
            if(damage_outline) plot3(delam1(:,1),delam1(:,2),repmat(Smax,[length(delam1),1]),'k:','LineWidth',0.5); end
            
            %caxis([caxis_cut*Smin,caxis_cut*Smax]);    
            %caxis([Smin,Smax]);  
            %caxis([0 500]); 
            %title(['Mean wavenumbers']);
            set(gcf,'color','white');set(gca,'TickDir','out');
            %set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
            set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
            set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
            % remove unnecessary white space
            set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
            
            set(gcf,'PaperPositionMode','auto');
            drawnow;
            processed_filename = ['A0stop_Mean_wavenumbers_selected_smooth_',filename]; % filename of processed .mat data
            print([figure_output_path,processed_filename],'-dpng', '-r600'); 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % RMS amplitude
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            figure;
            surf(x(:,1:end-radius_cut_amplitude),y(:,1:end-radius_cut_amplitude),squeeze(RMS_amplitude_selected(:,1:end-radius_cut_amplitude)));shading interp; view(2); colorbar; colormap(Cmap);       
            Smax=max(max(squeeze(RMS_amplitude_selected(:,1:end-radius_cut_amplitude))));Smin=0;
            set(gcf,'Renderer','zbuffer');
            xlim([-0.25 0.25]);
            ylim([-0.25, 0.25]);
            set(gca,'Fontsize',10);
            axis square;
            caxis([caxis_cut*Smin,caxis_cut*Smax]);   
            hold on;
            if(damage_outline) plot3(delam1(:,1),delam1(:,2),repmat(caxis_cut*Smax,[length(delam1),1]),'k:','LineWidth',0.5); end
           
            %caxis([0 7e-4]); 
            %title(['RMS amplitude']);
            set(gcf,'color','white');set(gca,'TickDir','out');
            %set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
            set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
            set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
            % remove unnecessary white space
            set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
            
            set(gcf,'PaperPositionMode','auto');
            drawnow;
            processed_filename = ['A0stop_RMS_amplitude_selected_',filename]; % filename of processed .mat data
            print([figure_output_path,processed_filename],'-dpng', '-r600');
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Mean amplitude
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            figure;
            surf(x(:,1:end-radius_cut_amplitude),y(:,1:end-radius_cut_amplitude),squeeze(Mean_amplitude_selected (:,1:end-radius_cut_amplitude)));shading interp; view(2); colorbar; colormap(Cmap);       
            Smax=max(max(squeeze(Mean_amplitude_selected (:,1:end-radius_cut_amplitude))));Smin=0;
            set(gcf,'Renderer','zbuffer');
            xlim([-0.25 0.25]);
            ylim([-0.25, 0.25]);
            set(gca,'Fontsize',10);
            axis square;
            caxis([caxis_cut*Smin,caxis_cut*Smax]);  
            hold on;
            if(damage_outline) plot3(delam1(:,1),delam1(:,2),repmat(caxis_cut*Smax,[length(delam1),1]),'k:','LineWidth',0.5); end
           
            %caxis([0 4.5e-3]); 
            %title(['Mean amplitude']);
            set(gcf,'color','white');set(gca,'TickDir','out');
            %set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
            set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
            set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
            % remove unnecessary white space
            set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
            
            set(gcf,'PaperPositionMode','auto');
            drawnow;
            processed_filename = ['A0stop_Mean_amplitude_selected_',filename]; % filename of processed .mat data
            print([figure_output_path,processed_filename],'-dpng', '-r600'); 
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% Figures refined
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % RMS wavenumbers
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            radius_cut_wavenumbers=0;
            figure;
            surf(x(:,1:end-radius_cut_wavenumbers-1),y(:,1:end-radius_cut_wavenumbers-1),squeeze(RMS_wavenumbers_refined(:,1:end-radius_cut_wavenumbers)));shading interp; view(2);  colorbar; colormap(Cmap);       
            Smax=max(max(squeeze(RMS_wavenumbers_refined(:,1:end-radius_cut_wavenumbers))));Smin=0;
            set(gcf,'Renderer','zbuffer');
            xlim([-0.25 0.25]);
            ylim([-0.25, 0.25]);
            set(gca,'Fontsize',10);
            axis square;
            hold on;
            if(damage_outline) plot3(delam1(:,1),delam1(:,2),repmat(Smax,[length(delam1),1]),'k:','LineWidth',0.5); end
            %caxis([caxis_cut*Smin,caxis_cut*Smax]);  
            %caxis([Smin,Smax]); 
            %caxis([0 70]); 
            %title(['RMS wavenumbers']);
            set(gcf,'color','white');set(gca,'TickDir','out');
            %set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
            set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
            set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
            % remove unnecessary white space
            set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
            
            set(gcf,'PaperPositionMode','auto');
            drawnow;
            processed_filename = ['A0stop_RMS_wavenumbers_refined_',filename]; % filename of processed .mat data
            print([figure_output_path,processed_filename],'-dpng', '-r600'); 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Mean wavenumbers
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            figure;
            surf(x(:,1:end-radius_cut_wavenumbers-1),y(:,1:end-radius_cut_wavenumbers-1),squeeze(Mean_wavenumbers_refined (:,1:end-radius_cut_wavenumbers)));shading interp; view(2); ch=colorbar; colormap(Cmap);       
            Smax=max(max(squeeze(Mean_wavenumbers_refined (:,1:end-radius_cut_wavenumbers))));Smin=mean(mean(squeeze(Mean_wavenumbers_refined (:,1:end-radius_cut_wavenumbers))));
            set(gcf,'Renderer','zbuffer');
            xlim([-0.25 0.25]);
            ylim([-0.25, 0.25]);
            set(gca,'Fontsize',10);
%             ch.Label.String = '[rad/m]';
%             ch.Label.FontSize = 10;
%             ch.Label.Position(1) = 0;
%             ch.Label.Position(2) = 500;
%             ylabel(ch,'[rad/m]','FontSize',10,'Rotation',0);
%             switch k
%                 case 1   
%                     if(radians_flag) 
%                         caxis([200 1000]*scaling_factor); % chirp
%                     else
%                         caxis([200 1000]/(2*pi)*scaling_factor); % chirp
%                     end
%                 case 2
%                     if(radians_flag) 
%                         caxis([300 500]*scaling_factor); % 50 kHz
%                     else
%                         caxis([300 500]/(2*pi)*scaling_factor); % 50 kHz
%                     end
%                 case 3
%                     if(radians_flag) 
%                         caxis([350 650]*scaling_factor); % 75 kHz
%                     else
%                         caxis([350 650]/(2*pi)*scaling_factor); % 75 kHz
%                     end
%                 case 4
%                     if(radians_flag) 
%                         caxis([400 700]*scaling_factor); % 100 kHz
%                     else
%                         caxis([400 700]/(2*pi)*scaling_factor); % 100 kHz
%                     end
%                 case 5
%                     if(radians_flag) 
%                         caxis([550 850]*scaling_factor); % 150 kHz
%                     else
%                         caxis([550 850]/(2*pi)*scaling_factor); % 150 kHz
%                     end
%             end
            axis square;
            hold on;
            if(damage_outline) plot3(delam1(:,1),delam1(:,2),repmat(Smax,[length(delam1),1]),'k:','LineWidth',0.5); end
            set(gca,'Fontsize',10);
            %caxis([caxis_cut*Smin,caxis_cut*Smax]);    
            %caxis([Smin,Smax]);  
            %caxis([0 500]); 
            %title(['Mean wavenumbers']);
            set(gcf,'color','white');set(gca,'TickDir','out');
            %set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
            set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
            set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
            % remove unnecessary white space
            set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
            
            set(gcf,'PaperPositionMode','auto');
            drawnow;
            processed_filename = ['A0stop_Mean_wavenumbers_refined_',filename]; % filename of processed .mat data
            print([figure_output_path,processed_filename],'-dpng', '-r600'); 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Mean wavenumbers smoothed
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            figure;
            surf(XI,YI,Mean_wavenumbers_refined_smooth);shading interp; view(2); colorbar; colormap(Cmap);       
            Smax=max(max(Mean_wavenumbers_refined_smooth));Smin=min(min(Mean_wavenumbers_refined_smooth));
            set(gcf,'Renderer','zbuffer');
            xlim([-0.25 0.25]);
            ylim([-0.25, 0.25]);
            set(gca,'Fontsize',10);
%             switch k
%                 case 1   
%                     if(radians_flag) 
%                         caxis([200 1000]*scaling_factor); % chirp
%                     else
%                         caxis([200 1000]/(2*pi)*scaling_factor); % chirp
%                     end
%                 case 2
%                     if(radians_flag) 
%                         caxis([300 500]*scaling_factor); % 50 kHz
%                     else
%                         caxis([300 500]/(2*pi)*scaling_factor); % 50 kHz
%                     end
%                 case 3
%                     if(radians_flag) 
%                         caxis([350 650]*scaling_factor); % 75 kHz
%                     else
%                         caxis([350 650]/(2*pi)*scaling_factor); % 75 kHz
%                     end
%                 case 4
%                     if(radians_flag) 
%                         caxis([400 700]*scaling_factor); % 100 kHz
%                     else
%                         caxis([400 700]/(2*pi)*scaling_factor); % 100 kHz
%                     end
%                 case 5
%                     if(radians_flag) 
%                         caxis([550 850]*scaling_factor); % 150 kHz
%                     else
%                         caxis([550 850]/(2*pi)*scaling_factor); % 150 kHz
%                     end
%             end
            axis square;
            hold on;
            if(damage_outline) plot3(delam1(:,1),delam1(:,2),repmat(Smax,[length(delam1),1]),'k:','LineWidth',0.5); end
            %caxis([caxis_cut*Smin,caxis_cut*Smax]);    
            %caxis([Smin,Smax]);  
            %caxis([0 500]); 
            %title(['Mean wavenumbers']);
            set(gcf,'color','white');set(gca,'TickDir','out');
            %set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
            set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
            set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
            % remove unnecessary white space
            set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
            
            set(gcf,'PaperPositionMode','auto');
            drawnow;
            processed_filename = ['A0stop_Mean_wavenumbers_refined_smooth_',filename]; % filename of processed .mat data
            print([figure_output_path,processed_filename],'-dpng', '-r600'); 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % RMS amplitude
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            figure;
            surf(x(:,1:end-radius_cut_amplitude),y(:,1:end-radius_cut_amplitude),squeeze(RMS_amplitude_refined(:,1:end-radius_cut_amplitude)));shading interp; view(2); colorbar; colormap(Cmap);       
            Smax=max(max(squeeze(RMS_amplitude_refined(:,1:end-radius_cut_amplitude))));Smin=0;
            set(gcf,'Renderer','zbuffer');
            xlim([-0.25 0.25]);
            ylim([-0.25, 0.25]);
            set(gca,'Fontsize',10);
            axis square;
            caxis([caxis_cut*Smin,caxis_cut*Smax]);   
            hold on;
            if(damage_outline) plot3(delam1(:,1),delam1(:,2),repmat(caxis_cut*Smax,[length(delam1),1]),'k:','LineWidth',0.5); end
            %caxis([0 7e-4]); 
            %title(['RMS amplitude']);
            set(gcf,'color','white');set(gca,'TickDir','out');
            %set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
            set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
            set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
            % remove unnecessary white space
            set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
            
            set(gcf,'PaperPositionMode','auto');
            drawnow;
            processed_filename = ['A0stop_RMS_amplitude_refined_',filename]; % filename of processed .mat data
            print([figure_output_path,processed_filename],'-dpng', '-r600');
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Mean amplitude
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            figure;
            surf(x(:,1:end-radius_cut_amplitude),y(:,1:end-radius_cut_amplitude),squeeze(Mean_amplitude_refined (:,1:end-radius_cut_amplitude)));shading interp; view(2); colorbar; colormap(Cmap);       
            Smax=max(max(squeeze(Mean_amplitude_refined (:,1:end-radius_cut_amplitude))));Smin=0;
            set(gcf,'Renderer','zbuffer');
            xlim([-0.25 0.25]);
            ylim([-0.25, 0.25]);
            set(gca,'Fontsize',10);
            axis square;
            caxis([caxis_cut*Smin,caxis_cut*Smax]);
            hold on;
            if(damage_outline) plot3(delam1(:,1),delam1(:,2),repmat(caxis_cut*Smax,[length(delam1),1]),'k:','LineWidth',0.5); end
            %caxis([0 4.5e-3]); 
            %title(['Mean amplitude']);
            set(gcf,'color','white');set(gca,'TickDir','out');
            %set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
            set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
            set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
            % remove unnecessary white space
            set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
            
            set(gcf,'PaperPositionMode','auto');
            drawnow;
            processed_filename = ['A0stop_Mean_amplitude_refined_',filename]; % filename of processed .mat data
            print([figure_output_path,processed_filename],'-dpng', '-r600'); 
            %close all;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% wavenumber to thickness scaling
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            [m1,~,~] = size(thickness_sensitivities);
            Thickness_map = zeros(Nx,Ny);
            phi = atan2(YI,XI)*180/pi; % range -180:+180
            J=find(phi<0) ;
            temp=phi(J);
            phi(J) = 360 + temp; % change range to 0:360
            dbeta1 = 360/(m1-1); 
            beta1 = (dbeta1:dbeta1:(m1)*dbeta1)-dbeta1;
            for i=1:Ny
                for j=1:Nx
                    [~,I] = min(abs(phi(i,j)-beta1));
                    if(Mean_wavenumbers_refined_smooth(i,j)<min(thickness_sensitivities(I,:,k-1)))
                        Thickness_map(i,j) = 6;
                    else
                        [~,J] = min( abs( thickness_sensitivities(I,:,k-1)-Mean_wavenumbers_refined_smooth(i,j) ) );
                        Thickness_map(i,j) = thickness(J);
                        %Thickness_map(i,j)=interp1(squeeze(thickness_sensitivities(I,:,k-1)),thickness,Mean_wavenumbers_refined_smooth(i,j),'linear','extrap');
                
                    end
                end
            end
            Thickness_map_avg = Thickness_map_avg + Thickness_map/length(freq_list);
            figure;
            surf(XI,YI,Thickness_map);shading interp; view(2); colorbar; colormap(flipud(Cmap));       
            %Smax=max(max(Thickness_map));Smin=min(min(Thickness_map));
            set(gcf,'Renderer','zbuffer');
            xlim([-0.25 0.25]);
            ylim([-0.25, 0.25]);
            set(gca,'Fontsize',10);
            axis square;
            hold on;
            if(damage_outline) plot3(delam1(:,1),delam1(:,2),repmat(4,[length(delam1),1]),'k:','LineWidth',0.5); end   
            caxis([0 4]);   
            %caxis([0 4.5e-3]); 
            %title(['Mean amplitude']);
            set(gcf,'color','white');set(gca,'TickDir','out');
            %set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
            set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
            set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
            % remove unnecessary white space
            set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
            
            set(gcf,'PaperPositionMode','auto');
            drawnow;
            processed_filename = ['A0stop_Thickness_map_',filename]; % filename of processed .mat data
            print([figure_output_path,processed_filename],'-dpng', '-r600');
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% RMS of A0 stop band filtered wavefield
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            RMS_A0stop = sqrt(sum(Data(:,:,selected_frames2).^2,3))./length(selected_frames2);
            RMS_A0stop_avg = RMS_A0stop_avg + RMS_A0stop/length(freq_list);
            figure;
            surf(XI,YI,RMS_A0stop);shading interp; view(2); colorbar; colormap((Cmap));       
            Smax=max(max(RMS_A0stop));Smin=0;
            set(gcf,'Renderer','zbuffer');
            xlim([-0.25 0.25]);
            ylim([-0.25, 0.25]);
            set(gca,'Fontsize',10);
            axis square;
            hold on;
            if(damage_outline) plot3(delam1(:,1),delam1(:,2),repmat(Smax,[length(delam1),1]),'k:','LineWidth',0.5); end
            
            set(gcf,'color','white');set(gca,'TickDir','out');
            %set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
            set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
            set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
            % remove unnecessary white space
            set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));

            set(gcf,'PaperPositionMode','auto');
            drawnow;
            processed_filename = ['A0stop_RMS_A0stop_',filename]; % filename of processed .mat data
            print([figure_output_path,processed_filename],'-dpng', '-r600');
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% END OF PROCESSING
            [filepath,name,ext] = fileparts(filename);
            fprintf('Successfully processed:\n%s\n', name);% successfully processed
        
    else
        fprintf('Filename: \n%s \nalready exist\n', processed_filename);
    end
end
figure;
surf(XI,YI,Thickness_map_avg);shading interp; view(2); colorbar; colormap(flipud(Cmap));       
%Smax=max(max(Thickness_map));Smin=min(min(Thickness_map));
set(gcf,'Renderer','zbuffer');
xlim([-0.25 0.25]);
ylim([-0.25, 0.25]);
set(gca,'Fontsize',10);
axis square;
hold on;
if(damage_outline) plot3(delam1(:,1),delam1(:,2),repmat(4,[length(delam1),1]),'k:','LineWidth',0.5); end  
caxis([0 4]);    
%caxis([0 4.5e-3]); 
%title(['Mean amplitude']);
set(gcf,'color','white');set(gca,'TickDir','out');
%set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));

set(gcf,'PaperPositionMode','auto');
drawnow;
processed_filename = ['A0stop_Thickness_map_avg']; % filename of processed .mat data
print([figure_output_path,processed_filename],'-dpng', '-r600');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
surf(XI,YI,RMS_A0stop_avg);shading interp; view(2); colorbar; colormap((Cmap));       
Smax=max(max(RMS_A0stop_avg));Smin=0;
set(gcf,'Renderer','zbuffer');
xlim([-0.25 0.25]);
ylim([-0.25, 0.25]);
set(gca,'Fontsize',10);
axis square;
hold on;
if(damage_outline) plot3(delam1(:,1),delam1(:,2),repmat(Smax,[length(delam1),1]),'k:','LineWidth',0.5); end
            
set(gcf,'color','white');set(gca,'TickDir','out');
%set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));

set(gcf,'PaperPositionMode','auto');
drawnow;
processed_filename = ['A0stop_RMS_A0stop_avg']; % filename of processed .mat data
print([figure_output_path,processed_filename],'-dpng', '-r600');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% End of Yet another wavenumber damage imaging (YAWDI)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


toc
  