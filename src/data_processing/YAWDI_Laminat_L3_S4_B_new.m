%% Yet Another Wavenumber Damage Imaging (YAWDI)
% variant with A0 mode pass band filter 
% (better for phase tracking and thickness mapping)
% Algorithms needs at least two matrices as an input:
% 1. chirp wavefield
% 2. wavefield pass-banded around frequency fc
% Dimensions of matrices should be 512x512xm512
% which corresponds to spatial dimensions (x,y) and time (t)
% Other inputs:
% time vector as "time" variable
% Plate dimensions as "WL" variable (two-element column vector)

% Author: Pawel Kudela, D.Sc., Ph.D., Eng. 
% Institute of Fluid Flow Machinery Polish Academy of Sciences 
% Mechanics of Intelligent Structures Department 
% email address: pk@imp.gda.pl 
% Website: https://www.imp.gda.pl/en/research-centres/o4/o4z1/people/ 

% Plain weave CFRP laminate L3_S4_B


clear all;close all;   warning off;clc;
tic
load project_paths projectroot src_path;
%% Input for signal processing
base_thickness = 3.9; % [mm] reference thickness of the plate
specimen_name='L3_S4_B';
% full field measurements
list = {'chirp_interp','Data50_packet_interp','Data75_packet_interp','Data100_packet_interp'};
freq_list =[50,75,100]; % frequency list in kHz according to files above (max 4 frequencies)
test_case=[2:4]; % select file numbers for processing (starting from 2, chirp should be excluded)
amp_threshold = [0.2,0.18,0.15]; % amplitude threshold
selected_frames={40:120,80:300,80:300,240:400}; % selected frames for Hilbert transform
%% Prepare output directories
% allow overwriting existing results if true
%overwrite=false;
overwrite=true;
interim_figs=true;
A0mode_filter=true; % true - A0 pass band filter is applied, false - algorithm on unfiltered data (fast)
% unfiltered means that A0 pass band filter in frequency-wavenumber domain is not applied
% the filenames will be preceded by term: 'unfiltered_'
% retrieve model name based on running file and folder
freq_filter=true; % apply additional frequency band pass filter 
fband = 20e3; % frequency band width [Hz]

currentFile = mfilename('fullpath');
[pathstr,name,ext] = fileparts( currentFile );
idx = strfind( pathstr,filesep );
modelfolder = pathstr(idx(end)+1:end); % name of folder
modelname = name; 
% prepare output paths
dataset_output_path = prepare_data_processing_paths('processed','exp',modelname);
figure_output_path = prepare_figure_paths(modelname);

radians_flag = false; % if true units of wanumbers [rad/m] if false [1/m]


scaling_factor=1;
%% input for figures
Cmap = jet(256); 
Cmap2 = turbo; 
caxis_cut = 0.8;
fig_width =6; % figure widht in cm
fig_height=6; % figure height in cm
damage_outline=true;
%% Damage outline - ellipse
%    N - numer of points in adjacent grid, integer
%    xCenter -  delamination x coordinate 
%    yCenter -  delamination y coordinate 
%    rotAngle - delamination rotation angle [0:180), Units: deg
%    a - semi-major axis
%    b - semi-minor axis
% delam 1 (ellipse)
rotAngle=0;
xCenter = 0;
yCenter = 0.155;
b=0.01/2;
a=0.02/2;
alpha=rotAngle*pi/180;
te=linspace(-pi,pi,50);
x=a*cos(te);
y=b*sin(te);
R  = [cos(alpha) -sin(alpha); ...
      sin(alpha)  cos(alpha)];
rCoords = R*[x ; y];   
xr = rCoords(1,:)';      
yr = rCoords(2,:)';     
delam1= [xr+xCenter,yr+yCenter];
% delam 2 (ellipse)
rotAngle=0;
xCenter = -0.150;
yCenter = 0.004;
b=0.01/2;
a=0.02/2;
alpha=rotAngle*pi/180;
te=linspace(-pi,pi,50);
x=a*cos(te);
y=b*sin(te);
R  = [cos(alpha) -sin(alpha); ...
      sin(alpha)  cos(alpha)];
rCoords = R*[x ; y];   
xr = rCoords(1,:)';      
yr = rCoords(2,:)';     
delam2= [xr+xCenter,yr+yCenter];
% delam 3 (ellipse)
a=0.01;
b=0.02;
xCenter = 0.105;
yCenter = -0.1;
b=0.01/2;
a=0.02/2;
alpha=rotAngle*pi/180;
te=linspace(-pi,pi,50);
x=a*cos(te);
y=b*sin(te);
R  = [cos(alpha) -sin(alpha); ...
      sin(alpha)  cos(alpha)];
rCoords = R*[x ; y];   
xr = rCoords(1,:)';      
yr = rCoords(2,:)';     
delam3= [xr+xCenter,yr+yCenter];

%% Processing parameters
Nx = 512;   % number of points after interpolation in X direction
Ny = 512;   % number of points after interpolation in Y direction
Nmed = 3;   % median filtering window size e.g. Nmed = 2 gives 2 by 2 points window size

N = 1024;% for zero padding

%% input for mask
if(radians_flag)
    mask_width_A0_1=200/2; % half wavenumber band width [rad/m]
    mask_width_A0_2=300/2;
else
%     mask_width_A0_1=80; % wavenumber band width [1/m]
%     mask_width_A0_2=80;
    mask_width_A0_1=40; % wavenumber band width [1/m]
    mask_width_A0_2=40; % wavenumber band width [1/m] at higher frequencies
end
offset = 50; % offset of wavenumbers from center of A0 mode towards positive wavenumbers [1/m]
% input for ridge picking algorithm for A0 mode extraction
% frequency range 15-110 kHz
f_start = 15000; % [Hz]
f_end = 130000;  % [Hz]
w = 40; % weight for linear wavenumber amplification (to avoid S0 mode contribution) w~20...100
% w = 20; %for 4mm thick CFRP
% w = 40; % 
%%
% create path to the experimental raw data folder

raw_data_path = ['/pkudela_odroid_laser/aidd/data/raw/exp/',specimen_name,'/'];

if(A0mode_filter)
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
    [KXKYF,kx_vec,ky_vec,f_vec] = spatial_to_wavenumber_wavefield_full2(Data,Length,Width,time); % full size data (-kx:+kx,-ky:+ky,-f:+f)
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
    %%{
    if(interim_figs)
        if(radians_flag)   
            [mkx,mky,mf] = meshgrid(kx_vec,ky_vec,f_vec/1e3);
        else
            [mkx,mky,mf] = meshgrid(kx_vec/(2*pi),ky_vec/(2*pi),f_vec/1e3);
        end
        % maxkx = 1000/(2*pi);
        % maxky = 1000/(2*pi);
%         maxkx = 200;
%         maxky = 200;
        maxkx = 400;
        maxky = 400;
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
%                 lightangle(ax2,-45,45)
%                 lightangle(ax2,-45,45)
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
                processed_filename = [specimen_name,'_KXKYF_chirp_',num2str(freq_slice),'_kHz']; % filename of processed .mat data
                print([figure_output_path,processed_filename],'-dpng', '-r600'); 
        end
    end
    %}
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
    
    [~,f_start_ind] = min(abs(f_vec-f_start));
    [~,f_end_ind] = min(abs(f_vec-f_end));
    Ind = zeros(f_end_ind - f_start_ind+1,length(beta)/2);
    k_vec = linspace(0,k_radius,n2);
    k_A0_smooth = zeros(length(beta),length(f_vec));
    k_A0_f_selected = zeros(length(beta)/2,f_end_ind-f_start_ind+1);
    n_outliers = round(0.1*(f_end_ind - f_start_ind+1)); % number of data points to remove
    weighting = linspace(1,w,n2); % promote higher wavenumbers - for better extraction of A0 mode
    
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
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
        if(length(freq_list)==2)
            plot(thickness(1:4:end),kq(2,1:4:end),'rv-','MarkerFaceColor','r','MarkerSize',2);
            legend([num2str(freq_list(1)),' kHz'], [num2str(freq_list(2)),' kHz'],'FontSize',10);
        end
        if(length(freq_list)==3)
            plot(thickness(1:4:end),kq(2,1:4:end),'rv-','MarkerFaceColor','r','MarkerSize',2);
            plot(thickness(1:4:end),kq(3,1:4:end),'go-','MarkerFaceColor','g','MarkerSize',2);
            legend([num2str(freq_list(1)),' kHz'], [num2str(freq_list(2)),' kHz'],[num2str(freq_list(3)),' kHz'],'FontSize',10);
        end
        if(length(freq_list)==4)
            plot(thickness(1:4:end),kq(2,1:4:end),'rv-','MarkerFaceColor','r','MarkerSize',2);
            plot(thickness(1:4:end),kq(3,1:4:end),'go-','MarkerFaceColor','g','MarkerSize',2);
            plot(thickness(1:4:end),kq(4,1:4:end),'bd-','MarkerFaceColor','b','MarkerSize',2);
            legend([num2str(freq_list(1)),' kHz'], [num2str(freq_list(2)),' kHz'],[num2str(freq_list(3)),' kHz'],[num2str(freq_list(4)),' kHz'],'FontSize',10);
       
        end
        
        xlabel('h [mm]','FontSize',10,'FontName','Times New Roman');
        ylabel('k [1/m]','FontSize',10,'FontName','Times New Roman');
        ylim([0 500]);
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
        processed_filename = [specimen_name,'_Thickness_sensitivities']; % filename of processed .mat data
        print([figure_output_path,processed_filename],'-dpng', '-r600'); 
    end
    save([dataset_output_path,filesep,'thickness_sensitivities'],'thickness','thickness_sensitivities');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% mask A0 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp('creating mask for A0 mode extraction');
    
    

    mask_width = [linspace(0,mask_width_A0_1,5),linspace(mask_width_A0_1,mask_width_A0_2,length(f_vec)-5)];
      wavenumber_lower_bound = (k_A0_smooth' - 0.5*repmat(mask_width',1,length(beta)))'; 
      wavenumber_upper_bound = (k_A0_smooth' + 0.5*repmat(mask_width',1,length(beta)))';
%       wavenumber_lower_bound = (k_A0_smooth' - 0.4*repmat(mask_width',1,length(beta)))'; 
%       wavenumber_upper_bound = (k_A0_smooth' + 0.6*repmat(mask_width',1,length(beta)))';
%     wavenumber_lower_bound = (k_A0_smooth' - 0.2*repmat(mask_width',1,length(beta)))'; 
%     wavenumber_upper_bound = (k_A0_smooth' + 0.8*repmat(mask_width',1,length(beta)))';
%     wavenumber_lower_bound = (k_A0_smooth' - 0.1*repmat(mask_width',1,length(beta)))'; 
%     wavenumber_upper_bound = (k_A0_smooth' + 0.9*repmat(mask_width',1,length(beta)))';
%     wavenumber_lower_bound = (k_A0_smooth' - 0.05*repmat(mask_width',1,length(beta)))'; 
%     wavenumber_upper_bound = (k_A0_smooth' + 0.95*repmat(mask_width',1,length(beta)))';
     
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
            %polar_mask_A0(n_angle,ind,n_freq) = hann(length(ind));
            % variant 2
            polar_mask_A0(n_angle,ind,n_freq) = 1;
            polar_mask_A0(n_angle,ind(end):ind(end)+offset,n_freq) = 1;
            h41=hann(41);
            polar_mask_A0(n_angle,ind(1):ind(1)+20,n_freq) = h41(1:21);
            polar_mask_A0(n_angle,ind(end)+offset:ind(end)+offset+20,n_freq) = h41(21:41);
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
    f_vec_mask = f_vec;
    save([dataset_output_path,filesep,'cart_mask_A0'],'cart_mask_A0','kx_vec','ky_vec','f_vec_mask','k_A0_smooth','beta','-v7.3');
    % figure;
    % surf(reshape(Vq,n1,m1));shading interp; 
    % figure;
    % surf(squeeze(cart_mask_A0(:,:,100)));shading interp;
    % figure;
    % surf(squeeze(cart_mask_A0_(:,:,100)));shading interp;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% mask slices
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if(radians_flag) 
        [mkx,mky,mf] = meshgrid(kx_vec,ky_vec,f_vec_mask/1e3);
    else
        [mkx,mky,mf] = meshgrid(kx_vec/(2*pi),ky_vec/(2*pi),f_vec_mask/1e3);
    end
    % maxkx = 1000/(2*pi);
    % maxky = 1000/(2*pi);
%     maxkx = 200;
%     maxky = 200;
    maxkx = 400;
    maxky = 400;
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
            line([0,0],[0,0],[0,max(f_vec_mask)],'Color','y','LineWidth',1);
            line([-maxkx -maxkx],[-maxky  maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
            line([-maxkx  maxkx],[ maxky  maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
            line([ maxkx  maxkx],[ maxky -maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
            line([ maxkx -maxkx],[-maxky -maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');


            hold on;
            kx = -k_A0_smooth(round(length(beta)/2)+1,:);      
            plot3(kx,zeros(length(kx),1),f_vec_mask/1e3,'c:','LineWidth',0.5); 

            [~ ,beta_ind] = min(abs(beta - 270));
            ky = -k_A0_smooth(beta_ind,:);      
            plot3(zeros(length(ky),1),ky,f_vec_mask/1e3,'c:','LineWidth',0.5); 

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
            colormap (gray)
            line([0,0],[0,0],[0,max(f_vec_mask)],'Color','y','LineWidth',1);
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
            [~,f_ind]=min(abs(f_vec_mask/1e3 - freq_slice));
            hold on;
            xx = k_A0_smooth(:,f_ind).* cos(beta'*pi/180); 
            yy = k_A0_smooth(:,f_ind).* sin(beta'*pi/180);
            plot3(xx,yy,repmat(freq_slice,[length(xx) 1]),'c:','LineWidth',0.5); 

            %title([num2str(freq_slice),' kHz'],'Fontsize',10,'interpreter','latex');
            text(-maxkx,maxky,freq_slice+0.01*freq_slice,[num2str(freq_slice),' kHz'],'HorizontalAlignment','left','Fontsize',10,'interpreter','latex');
            %view(-20,20)
            %view(-40,50)
            view(-30,50)
            caxis([0 1]);
            set(gcf,'color','white');set(gca,'TickDir','out');
            %set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
            set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
            %set(gcf, 'Units','centimeters', 'Position',[10 10 8 10]);
            set(gcf, 'Units','centimeters', 'Position',[10 10 7.5 10]);
            % remove unnecessary white space
            set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));

            set(gcf,'PaperPositionMode','auto');
            drawnow;
            processed_filename = [specimen_name,'_mask_A0_',num2str(freq_slice),'_kHz']; % filename of processed .mat data
            print([figure_output_path,processed_filename],'-dpng', '-r600'); 
        end
    end
else
    disp('Loading cartesian mask for A0 mode');
    load([dataset_output_path,filesep,'cart_mask_A0']);
    load([dataset_output_path,filesep,'thickness_sensitivities']);
    
end

close all;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Yet another wavenumber damage imaging (YAWDI)
disp('Yet another wavenumber damage imaging (YAWDI)');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Thickness_map_avg = zeros(Nx,Ny); % for average thickness map over frequencies

folder  = raw_data_path;
nFile   = length(test_case);
success = false(1, nFile);
for k = test_case
    filename = list{k};
    processed_filename = [specimen_name,'_RMS_wavenumbers_selected_',filename]; 
    % check if already exist
    if(overwrite||(~overwrite && ~exist([figure_output_path,processed_filename,'.png'], 'file')))
       
            % load raw experimental data file
            disp('loading data');
            load([raw_data_path,filename]); % Data, time, WL
            [nx,ny,nft] = size(Data);
            Width=WL(1);
            Length=WL(2);
            En = sum(reshape(Data,nx*ny,nft).^2,1);
            [~,frame_start] = max(En);
            % beta range for cylindrical coordinates
            % 0-360deg
            dbeta = 360/(4*nx-1);
            beta2 = (dbeta:dbeta:(4*nx)*dbeta)-dbeta;  
            %             0-360deg-dbeta
            %              dbeta = 360/(4*nx);
            %             beta = (dbeta:dbeta:(4*nx)*dbeta)-dbeta;
            Rmax = radius_max(beta2*pi/180,Length,Width);
            %nx=512;ny=512;nft=512;
            %% PROCESS DATA
            fprintf('Processing:\n%s\n',filename);
            if(A0mode_filter)
                disp('3D FFT filtering - A0 mode separation');
                disp('Transform to wavenumber-wavenumber-frequency domain');
                [KXKYF,kx_vec,ky_vec,f_vec] = spatial_to_wavenumber_wavefield_full2(Data,Length,Width,time); % full size data (-kx:+kx,-ky:+ky,-f:+f)
                clear Data;
                
                [mx,my,mf] = size(KXKYF);
                
                % interpolate mask for new frequency vector if needed
                if( sum( (f_vec_mask - f_vec).^2)>1e-6 )
                    disp('interpolate mask for new frequency vector');
                    cart_mask_A0_new = zeros(mx,my,mf);
                    for i=1:mx
                        for j=1:my
                            cart_mask_A0_new(i,j,length(f_vec)+1:end)=interp1(f_vec_mask',squeeze(cart_mask_A0(i,j,length(f_vec_mask)+1:end)),f_vec','linear','extrap');
                        end
                    end           
                    cart_mask_A0_new(:,:,1:length(f_vec)) = flip(cart_mask_A0_new(:,:,length(f_vec)+1:end),3); % flip for neagative frequencies
                    KXKYF_A0 = KXKYF.*cart_mask_A0_new;
                    disp('Interpolation done');
                else
                    KXKYF_A0 = KXKYF.*cart_mask_A0;
                end
                if(freq_filter)
                % frequency mask
                    f0=freq_list(k-1)*1e3;
                    
                    [~,f0_ind]=min(abs(f0-f_vec));
                    [~,fl_ind]=min(abs(f0-fband/2-f_vec));
                    [~,fh_ind]=min(abs(f0+fband/2-f_vec));
                    hw=hann(fh_ind-fl_ind+1);
                    freq_mask = zeros(1,1,mf);
                    freq_mask(1,1,length(f_vec)+fl_ind:length(f_vec)+fh_ind)= hw;
                    freq_mask(1,1,1:length(f_vec)) = flip(freq_mask(1,1,length(f_vec)+1:end),3);% flip for neagative frequencies
                    freq_mask = repmat(freq_mask,[mx my 1]);
                    
                    KXKYF_A0 = KXKYF_A0.*freq_mask;
                end
                
                
                % inverse Fourier transform for pure A0 wavefield
                Data = ifftn(ifftshift(KXKYF_A0),'symmetric'); % wavefield A0
                Data = Data(1:nx,1:ny,1:nft);
            end
            %% plot intermediate results for checking masks
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% KX-KY-F slices
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if(A0mode_filter)
            if(interim_figs)
                if(radians_flag) 
                    [mkx,mky,mf] = meshgrid(kx_vec,ky_vec,f_vec/1e3);
                else
                   [mkx,mky,mf] = meshgrid(kx_vec/(2*pi),ky_vec/(2*pi),f_vec/1e3);  
                end
                freq_slice = freq_list(k-1); % [kHz]
                % maxkx = 1000/(2*pi);
                % maxky = 1000/(2*pi);
%                 maxkx = 200;
%                 maxky = 200;
                maxkx = 400;
                maxky = 400;
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
                hold on;
                kx = -k_A0_smooth(round(length(beta)/2)+1,:);      
                plot3(kx,zeros(length(kx),1),f_vec_mask/1e3,'c:','LineWidth',0.5); 

                [~ ,beta_ind] = min(abs(beta - 270));
                ky = -k_A0_smooth(beta_ind,:);      
                plot3(zeros(length(ky),1),ky,f_vec_mask/1e3,'c:','LineWidth',0.5);
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
%                 lightangle(ax2,-45,45)
%                 lightangle(ax2,-45,45)
                line([0,0],[0,0],[0,max(f_vec)],'Color','y','LineWidth',1);
                line([-maxkx -maxkx],[-maxky  maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                line([-maxkx  maxkx],[ maxky  maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                line([ maxkx  maxkx],[ maxky -maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                line([ maxkx -maxkx],[-maxky -maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                
                 [~,f_ind]=min(abs(f_vec_mask/1e3 - freq_slice));
                hold on;
                xx = k_A0_smooth(:,f_ind).* cos(beta'*pi/180); 
                yy = k_A0_smooth(:,f_ind).* sin(beta'*pi/180);
                plot3(xx,yy,repmat(freq_slice,[length(xx) 1]),'c:','LineWidth',0.5); 
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
                processed_filename = [specimen_name,'_KXKYF_',num2str(freq_slice),'_kHz']; % filename of processed .mat data
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
                hold on;
                kx = -k_A0_smooth(round(length(beta)/2)+1,:);      
                plot3(kx,zeros(length(kx),1),f_vec_mask/1e3,'c:','LineWidth',0.5); 

                [~ ,beta_ind] = min(abs(beta - 270));
                ky = -k_A0_smooth(beta_ind,:);      
                plot3(zeros(length(ky),1),ky,f_vec_mask/1e3,'c:','LineWidth',0.5);
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
%                 lightangle(ax2,-45,45)
%                 lightangle(ax2,-45,45)
                line([0,0],[0,0],[0,max(f_vec)],'Color','y','LineWidth',1);
                line([-maxkx -maxkx],[-maxky  maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                line([-maxkx  maxkx],[ maxky  maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                line([ maxkx  maxkx],[ maxky -maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                line([ maxkx -maxkx],[-maxky -maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
                 [~,f_ind]=min(abs(f_vec_mask/1e3 - freq_slice));
                hold on;
                xx = k_A0_smooth(:,f_ind).* cos(beta'*pi/180); 
                yy = k_A0_smooth(:,f_ind).* sin(beta'*pi/180);
                plot3(xx,yy,repmat(freq_slice,[length(xx) 1]),'c:','LineWidth',0.5); 
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
                processed_filename = [specimen_name,'_KXKYF_A0_',num2str(freq_slice),'_kHz']; % filename of processed .mat data
                print([figure_output_path,processed_filename],'-dpng', '-r600'); 
               
            end
            end
            %% cylindrical coordinate  
            %[Data_polar,number_of_points,radius] =
            %cartesian_to_polar_wavefield_2pi2(Data,WL(1),WL(2),beta2);%slow - for scattered data
            [Data_polar,number_of_points,radius] = cartesian_to_polar_wavefield_2pi_gridded2(Data,WL(1),WL(2),beta2);%fast - for data on regural 
            [number_of_angles,number_of_points,number_of_time_steps]=size(Data_polar);
            %% spatial signal at selected angle and time
            %N = 2^nextpow2(number_of_points);
            N = number_of_points;
            wavenumbers = zeros(number_of_angles,N-1,length(selected_frames{k-1}));
            Amplitude = zeros(number_of_angles,N,length(selected_frames{k-1}));
            
            
            x=zeros(number_of_angles,N);
            y=zeros(number_of_angles,N);
            b = beta2*pi/180;
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
                Windowing_mask(n_angle,1:11) = Hn(1:11); % smooth out begining of the signal
%                 Windowing_mask(n_angle,usable_sig_length(n_angle)-10:usable_sig_length(n_angle)) = Hn(11:21);
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% main algorithm
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            xr=R; % the same radius for each angle
            % Numerical approximation of the Hilbert transform in the FFT domain:
            W = (-floor(N/2):ceil(N/2)-1)/N; % wavenumber coordinates normalized to interval -0.5:0.5
            H = ifftshift(  -1i * sign(W)  ); % sampled Fourier response
            c=0;
            for frame = selected_frames{k-1}
                [frame]
                c=c+1;
                wavenumbers_by_angle = zeros(number_of_angles,N-1);
                amplitude_by_angle = zeros(number_of_angles,N);
                s = zeros(number_of_angles,N);
                s(:,1:number_of_points) = squeeze(Data_polar(:,1:number_of_points,frame)).*Windowing_mask;
                %s(:,1:number_of_points) = squeeze(Data_polar(:,1:number_of_points,frame));
                %parfor n_angle=1:number_of_angles
                for n_angle=1:number_of_angles
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
%                    unwrapped_phase_flat_smooth = movmean(unwrapped_phase_flat,10);
                        unwrapped_phase_flat_smooth1 = movmean(unwrapped_phase_flat(1:nx),10);
                        unwrapped_phase_flat_smooth2 = movmean(unwrapped_phase_flat(nx+1-40:end),40); % larger window at the end
                        dh=unwrapped_phase_flat_smooth2(19)-unwrapped_phase_flat_smooth1(nx-20);
                        unwrapped_phase_flat_smooth = [unwrapped_phase_flat_smooth1(1:nx-20),unwrapped_phase_flat_smooth2(19:end-2)-dh];
                        % smoothing second pass
                        unwrapped_phase_flat_smooth = movmean(unwrapped_phase_flat_smooth,10);
                    
                    hd = diff(unwrapped_phase_flat_smooth+yfit)/dr; % first derivative
                    % insert nan for phase (wavenumbers) at zero amplitude
                    amp_norm=amp/max(amp);
                    
                    Inan=find(amp_norm<amp_threshold(k-1));
                    if(~isempty(Inan))
                        if(Inan(end) == length(amp)) % check if we are in the range of diff which is one point shorter
                            Inan(end) = [];
                        end
                        hd(Inan) = NaN;
                    end
                    wavenumbers_by_angle(n_angle,:) = movmean(hd,5,'omitnan'); % unwrapped phase
                    amplitude_by_angle(n_angle,:) = amp_smoothed;
                end
                
                % smoothing over angle
                wavenumbers_by_angle_smoothed = movmean(wavenumbers_by_angle,10,1,'omitnan');
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
            close all;
            %fgh1=figure('Position',[1 1 1920 1000]) ;
            fgh1=figure;
            set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
            set(fgh1, 'Units','centimeters', 'Position',[10 10 16 7.5]); 
            set(fgh1, 'color', 'white'); 
            
            % remove unnecessary white space
            %set(gca,'LooseInset', max(get(gca,'TightInset'), 0.04));
            c=0;
            for frame = selected_frames{k-1}  
                c=c+1;
                
                t=tiledlayout(1,2);
                t.TileSpacing = 'tight';
                %t.TileSpacing = 'none';
                %t.Padding = 'tight';
                t.Padding = 'loose';
                sbh1 = nexttile;
                surf(sbh1,x(:,1:end-radius_cut_wavenumbers-1),y(:,1:end-radius_cut_wavenumbers-1),squeeze(wavenumbers(:,1:end-radius_cut_wavenumbers,c)));shading interp; view(2);  c1=colorbar; colormap(Cmap);       
                Smax=max(max(squeeze(wavenumbers(:,1:end-radius_cut_wavenumbers,c))));Smin=min(min(squeeze(wavenumbers(:,1:end-radius_cut_wavenumbers,c))));
                set(gcf,'Renderer','zbuffer');
                xlim([-0.25 0.25]);
                ylim([-0.25, 0.25]);
                %axis equal;
                axis square;
                
                box on;
                set(sbh1,'Fontsize',10);
                c1.FontSize = 10;
                %caxis([caxis_cut*Smin,caxis_cut*Smax]);   
                %caxis([Smin,Smax]); 
                
                switch k
                    case 1           
                        %caxis([200 1000]); % chirp
                    case 2
                        caxis([0 80]); % 50 kHz
                    case 3
                        caxis([0 100]); % 75 kHz
                    case 4
                        caxis([0 150]); % 100 kHz
                    case 5
                        caxis([0 180]); % 150 kHz
                end
                %caxis([200 800]); 
                %caxis([-100 500]); 
                %title(['Wavenumber fn= ',num2str(frame),' ',num2str(freq_list(k-1)),' kHz']);
                
                
                sbh2 = nexttile;
                surf(sbh2,x(:,1:end-radius_cut_amplitude),y(:,1:end-radius_cut_amplitude),squeeze(Amplitude(:,1:end-radius_cut_amplitude,c)));shading interp; view(2); c2=colorbar; colormap(Cmap);       
                Smax=max(max(squeeze(Amplitude(:,1:end-radius_cut_amplitude,c))));Smin=0;
                set(fgh1,'Renderer','zbuffer');
                xlim([-0.25 0.25]);
                ylim([-0.25, 0.25]);
                
                axis square;  
                box on;
                set(sbh2,'Fontsize',10);
                c2.FontSize = 10;
                %caxis([caxis_cut*Smin,caxis_cut*Smax]);    
                
                switch k
                    case 1           
                        caxis([0 8e-3]);  % chirp
                    case 2
                        caxis([0 0.01]);  % 50 kHz
                    case 3
                        caxis([0 0.02]); % 75 kHz
                    case 4
                        caxis([0 0.03]); % 100 kHz
                    case 5
                        caxis([0 0.04]);  % 150 kHz
                end
                %title(['Amplitude fn= ',num2str(frame),' ',num2str(freq_list(k-1)),' kHz']);
                
                drawnow;
                pause(0.1);
                processed_filename = ['Hilbert_frame_',num2str(frame),'_freq_',num2str(freq_list(k-1)),' kHz'];
                print([figure_output_path,processed_filename],'-dpng', '-r600'); 
            end
%}            
            % selected frames
            RMS_wavenumbers_selected = sqrt(sum(wavenumbers.^2,3,'omitnan'))/length(selected_frames{k-1});
            Mean_wavenumbers_selected = mean(wavenumbers,3,'omitnan');
            % check if we still have NaNs and replace by mean
            Mean_wavenumbers_selected(isnan(Mean_wavenumbers_selected))=mean(Mean_wavenumbers_selected,'all','omitnan');
            
            RMS_amplitude_selected = sqrt(sum(Amplitude.^2,3))/length(selected_frames{k-1});
            Mean_amplitude_selected = mean(Amplitude,3);
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % convert back to cartesian coordinates by using linear interpolation
            [TH,Radius] = meshgrid(beta2*pi/180,R(1:end-1));
            [Xk,Yk,Zk] = pol2cart(TH,Radius,Mean_wavenumbers_selected');
            %figure;surf(Xk,Yk,Zk);shading interp;view(2);
            F = scatteredInterpolant(reshape(Xk,[],1),reshape(Yk,[],1),reshape(Zk,[],1),'linear','none'); % requires ndgrid format; no extrapolation
            [XI,YI] = meshgrid(linspace(-WL(1)/2,WL(1)/2,Nx),linspace(-WL(2)/2,WL(2)/2,Ny)); % 
            Data_cart=F(XI,YI);Data_cart(isnan(Data_cart))=0;
            %figure;surf(XI,YI,Data_cart);shading interp;view(2);xlim([-0.25,0.25]);ylim([-0.25 0.25]);axis square;
            %Mean_wavenumbers_selected_smooth = medfilt2(Data_cart,[16,16]);
            %Mean_wavenumbers_selected_smooth = medfilt2(Data_cart,[8,8]);
            Mean_wavenumbers_selected_smooth = Data_cart; % no median filter
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
            if(damage_outline) 
                plot3(delam1(:,1),delam1(:,2),repmat(Smax,[length(delam1),1]),'k:','LineWidth',0.5); 
                plot3(delam2(:,1),delam2(:,2),repmat(Smax,[length(delam2),1]),'k:','LineWidth',0.5); 
                plot3(delam3(:,1),delam3(:,2),repmat(Smax,[length(delam3),1]),'k:','LineWidth',0.5); 
            end
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
            if(A0mode_filter)
                processed_filename = [specimen_name,'_RMS_wavenumbers_selected_',filename]; % filename of processed .mat data
            else
                processed_filename = [specimen_name,'_unfiltered_RMS_wavenumbers_selected_',filename]; % filename of processed .mat data
            end
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
            if(damage_outline) 
                plot3(delam1(:,1),delam1(:,2),repmat(Smax,[length(delam1),1]),'k:','LineWidth',0.5); 
                plot3(delam2(:,1),delam2(:,2),repmat(Smax,[length(delam2),1]),'k:','LineWidth',0.5); 
                plot3(delam3(:,1),delam3(:,2),repmat(Smax,[length(delam3),1]),'k:','LineWidth',0.5); 
            end
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
            if(A0mode_filter)
                processed_filename = [specimen_name,'_Mean_wavenumbers_selected_',filename]; % filename of processed .mat data
            else
                processed_filename = [specimen_name,'_unfiltered_Mean_wavenumbers_selected_',filename]; % filename of processed .mat data
            end
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
            if(damage_outline) 
                plot3(delam1(:,1),delam1(:,2),repmat(Smax,[length(delam1),1]),'k:','LineWidth',0.5); 
                plot3(delam2(:,1),delam2(:,2),repmat(Smax,[length(delam2),1]),'k:','LineWidth',0.5); 
                plot3(delam3(:,1),delam3(:,2),repmat(Smax,[length(delam3),1]),'k:','LineWidth',0.5); 
            end
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
            if(A0mode_filter)
                processed_filename = [specimen_name,'_Mean_wavenumbers_selected_smooth_',filename]; % filename of processed .mat data
            else
                processed_filename = [specimen_name,'_unfiltered_Mean_wavenumbers_selected_smooth_',filename]; % filename of processed .mat data
            end
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
            if(damage_outline) 
                plot3(delam1(:,1),delam1(:,2),repmat(caxis_cut*Smax,[length(delam1),1]),'k:','LineWidth',0.5); 
                plot3(delam2(:,1),delam2(:,2),repmat(caxis_cut*Smax,[length(delam2),1]),'k:','LineWidth',0.5); 
                plot3(delam3(:,1),delam3(:,2),repmat(caxis_cut*Smax,[length(delam3),1]),'k:','LineWidth',0.5); 
            end
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
            if(A0mode_filter)
                processed_filename = [specimen_name,'_RMS_amplitude_selected_',filename]; % filename of processed .mat data
            else
                processed_filename = [specimen_name,'_unfiltered_RMS_amplitude_selected_',filename]; % filename of processed .mat data
            end
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
            if(damage_outline) 
                plot3(delam1(:,1),delam1(:,2),repmat(caxis_cut*Smax,[length(delam1),1]),'k:','LineWidth',0.5); 
                plot3(delam2(:,1),delam2(:,2),repmat(caxis_cut*Smax,[length(delam2),1]),'k:','LineWidth',0.5); 
                plot3(delam3(:,1),delam3(:,2),repmat(caxis_cut*Smax,[length(delam3),1]),'k:','LineWidth',0.5);  
            end
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
            if(A0mode_filter)
                processed_filename = [specimen_name,'_Mean_amplitude_selected_',filename]; % filename of processed .mat data
            else
                processed_filename = [specimen_name,'_unfilteredMean_amplitude_selected_',filename]; % filename of processed .mat data
            end
            print([figure_output_path,processed_filename],'-dpng', '-r600'); 
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% wavenumber to thickness scaling
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if(A0mode_filter)
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
                        if(Mean_wavenumbers_selected_smooth(i,j)<min(thickness_sensitivities(I,:,k-1)))
                            Thickness_map(i,j) = 1.5*base_thickness;
                        else
%                             [~,J] = min( abs( thickness_sensitivities(I,:,k-1)-Mean_wavenumbers_selected_smooth(i,j) ) );
%                             Thickness_map(i,j) = thickness(J);
                            Thickness_map(i,j)=interp1(squeeze(thickness_sensitivities(I,:,k-1)),thickness,Mean_wavenumbers_selected_smooth(i,j),'linear','extrap');

                        end
                    end
                end
                Thickness_map_avg = Thickness_map_avg + Thickness_map/length(freq_list);
                fgh=figure;
                axh = axes('Parent',fgh);
                surf(XI,YI,Thickness_map);shading interp; view(2); colorbar; colormap(flipud(Cmap));       
                Smax=max(max(Thickness_map));Smin=min(min(Thickness_map));
                set(fgh,'Renderer','zbuffer');
                xlim([-0.25 0.25]);
                ylim([-0.25, 0.25]);
                set(gca,'Fontsize',10);
                axis square;
                hold on;
                if(damage_outline) 
                    plot3(delam1(:,1),delam1(:,2),repmat(1.5*base_thickness,[length(delam1),1]),'k:','LineWidth',0.5); 
                    plot3(delam2(:,1),delam2(:,2),repmat(1.5*base_thickness,[length(delam2),1]),'k:','LineWidth',0.5); 
                    plot3(delam3(:,1),delam3(:,2),repmat(1.5*base_thickness,[length(delam3),1]),'k:','LineWidth',0.5); 
                end
                %caxis([0 5]);   
                %caxis([Smin 3]); 
                caxis([0.5*base_thickness 1.5*base_thickness]);
                %caxis([0 4.5e-3]); 
                %title(['Mean amplitude']);
                set(fgh,'color','white');set(gca,'TickDir','out');
                
                set(axh, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
                set(fgh, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
                % remove unnecessary white space
                set(axh,'LooseInset', max(get(gca,'TightInset'), 0.02));

                set(fgh,'PaperPositionMode','auto');
                drawnow;       
                processed_filename = [specimen_name,'_Thickness_map_',filename]; % filename of processed .mat data     
                print([figure_output_path,processed_filename],'-dpng', '-r600');
                
                fgh2=figure;
                axh2 = axes('Parent',fgh2);
                surf(XI,YI,Thickness_map);shading interp; view(2); colorbar; colormap(flipud(Cmap));       
                Smax=max(max(Thickness_map));Smin=min(min(Thickness_map));
                set(fgh2,'Renderer','zbuffer');
                xlim([-0.25 0.25]);
                ylim([-0.25, 0.25]);
                set(gca,'Fontsize',10);
                axis square;
                hold on;
                if(damage_outline) 
                    plot3(delam1(:,1),delam1(:,2),repmat(1.1*base_thickness,[length(delam1),1]),'k:','LineWidth',0.5); 
                    plot3(delam2(:,1),delam2(:,2),repmat(1.1*base_thickness,[length(delam2),1]),'k:','LineWidth',0.5); 
                    plot3(delam3(:,1),delam3(:,2),repmat(1.1*base_thickness,[length(delam3),1]),'k:','LineWidth',0.5); 
                end
                %caxis([0 5]);   
                %caxis([Smin 3]); 
                caxis([0.5*base_thickness 1.1*base_thickness]);
                %caxis([0 4.5e-3]); 
                %title(['Mean amplitude']);
                set(fgh2,'color','white');set(gca,'TickDir','out');
                
                set(axh2, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
                set(fgh2, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
                % remove unnecessary white space
                set(axh2,'LooseInset', max(get(gca,'TightInset'), 0.02));

                set(fgh2,'PaperPositionMode','auto');
                drawnow;       
                processed_filename = [specimen_name,'_Thickness_map2_',filename]; % filename of processed .mat data     
                print([figure_output_path,processed_filename],'-dpng', '-r600');
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% END OF PROCESSING
            [filepath,name,ext] = fileparts(filename);
            fprintf('Successfully processed:\n%s\n', name);% successfully processed
        
    else
        fprintf('Filename: \n%s \nalready exist\n', processed_filename);
    end
    close all;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(A0mode_filter)
    figure;
    surf(XI,YI,Thickness_map_avg);shading interp; view(2); colorbar; colormap(flipud(Cmap));       
    Smax=max(max(Thickness_map_avg));Smin=min(min(Thickness_map_avg));
    set(gcf,'Renderer','zbuffer');
    xlim([-0.25 0.25]);
    ylim([-0.25, 0.25]);
    set(gca,'Fontsize',10);
    axis square;
    hold on;
    if(damage_outline) 
        plot3(delam1(:,1),delam1(:,2),repmat(1.5*base_thickness,[length(delam1),1]),'k:','LineWidth',0.5); 
        plot3(delam2(:,1),delam2(:,2),repmat(1.5*base_thickness,[length(delam2),1]),'k:','LineWidth',0.5); 
        plot3(delam3(:,1),delam3(:,2),repmat(1.5*base_thickness,[length(delam3),1]),'k:','LineWidth',0.5); 
    end
    %caxis([0 5]); 
    %caxis([Smin 3]); 
    caxis([0.5*base_thickness 1.5*base_thickness]);
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
    processed_filename = [specimen_name,'_Thickness_map_avg']; % filename of processed .mat data
    print([figure_output_path,processed_filename],'-dpng', '-r600');
    figure;
    surf(XI,YI,Thickness_map_avg);shading interp; view(2); colorbar; colormap(flipud(Cmap));       
    Smax=max(max(Thickness_map_avg));Smin=min(min(Thickness_map_avg));
    set(gcf,'Renderer','zbuffer');
    xlim([-0.25 0.25]);
    ylim([-0.25, 0.25]);
    set(gca,'Fontsize',10);
    axis square;
    hold on;
    if(damage_outline) 
        plot3(delam1(:,1),delam1(:,2),repmat(1.1*base_thickness,[length(delam1),1]),'k:','LineWidth',0.5); 
        plot3(delam2(:,1),delam2(:,2),repmat(1.1*base_thickness,[length(delam2),1]),'k:','LineWidth',0.5); 
        plot3(delam3(:,1),delam3(:,2),repmat(1.1*base_thickness,[length(delam3),1]),'k:','LineWidth',0.5); 
    end
    %caxis([0 5]); 
    %caxis([Smin 3]); 
    caxis([0.5*base_thickness 1.1*base_thickness]);
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
    processed_filename = [specimen_name,'_Thickness_map_avg2']; % filename of processed .mat data
    print([figure_output_path,processed_filename],'-dpng', '-r600');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% End of Yet another wavenumber damage imaging (YAWDI)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
toc
  