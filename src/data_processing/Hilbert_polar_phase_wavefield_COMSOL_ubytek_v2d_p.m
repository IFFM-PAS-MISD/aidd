clear all;close all;   warning off;clc;
tic
load project_paths projectroot src_path;
%% Prepare output directories
% allow overwriting existing results if true
%overwrite=false;
overwrite=true;
% retrieve model name based on running file and folder
currentFile = mfilename('fullpath');
[pathstr,name,ext] = fileparts( currentFile );
idx = strfind( pathstr,filesep );
modelfolder = pathstr(idx(end)+1:end); % name of folder
modelname = name; 
% prepare output paths
dataset_output_path = prepare_data_processing_paths('processed','num',modelname);
figure_output_path = prepare_figure_paths(modelname);
%image_label_path=fullfile(projectroot,'data','interim','exp',filesep);
image_label_path='/pkudela_odroid_sensors/aidd/data/interim/exp/new_exp/';

test_case=[4]; % select file numbers for processing

%% input for figures
Cmap = jet(256); 
caxis_cut = 0.8;
fig_width =6; % figure widht in cm
fig_height=6; % figure height in cm
%% Input for signal processing

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
selected_frames=200:280; % selected frames for Hilbert transform
N = 1024;% for zero padding

%% input for mask
mask_width_A0_1=120/2; % half wavenumber band width
mask_width_A0_2=200/2;
%%
% create path to the experimental raw data folder

raw_data_path = ['/pkudela_odroid_laser/COMSOL_ubytek/'];

% create path to the numerical interim data folder
interim_data_path = fullfile( projectroot, 'data','interim','num', filesep );

% create path to the numerical processed data folder
processed_data_path = fullfile( projectroot, 'data','processed','num', filesep );

% full field measurements
list = {'4_chirp','Data50','Data75','Data100','Data150'};

% load chirp signal
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
            
% cartesian to polar            
% 0-360deg
dbeta = 360/(m1-1);
beta = (dbeta:dbeta:(m1)*dbeta)-dbeta; 
[KXKYF_polar,~,k_radius] = cartesian_to_polar_wavefield_2pi_gridded2(KXKYF(:,:,nft+1:end),2*kxmax,2*kymax,beta);%fast - for data on regural 
%clear KXKYF;
[m2,n2,nft2]=size(KXKYF_polar);
%figure;surf(abs(squeeze(KXKYF_polar(:,:,100))));shading interp; view(2);      
figure;surf(abs(squeeze(KXKYF_polar(100,:,:))));shading interp; view(2);  

return;
%[TH,Radius] = meshgrid(beta*pi/180,linspace(0,k_radius,n2));
% [Xk,Yk,Zk] = pol2cart(TH,Radius,KXKYF_polar(:,:,100));
% figure;surf(Xk,Yk,abs(Zk)');shading interp;view(2);

%figure;surf(abs(squeeze(KXKYF_polar_smooth(:,:,280))));shading interp; view(2);
% ridge picking
% frequency range 5-150 kHz
f_start = 5000;
f_end = 150000;
[~,f_start_ind] = min(abs(f_vec-f_start));
[~,f_end_ind] = min(abs(f_vec-f_end));
Ind = zeros(f_end_ind - f_start_ind+1,1);
k_vec = linspace(0,k_radius,n2);
k_A0 = zeros(length(beta),length(f_vec));
n_outliers = round(0.1*(f_end_ind - f_start_ind+1)); % number of data points to remove

% coefficient range for fitting a curve of the form Y=a*X^b
a=[0.3:0.01:0.6]; 
b=[0.4:0.01:0.6];
disp('Extracting A0 mode');
for n_angle = 1:length(beta)
    [n_angle,length(beta)]
    n_freq = f_start_ind;
    c=1;
    [~,I] = max(abs(squeeze(KXKYF_polar(n_angle,:,n_freq))));
    Ind(c) = I;
    for n_freq = f_start_ind+1:f_end_ind
        c=c+1;
        [~,I] = max(abs(squeeze(KXKYF_polar(n_angle,:,n_freq))));
        Ind(c) =I;
    end
    k_new=movmean(k_vec(Ind),4);
    % remove outliers
    [p2]=polyfit([f_vec(f_start_ind:f_end_ind)],[k_new(1:end)],2);
    yfit2 = polyval(p2,f_vec(f_start_ind:f_end_ind));
    [kd]=abs(yfit2-k_new);
    [~,J]=sort(kd,'ascend');
    correct_data_ind=(sort(J(1:end-n_outliers)));
    k_new = k_new(correct_data_ind);
    % fit a curve
    RMS_temp=1e12;
    for j1=1:length(a)
        for j2=1:length(b)
            Y = a(j1).*f_vec(f_start_ind+correct_data_ind).^b(j2);
            RMS=sum(sqrt((Y-k_new).^2));
            if(RMS<RMS_temp) 
                RMS_temp = RMS;
                a_selected = a(j1);
                b_selected = b(j2);
            end
        end
    end
    Y = a_selected.*f_vec.^b_selected;   
    k_A0(n_angle,:) = Y;

end
k_A0_smooth = movmean([k_A0;k_A0(1:50,:)],50,1); % wrap around data for improved smoothing
k_A0_smooth = k_A0_smooth(1:length(beta),:);
clear k_A0;
clear KXKYF_polar;
% figure;
% for n_angle = 1:length(beta)
%     plot(f_vec/1e3,k_A0_smooth(n_angle,:),'r');hold on;
% end
% figure;
% for j = 1:length(f_vec)
%     plot3( k_A0_smooth(:,j).*cos(beta'*pi/180),k_A0_smooth(:,j).*sin(beta'*pi/180), repmat(f_vec(j),[length(beta),1]),'r.'); hold on;
% end
% figure;
% for j = 1:length(f_vec)
%     plot3( k_A0_smooth(1:10:end,j).*cos(beta(1:10:end)'*pi/180),k_A0_smooth(1:10:end,j).*sin(beta(1:10:end)'*pi/180), repmat(f_vec(j),[length(beta(1:10:end)),1]),'r.'); hold on;
% end
% figure;surf(k_A0_smooth);shading interp;

%% mask A0 
disp('creating mask for A0 mode extraction');
mask_width = [linspace(0,mask_width_A0_1,5),linspace(mask_width_A0_1,mask_width_A0_2,length(f_vec)-5)];
wavenumber_lower_bound = (k_A0_smooth' - repmat(mask_width',1,length(beta)))'; 
wavenumber_upper_bound = (k_A0_smooth' + repmat(mask_width',1,length(beta)))';

figure;
surf(k_A0_smooth);shading interp; hold on;
surf(wavenumber_lower_bound);shading interp; 
surf(wavenumber_upper_bound);shading interp; 

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
figure;
surf(squeeze(polar_mask_A0(end,:,:)));shading interp;
disp('converting mask from cylindrical to cartesian coordinates');
% convert back to cartesian coordinates by using linear interpolation
[TH,Radius] = meshgrid(beta*pi/180,ka);
[Xk,Yk,Zk] = pol2cart(TH',Radius',polar_mask_A0);
%figure;surf(Xk,Yk,Zk(:,:,100));shading interp;view(2);
[XI,YI] = meshgrid(linspace(-kxmax,kxmax,m1),linspace(-kymax,kymax,n1)); % 
cart_mask_A0 = zeros(n1,m1,2*length(f_vec));
for n_freq = 1:length(f_vec)
    [n_freq length(f_vec)]
    F = scatteredInterpolant(reshape(Xk,[],1),reshape(Yk,[],1),reshape(squeeze(Zk(:,:,n_freq)),[],1),'linear','none'); % requires ndgrid format; no extrapolation
    %F = scatteredInterpolant(reshape(Xk,[],1),reshape(Yk,[],1),reshape(squeeze(Zk(:,:,n_freq)),[],1),'nearest','none');
    
    cart_mask_A0(:,:,length(f_vec)+n_freq)=F(XI,YI);
end
cart_mask_A0(:,:,1:length(f_vec)) = flip(cart_mask_A0(:,:,length(f_vec)+1:end),3); % flip for neagative frequencies
cart_mask_A0(isnan(cart_mask_A0))=0;
% phi = atan2(Y,X);
% r = sqrt(X.^2+Y.^2);            
%% convert back to cartesian coordinates by using linear interpolation
% Delaunay triangulation approach

figure;
surf(squeeze(cart_mask_A0(:,:,150)));shading interp;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% mask slices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[mkx,mky,mf] = meshgrid(kx_vec/(2*pi),ky_vec/(2*pi),f_vec/1e3);
freq_slice = 100; % [kHz]
% maxkx = 1000/(2*pi);
% maxky = 1000/(2*pi);
maxkx = 200;
maxky = 200;
xslice1 = []; yslice1 = []; zslice1 = freq_slice;
xslice2 = 0; yslice2 = 0; zslice2 = [];

figure;
%h = slice(permute(mkx,[3,2,1]),permute(mky,[3,2,1]),permute(mf,[3,2,1]),permute(cart_mask_A0(:,:,end/2+1:end),[3,2,1]),xslice,yslice,zslice);
t=tiledlayout(2,1);
%t.TileSpacing = 'tight';
t.TileSpacing = 'none';
t.Padding = 'tight';
% Top plot
ax1 = nexttile;

h = slice(ax1,mkx,mky,mf,cart_mask_A0(:,:,end/2+1:end),xslice2,yslice2,zslice2);

set(h,'FaceColor','interp','EdgeColor','none'); set(gcf,'Renderer','zbuffer');
hold on;
%ylabel({'$k_y$ [1/m]'},'Rotation',-37,'Fontsize',10,'interpreter','latex');% for 8cm figure
ylabel({'$k_y$ [1/m]'},'Rotation',-38,'Fontsize',10,'interpreter','latex');% for 7.5cm figure
xlabel({'$k_x$ [1/m]'},'Rotation', 15,'Fontsize',10,'interpreter','latex');
zlabel({'$f$ [kHz]'},'Fontsize',10,'interpreter','latex')
set(gca,'Fontsize',8,'linewidth',1);
set(gca,'FontName','Times');
grid(ax1,'off');
view(3);
lightangle(-45,45)
lightangle(-45,45)
colormap (gray)
line([0,0],[0,0],[0,max(f_vec)],'Color','y','LineWidth',1);
line([-maxkx -maxkx],[-maxky  maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
line([-maxkx  maxkx],[ maxky  maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
line([ maxkx  maxkx],[ maxky -maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');
line([ maxkx -maxkx],[-maxky -maxky],[freq_slice,freq_slice],'Color','r','LineWidth',1,'LineStyle','--');

xlim([-maxkx maxkx])
ylim([-maxky maxky])
zlim([0 500])
box on; ax = gca; ax.BoxStyle = 'full';
%view(-20,20)
%view(-40,15)
view(-30,50)

% bottom plot
ax2 = nexttile;

h = slice(ax2,mkx,mky,mf,cart_mask_A0(:,:,end/2+1:end),xslice1,yslice1,zslice1);

set(h,'FaceColor','interp','EdgeColor','none'); set(gcf,'Renderer','zbuffer');
hold on;
%ylabel({'$k_y$ [1/m]'},'Rotation',-37,'Fontsize',10,'interpreter','latex');% for 8cm figure
ylabel({'$k_y$ [1/m]'},'Rotation',-38,'Fontsize',10,'interpreter','latex');% for 7.5cm figure
xlabel({'$k_x$ [1/m]'},'Rotation', 15,'Fontsize',10,'interpreter','latex');
zlabel({'$f$ [kHz]'},'Fontsize',10,'interpreter','latex')
set(gca,'Fontsize',8,'linewidth',1);
set(gca,'FontName','Times');

view(3);
lightangle(-45,45)
lightangle(-45,45)
colormap (gray)
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


set(gcf,'color','white');
%set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
%set(gcf, 'Units','centimeters', 'Position',[10 10 8 10]);
set(gcf, 'Units','centimeters', 'Position',[10 10 7.5 10]);
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));

set(gcf,'PaperPositionMode','auto');
drawnow;
processed_filename = ['mask_A0_',filename]; % filename of processed .mat data
print([figure_output_path,processed_filename],'-dpng', '-r600'); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figures showing mask and wavefield in wavenumber-frequency domain

disp('3D FFT filtering - A0 mode separation');


disp('Yet another wavenumber damage imaging (YAWDI)');
folder  = raw_data_path;
nFile   = length(test_case);
success = false(1, nFile);
for k = test_case
    filename = list{k};
    processed_filename = ['RMS_wavenumbers_selected_',filename]; 
    % check if already exist
    if(overwrite||(~overwrite && ~exist([figure_output_path,processed_filename,'.png'], 'file')))
       
            % load raw experimental data file
            disp('loading data');
            load([raw_data_path,filename]); % Data, (time XI YI ZI)
            [nx,ny,nft] = size(Data);
            %nx=512;ny=512;nft=512;
            %% PROCESS DATA
            fprintf('Processing:\n%s\n',filename);
            %% cylindrical coordinate
            % 0-360deg
             dbeta = 360/(4*nx-1);
             beta = (dbeta:dbeta:(4*nx)*dbeta)-dbeta;  
%             0-360deg-dbeta
%              dbeta = 360/(4*nx);
%             beta = (dbeta:dbeta:(4*nx)*dbeta)-dbeta;  
            %[Data_polar,number_of_points,radius] =
            %cartesian_to_polar_wavefield_2pi2(Data,WL(1),WL(2),beta);%slow - for scattered data
            [Data_polar,number_of_points,radius] = cartesian_to_polar_wavefield_2pi_gridded2(Data,WL(1),WL(2),beta);%fast - for data on regural 
            %    dimensions [number_of_angles,number_of_points,number_of_time_steps]
            %save('Data_polar','Data_polar','beta','radius','-v7.3');
%             disp('loading polar data');
%             load('Data_polar');
            [number_of_angles,number_of_points,number_of_time_steps]=size(Data_polar);
            %% spatial signal at selected angle and time
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
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% main algorithm
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            xr=R; % the same radius for each angle
            % Numerical approximation of the Hilbert transform in the FFT domain:
            W = (-floor(N/2):ceil(N/2)-1)/N; % wavenumber coordinates normalized to interval -1:1
            H = ifftshift(  -1i * sign(W)  ); % sampled Fourier response
            c=0;
            for frame = selected_frames
                [frame]
                c=c+1;
                wavenumbers_by_angle = zeros(number_of_angles,N-1);
                amplitude_by_angle = zeros(number_of_angles,N);
                
                
                s = zeros(number_of_angles,N);
                s(:,1:number_of_points) = squeeze(Data_polar(:,1:number_of_points,frame));
                
                    % add tapering window at the end of signal
%                     if(beta(n_angle)>90 && beta(n_angle) <180)
%                         [~,nn] = min(abs(R-WL(2)/2/sin(b(n_angle)-pi/2)));
%                     end
%                     s(nn:end) = linspace(s(nn-1),0,N-nn+1);
                    
                    
                parfor n_angle=1:number_of_angles
 
                    % FFT-domain Hilbert transform of the input signal 's':
                    hilb = real(ifft(  fft(s(n_angle,:)) .* H  )); 
                    
                    sa = s(n_angle,:) + 1i*hilb; % complex valued analytic signal associated to input signal
                    amp = abs(sa);    % instantaneous amplitude envelope
                    phz = angle(sa);  % instantaneous phase
                    amp_smoothed = movmean(amp,20); % moving mean for smoothing amplitude
                    
                    %
                    unwrapped_phase = unwrap(phz); % unwrapped phase
                    
                    
                    [p]=polyfit(xr(10:round(nx/2)),unwrapped_phase(10:round(nx/2)),1);
                    yfit = polyval(p,xr);
                    unwrapped_phase_flat = unwrapped_phase-yfit;
                    unwrapped_phase_flat_smooth = movmean(unwrapped_phase_flat,10);
                   
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
                   %wavenumbers(:,:,c) = medfilt2(wavenumbers(:,:,c),[16 4]);
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% end of main algorithm
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % remove central point
            Amplitude(:,1,:)=0;
            wavenumbers(:,1,:)=0;
            radius_cut_wavenumbers = 40;
            radius_cut_amplitude = 0;
%{
            figure('Position',[1 1 1920 1000])         
            c=0;
            for frame = selected_frames  
                c=c+1;
                subplot(1,2,1);
                surf(x(:,1:end-radius_cut_wavenumbers),y(:,1:end-radius_cut_wavenumbers),squeeze(wavenumbers(:,1:end-radius_cut_wavenumbers+1,c)));shading interp; view(2);  colorbar; colormap(Cmap);       
                Smax=max(max(squeeze(wavenumbers(:,1:end-radius_cut_wavenumbers+1,c))));Smin=min(min(squeeze(wavenumbers(:,1:end-radius_cut_wavenumbers+1,c))));
                set(gcf,'Renderer','zbuffer');
                xlim([-0.25 0.25]);
                ylim([-0.25, 0.25]);
                %axis equal;
                axis square;
                %caxis([caxis_cut*Smin,caxis_cut*Smax]);   
                %caxis([Smin,Smax]); 
                
                switch k
                    case 1           
                        caxis([200 1000]); % chirp
                    case 2
                        caxis([300 500]); % 50 kHz
                    case 3
                        caxis([350 650]); % 75 kHz
                    case 4
                        caxis([400 700]); % 100 kHz
                    case 5
                        caxis([550 850]); % 150 kHz
                end
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
                
                switch k
                    case 1           
                        caxis([0 8e-3]);  % chirp
                    case 2
                        caxis([0 7e-3]);  % 50 kHz
                    case 3
                        caxis([0 7e-3]); % 75 kHz
                    case 4
                        caxis([0 7e-3]);  % 100 kHz
                    case 5
                        caxis([0 7e-3]);  % 150 kHz
                end
                title(['Amplitude f= ',num2str(frame)]);
                pause(0.1);
            end
%}            
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
            %figure;surf(XI,YI,Mean_wavenumbers_refined);shading interp;view(2);xlim([-0.25,0.25]);ylim([-0.25 0.25]);axis square;
           
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
           
            axis square;
            %caxis([caxis_cut*Smin,caxis_cut*Smax]);  
            %caxis([Smin,Smax]); 
            %caxis([0 70]); 
            %title(['RMS wavenumbers']);
            set(gcf,'color','white');
            %set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
            set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
            set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
            % remove unnecessary white space
            set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
            
            set(gcf,'PaperPositionMode','auto');
            drawnow;
            processed_filename = ['RMS_wavenumbers_selected_',filename]; % filename of processed .mat data
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
            switch k
                case 1           
                    caxis([200 1000]); % chirp
                case 2
                    caxis([300 500]); % 50 kHz
                case 3
                    caxis([350 650]); % 75 kHz
                case 4
                    caxis([400 700]); % 100 kHz
                case 5
                    caxis([550 850]); % 150 kHz
            end
            axis square;
            %caxis([caxis_cut*Smin,caxis_cut*Smax]);    
            %caxis([Smin,Smax]);  
            %caxis([0 500]); 
            %title(['Mean wavenumbers']);
            set(gcf,'color','white');
            %set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
            set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
            set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
            % remove unnecessary white space
            set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
            
            set(gcf,'PaperPositionMode','auto');
            drawnow;
            processed_filename = ['Mean_wavenumbers_selected_',filename]; % filename of processed .mat data
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
            switch k
                case 1           
                    caxis([200 1000]); % chirp
                case 2
                    caxis([300 500]); % 50 kHz
                case 3
                    caxis([350 650]); % 75 kHz
                case 4
                    caxis([400 700]); % 100 kHz
                case 5
                    caxis([550 850]); % 150 kHz
            end
            axis square;
            %caxis([caxis_cut*Smin,caxis_cut*Smax]);    
            %caxis([Smin,Smax]);  
            %caxis([0 500]); 
            %title(['Mean wavenumbers']);
            set(gcf,'color','white');
            %set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
            set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
            set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
            % remove unnecessary white space
            set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
            
            set(gcf,'PaperPositionMode','auto');
            drawnow;
            processed_filename = ['Mean_wavenumbers_selected_smooth',filename]; % filename of processed .mat data
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
            axis square;
            caxis([caxis_cut*Smin,caxis_cut*Smax]);    
            %caxis([0 7e-4]); 
            %title(['RMS amplitude']);
            set(gcf,'color','white');
            %set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
            set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
            set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
            % remove unnecessary white space
            set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
            
            set(gcf,'PaperPositionMode','auto');
            drawnow;
            processed_filename = ['RMS_amplitude_selected_',filename]; % filename of processed .mat data
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
            axis square;
            caxis([caxis_cut*Smin,caxis_cut*Smax]);    
            %caxis([0 4.5e-3]); 
            title(['Mean amplitude']);
            set(gcf,'color','white');
            %set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
            set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
            set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
            % remove unnecessary white space
            set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
            
            set(gcf,'PaperPositionMode','auto');
            drawnow;
            processed_filename = ['Mean_amplitude_selected_',filename]; % filename of processed .mat data
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
           
            axis square;
            %caxis([caxis_cut*Smin,caxis_cut*Smax]);  
            %caxis([Smin,Smax]); 
            %caxis([0 70]); 
            %title(['RMS wavenumbers']);
            set(gcf,'color','white');
            %set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
            set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
            set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
            % remove unnecessary white space
            set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
            
            set(gcf,'PaperPositionMode','auto');
            drawnow;
            processed_filename = ['RMS_wavenumbers_refined_',filename]; % filename of processed .mat data
            print([figure_output_path,processed_filename],'-dpng', '-r600'); 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Mean wavenumbers
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            figure;
            surf(x(:,1:end-radius_cut_wavenumbers-1),y(:,1:end-radius_cut_wavenumbers-1),squeeze(Mean_wavenumbers_refined (:,1:end-radius_cut_wavenumbers)));shading interp; view(2); colorbar; colormap(Cmap);       
            Smax=max(max(squeeze(Mean_wavenumbers_refined (:,1:end-radius_cut_wavenumbers))));Smin=mean(mean(squeeze(Mean_wavenumbers_refined (:,1:end-radius_cut_wavenumbers))));
            set(gcf,'Renderer','zbuffer');
            xlim([-0.25 0.25]);
            ylim([-0.25, 0.25]);
            switch k
                case 1           
                    caxis([200 1000]); % chirp
                case 2
                    caxis([300 500]); % 50 kHz
                case 3
                    caxis([350 650]); % 75 kHz
                case 4
                    caxis([400 700]); % 100 kHz
                case 5
                    caxis([550 850]); % 150 kHz
            end
            axis square;
            %caxis([caxis_cut*Smin,caxis_cut*Smax]);    
            %caxis([Smin,Smax]);  
            %caxis([0 500]); 
            %title(['Mean wavenumbers']);
            set(gcf,'color','white');
            %set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
            set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
            set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
            % remove unnecessary white space
            set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
            
            set(gcf,'PaperPositionMode','auto');
            drawnow;
            processed_filename = ['Mean_wavenumbers_refined_',filename]; % filename of processed .mat data
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
            switch k
                case 1           
                    caxis([200 1000]); % chirp
                case 2
                    caxis([300 500]); % 50 kHz
                case 3
                    caxis([350 650]); % 75 kHz
                case 4
                    caxis([400 700]); % 100 kHz
                case 5
                    caxis([550 850]); % 150 kHz
            end
            axis square;
            %caxis([caxis_cut*Smin,caxis_cut*Smax]);    
            %caxis([Smin,Smax]);  
            %caxis([0 500]); 
            %title(['Mean wavenumbers']);
            set(gcf,'color','white');
            %set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
            set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
            set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
            % remove unnecessary white space
            set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
            
            set(gcf,'PaperPositionMode','auto');
            drawnow;
            processed_filename = ['Mean_wavenumbers_refined_smooth_',filename]; % filename of processed .mat data
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
            axis square;
            caxis([caxis_cut*Smin,caxis_cut*Smax]);    
            %caxis([0 7e-4]); 
            %title(['RMS amplitude']);
            set(gcf,'color','white');
            %set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
            set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
            set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
            % remove unnecessary white space
            set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
            
            set(gcf,'PaperPositionMode','auto');
            drawnow;
            processed_filename = ['RMS_amplitude_refined_',filename]; % filename of processed .mat data
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
            axis square;
            caxis([caxis_cut*Smin,caxis_cut*Smax]);    
            %caxis([0 4.5e-3]); 
            %title(['Mean amplitude']);
            set(gcf,'color','white');
            %set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
            set(gca, 'OuterPosition',[0 0 1. 1.]); % figure without axis and white border
            set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
            % remove unnecessary white space
            set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
            
            set(gcf,'PaperPositionMode','auto');
            drawnow;
            processed_filename = ['Mean_amplitude_refined_',filename]; % filename of processed .mat data
            print([figure_output_path,processed_filename],'-dpng', '-r600'); 
            %close all;
                        % 
            %% END OF PROCESSING
            [filepath,name,ext] = fileparts(filename);
            fprintf('Successfully processed:\n%s\n', name);% successfully processed
        
    else
        fprintf('Filename: \n%s \nalready exist\n', processed_filename);
    end
end

toc
  
