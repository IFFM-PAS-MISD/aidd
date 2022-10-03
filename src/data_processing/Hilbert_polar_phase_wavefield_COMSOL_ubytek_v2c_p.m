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
%%
% create path to the experimental raw data folder

raw_data_path = ['/pkudela_odroid_laser/COMSOL_ubytek/'];

% create path to the numerical interim data folder
interim_data_path = fullfile( projectroot, 'data','interim','num', filesep );

% create path to the numerical processed data folder
processed_data_path = fullfile( projectroot, 'data','processed','num', filesep );

% full field measurements
list = {'4_chirp','Data50','Data75','Data100','Data150'};


disp('Adaptive filtering calcualation');
folder  = raw_data_path;
nFile   = length(test_case);
success = false(1, nFile);
for k = test_case
    filename = list{k};
    processed_filename = ['ERMSF_',filename]; % filename of processed .mat data
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
            W = (-floor(N/2):ceil(N/2)-1)/N; % frequency coordinates
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
  
