clear all;close all;   warning off;clc;

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

test_case=[1:5]; % select file numbers for processing

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
Nx = 500;   % number of points after interpolation in X direction
Ny = 500;   % number of points after interpolation in Y direction
Nmed = 3;   % median filtering window size e.g. Nmed = 2 gives 2 by 2 points window size
selected_frames=200:280; % selected frames for Hilbert transform
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
            disp('cartesian to polar conversion');
            [Data_polar,number_of_points,radius_max] = cartesian_to_polar_wavefield_2pi_variable_radius2(Data,WL(1),WL(2),beta);
            %    dimensions [number_of_angles,number_of_points,number_of_time_steps]
            
            %save([interim_data_path,'Data_polar2_',filename],'Data_polar','beta','radius_max','-v7.3');
%             disp('loading polar data');
            %load([interim_data_path,'Data_polar_',filename]);

            [number_of_angles,number_of_points,number_of_time_steps]=size(Data_polar);
            %% spatial signal at selected angle and time
            wavenumbers = zeros(number_of_angles,number_of_points-1,length(selected_frames));
            Amplitude = zeros(number_of_angles,number_of_points,length(selected_frames));
            
            polyn_order=20; % polynomial order for fitting unwrapped phase
            x=zeros(number_of_angles,number_of_points);
            y=zeros(number_of_angles,number_of_points);
            b = beta*pi/180;
            for ka=1:number_of_angles 
                R=linspace(0,radius_max(ka),number_of_points);
                x(ka,:) = R*cos(b(ka));
                y(ka,:) = R*sin(b(ka));
            end
            c=0;
            for frame = selected_frames
                [frame]
                c=c+1;
                unwrapped_phase_by_angle = zeros(number_of_angles,number_of_points);
                amplitude_by_angle = zeros(number_of_angles,number_of_points);
                
                for n_angle=1:number_of_angles
                    s = squeeze(Data_polar(n_angle,:,frame));
                    N = number_of_points; % number of samples

                    % Numerical approximation of the Hilbert transform in the FFT domain:
                    W = (-floor(N/2):ceil(N/2)-1)/N; % frequency coordinates
                    H = ifftshift(  -1i * sign(W)  ); % sampled Fourier response

                    % FFT-domain Hilbert transform of the input signal 's':
                    hilb = real(ifft(  fft(s) .* H  )); 


                    sa = s + 1i*hilb; % complex valued analytic signal associated to input signal
                    amp = abs(sa);    % instantaneous amplitude envelope
                    phz = angle(sa);  % instantaneous phase
                    amp_smoothed = movmean(amp,20); % moving mean for smoothing amplitude
                    unwrapped_phase_by_angle(n_angle,:) = unwrap(phz); % unwrapped phase
                    
                    amplitude_by_angle(n_angle,:) = amp_smoothed;
                end
                % smoothing over angle
                unwrapped_phase_by_angle_smoothed = movmean(unwrapped_phase_by_angle,3,1);
                amplitude_by_angle_smoothed = movmean(amplitude_by_angle,20,1);
                for n_angle=1:number_of_angles
                    Amplitude(n_angle,:,c) = amplitude_by_angle_smoothed(n_angle,:);
                    %Amplitude(n_angle,:,c) = amplitude_by_angle(n_angle,:);
                    unwrapped_phase = unwrapped_phase_by_angle_smoothed(n_angle,:);
                    %unwrapped_phase = unwrapped_phase_by_angle(n_angle,:);
                    xr=linspace(0,radius_max(n_angle),number_of_points); % variable radius for each angle
                    [p]=polyfit(xr,unwrapped_phase,polyn_order);
                    hm=polyval(p,xr); % fit polynomial
                    dx = xr(2:end) - xr(1:end-1);
                    hd = diff(hm)./dx; % first derivative 
                    %hd = diff(unwrapped_phase)./dx; % first derivative 
                    wavenumbers(n_angle,:,c)  = hd;
                end
                
            end
            
            figure('Position',[1 1 1920 1000])   
            radius_cut_wavenumbers = 40;
            radius_cut_amplitude = 0;
            Smax=max(max(max(squeeze(wavenumbers(:,1:end-radius_cut_wavenumbers+1,:)))));
            Smin=min(min(min(squeeze(wavenumbers(:,1:end-radius_cut_wavenumbers+1,:)))));
               
            c=0;
            for frame = selected_frames  
                c=c+1;
                subplot(1,2,1);
                surf(x(:,10:end-radius_cut_wavenumbers),y(:,10:end-radius_cut_wavenumbers),squeeze(wavenumbers(:,10:end-radius_cut_wavenumbers+1,c)));shading interp; view(2);  colorbar; colormap(Cmap);       
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
                        caxis([200 500]); % 50 kHz
                    case 3
                        caxis([200 500]); % 75 kHz
                    case 4
                        caxis([200 800]); % 100 kHz
                    case 5
                        caxis([200 500]); % 150 kHz
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
                caxis([0 7e-3]); 
                title(['Amplitude f= ',num2str(frame)]);
                pause(1);
            end
            % refine frame selection
            Kstd = zeros(length(selected_frames),1);
            c=0;
            for frame = selected_frames 
                c=c+1;
                Kstd(c) = std(wavenumbers(:,:,c),1,'all'); 
            end
            % select nt frames with lowest standard deviation
            nt=16;
            [B,I]=sort(Kstd,'ascend');
            RMS_wavenumbers = sqrt(sum(wavenumbers(:,:,I(1:nt)).^2,3))/length(selected_frames);
            Mean_wavenumbers = mean(wavenumbers(:,:,I(1:nt)),3);
            RMS_amplitude = sqrt(sum(Amplitude.^2,3))/length(selected_frames);
            Mean_amplitude = mean(Amplitude,3);
%             [TH,Radius] = meshgrid(beta*pi/180,R(1:end-1));
%             [Xk,Yk,Zk] = pol2cart(TH,Radius,Mean_wavenumbers');
%             figure;surf(Xk,Yk,Zk);shading interp;
            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% Figures
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            radius_cut_wavenumbers=15;
            figure;
            surf(x(:,1:end-radius_cut_wavenumbers),y(:,1:end-radius_cut_wavenumbers),squeeze(RMS_wavenumbers(:,1:end-radius_cut_wavenumbers+1)));shading interp; view(2);  colorbar; colormap(Cmap);       
            Smax=max(max(squeeze(RMS_wavenumbers(:,1:end-radius_cut_wavenumbers+1))));Smin=0;
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
            processed_filename = ['RMS_wavenumbers_',filename]; % filename of processed .mat data
            print([figure_output_path,processed_filename],'-dpng', '-r600'); 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            figure;
            surf(x(:,1:end-radius_cut_wavenumbers),y(:,1:end-radius_cut_wavenumbers),squeeze(Mean_wavenumbers (:,1:end-radius_cut_wavenumbers+1)));shading interp; view(2); colorbar; colormap(Cmap);       
            Smax=max(max(squeeze(Mean_wavenumbers (:,1:end-radius_cut_wavenumbers+1))));Smin=mean(mean(squeeze(Mean_wavenumbers (:,1:end-radius_cut_wavenumbers+1))));
            set(gcf,'Renderer','zbuffer');
            xlim([-0.25 0.25]);
            ylim([-0.25, 0.25]);
            axis square;
            %caxis([caxis_cut*Smin,caxis_cut*Smax]);    
            caxis([Smin,Smax]);  
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
            processed_filename = ['Mean_wavenumbers_',filename]; % filename of processed .mat data
            print([figure_output_path,processed_filename],'-dpng', '-r600'); 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            figure;
            surf(x(:,1:end-radius_cut_amplitude),y(:,1:end-radius_cut_amplitude),squeeze(RMS_amplitude(:,1:end-radius_cut_amplitude)));shading interp; view(2); colorbar; colormap(Cmap);       
            Smax=max(max(squeeze(RMS_amplitude(:,1:end-radius_cut_amplitude))));Smin=0;
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
            processed_filename = ['RMS_amplitude_',filename]; % filename of processed .mat data
            print([figure_output_path,processed_filename],'-dpng', '-r600'); 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            figure;
            surf(x(:,1:end-radius_cut_amplitude),y(:,1:end-radius_cut_amplitude),squeeze(Mean_amplitude (:,1:end-radius_cut_amplitude)));shading interp; view(2); colorbar; colormap(Cmap);       
            Smax=max(max(squeeze(Mean_amplitude (:,1:end-radius_cut_amplitude))));Smin=0;
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
            processed_filename = ['Mean_amplitude_',filename]; % filename of processed .mat data
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


  
