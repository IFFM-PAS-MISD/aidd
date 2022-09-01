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

test_case=[1:6]; % select file numbers for processing

%% input for figures
Cmap = jet(256); 
caxis_cut = 0.8;
fig_width =5; % figure widht in cm
fig_height=5; % figure height in cm
%% Input for signal processing

WL = [0.5 0.5];
%mask_thr = 5;% percentage of points removed by filter mask, should be in range 0.5 - 5 
mask_thr = 2;
PLT = 0.5;% if PLT = 0 do no create plots; 0<PLT<=0.5 only ERMSF plot ;0.5<PLT<=1 - RMS plots;  1<PLT - all plots

threshold = 0.0018; % threshold for binarization
%% Processing parameters
Nx = 500;   % number of points after interpolation in X direction
Ny = 500;   % number of points after interpolation in Y direction
Nmed = 3;   % median filtering window size e.g. Nmed = 2 gives 2 by 2 points window size
%%
% create path to the experimental raw data folder
specimen_folder = 'L3_S3_B';

raw_data_path = ['/pkudela_odroid_laser/aidd/data/raw/exp/',specimen_folder,'/'];

% create path to the experimental interim data folder
interim_data_path = fullfile( projectroot, 'data','interim','exp', filesep );

% create path to the experimental processed data folder
processed_data_path = fullfile( projectroot, 'data','processed','exp', filesep );

% full field measurements
list = {'333x333p_16_5kHz_5HC_18Vpp_x10_pzt', ...          % 1  Length = ?;Width = ?;           
        '333x333p_50kHz_5HC_18Vpp_x10_pzt', ... % 2
        '333x333p_75kHz_10HC_18Vpp_x10_pzt_no_filter', ... % 3
        '333x333p_75kHz_10HC_18Vpp_x10_pzt_no_filter_v2', ... % 4
        '333x333p_100kHz_5HC_15Vpp_x10_pzt', ... % 5
        '333x333p_100kHz_10HC_15Vpp_x20_pzt_no_filter'};% 6


disp('Adaptive filtering calcualation');
folder  = raw_data_path;
nFile   = length(test_case);
success = false(1, nFile);
for k = test_case
    filename = list{k};
    processed_filename = ['ERMSF_',filename]; % filename of processed .mat data
    % check if already exist
    if(overwrite||(~overwrite && ~exist([figure_output_path,processed_filename,'.png'], 'file')))
        try 
            % load raw experimental data file
            disp('loading data');
            load([raw_data_path,filename]); % Data, (time XI YI ZI)
            
            %% PROCESS DATA
            fprintf('Processing:\n%s\n',filename);
            % exclude points at the boundary
            Data=Data(2:end-2,2:end-2,:);
            [nx,ny,nft]=size(Data);
            [X,Y] = meshgrid(1:ny,1:nx);                                        % original value grid
            [XI,YI] = meshgrid(1:(ny-1)/(Ny-1):ny,1:(nx-1)/(Nx-1):nx);          % new value grid
            %% Median filtering
             if Nmed > 1      
                 for frame = 1:nft
%                      Data(:,:,frame) = medfilt2(Data(:,:,frame),[Nmed Nmed],'symmetric');  
                       Data(:,:,frame) = mymedian3x3(Data(:,:,frame)); % 3x3 median filtering
                 end
             end
             %% Adaptive filtering
                        [RMSF,ERMSF,WRMSF] = AdaptiveFiltering(Data,time,WL,mask_thr,PLT);
                        %% interpolate on Nx x Ny grid
                        ERMSF_interp = interp2(X,Y,ERMSF,XI,YI,'spline');
                       %% save picture
                       figure;
                       imagesc(ERMSF_interp);
                       colormap(Cmap);
                       set(gca,'YDir','normal');axis square;axis off;
                       set(gcf,'color','white');
                       
                       Smin=0;
                       Smax=max(max(ERMSF_interp));
                       set(gcf,'Renderer','zbuffer');
                       caxis([caxis_cut*Smin,caxis_cut*Smax]);
                        set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
                        set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
                        % remove unnecessary white space
                        %set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
                        set(gcf,'PaperPositionMode','auto');

                        print([figure_output_path,processed_filename],'-dpng', '-r600'); 
                       %% binarization                
                        %Binary = uint8(ERMSF_interp >= threshold);
                        Binary = (ERMSF_interp >= 0.5*max(max(ERMSF_interp)));
                        %save(filename,'ERMSF_interp','WL','mask_thr','Binary','threshold');
                        % plot
%                         whitethreshold = .05;
%                         blackthreshold = .05;
%                         CmapB = 1-([blackthreshold:1/255:1-whitethreshold ; blackthreshold:1/255:1-whitethreshold ; blackthreshold:1/255:1-whitethreshold]');
                         figure
                         imagesc(flipud(Binary));
                         CMap=[0,0,0; 1,1,1];
                         colormap(CMap);
%                         colormap(1-CmapB)
%                         set(gca,'YDir','normal');
                        axis square;axis off;
                        set(gcf,'color','white');
                        set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
                        set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
                        print([figure_output_path,'Binary_',processed_filename],'-dpng', '-r600'); 
                        %imwrite(flipud(Binary),[figure_output_path,'Binary_',processed_filename,'.png'],'png');         
                        %% intersection over union
                        labelname=[image_label_path,'label_',specimen_folder,'_',filename,'.png'];
                        A=imread(labelname)/255;
                        IoU=intersect_over_union_fun(flipud(Binary),logical(A));
                        area=sum(sum(Binary))/(Nx*Ny)*WL(1)*1e3*WL(2)*1e3; % [mm^2]
                        disp('Intersection over union: ');IoU
                        close all;
                        % 
            %% END OF PROCESSING
            [filepath,name,ext] = fileparts(filename);
            fprintf('Successfully processed:\n%s\n', name);% successfully processed
        catch
            fprintf('Failed: %s\n', filename);
        end
    else
        fprintf('Filename: \n%s \nalready exist\n', processed_filename);
    end
end



