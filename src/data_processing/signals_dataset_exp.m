clear all;close all;   warning off;clc;

load project_paths projectroot src_path;
%% Prepare output directories
% allow overwriting existing results if true
%overwrite=false;
overwrite=true;
test_case=[1:4]; % select file numbers for processing
% retrieve model name based on running file and folder
currentFile = mfilename('fullpath');
[pathstr,name,ext] = fileparts( currentFile );
idx = strfind( pathstr,filesep );
modelfolder = pathstr(idx(end)+1:end); % name of folder
modelname = name; 
%% Processing parameters
Nx = 500;   % number of points after interpolation in X direction
Ny = 500;   % number of points after interpolation in Y direction
Nmed = 3;   % median filtering window size e.g. Nmed = 2 gives 2 by 2 points window size
m = 2.5;    % weight scale for wieghted RMS

trs = 0.95; % 
thrs = 20;  % if energy drops below x% stop processing ERMS
%%
% create path to the experimental raw data folder
raw_data_path = fullfile( projectroot, 'data','raw','exp', filesep );

% full field measurements
list = {[raw_data_path,'CFRP_teflon_3o_375_375p_50kHz_5HC_x12_15Vpp'],...%1
         ['/pkudela_odroid_laser/aidd/data/raw/exp/L3_S2_B/333x333p_50kHz_5HC_15Vpp_x10_pzt'],...%2
         ['/pkudela_odroid_laser/aidd/data/raw/exp/L3_S3_B/333x333p_50kHz_5HC_18Vpp_x10_pzt'],...%3
         ['/pkudela_odroid_laser/aidd/data/raw/exp/L3_S4_B/333x333p_50kHz_5HC_18Vpp_x10_pzt']}; %4
                 
nft=512;
S=zeros(length(test_case),8,nft);
t_frames=zeros(length(test_case),nft);
% prepare output paths
dataset_output_path = prepare_data_processing_paths('processed','exp',modelname);


folder  = raw_data_path;
nFile   = length(test_case);
success = false(1, nFile);
c=0;
for k = test_case
    c=c+1;
    filename = list{k};
    processed_filename = filename; % filename of processed .mat data
              
    % check if already exist
    if(overwrite||(~overwrite && ~exist([dataset_output_path,modelname], 'file')))
        try 
            % load raw experimental data file
            disp('loading data');
            load([filename]); % Data, (time XI YI ZI)
            
            %% PROCESS DATA
            fprintf('Processing:\n%s\n',filename);
            % exclude points at the boundary
            Data=Data(2:end-2,2:end-2,:);
            [nx,ny,nf]=size(Data);
            [X,Y] = meshgrid(1:ny,1:nx);                                        % original value grid
            [XI,YI] = meshgrid(1:(ny-1)/(Ny-1):ny,1:(nx-1)/(Nx-1):nx);          % new value grid
            %% Median filtering
             if Nmed > 1      
                 for frame = 1:nft
%                      Data(:,:,frame) = medfilt2(Data(:,:,frame),[Nmed Nmed],'symmetric');  
                       Data(:,:,frame) = mymedian3x3(Data(:,:,frame)); % 3x3 median filtering
                 end
             end
            %% make interpolation of full wavefield     
            for frame = 1:nft
                Data_frame_interp = squeeze(interp2(X,Y,Data(:,:,frame),XI,YI,'spline'));
                % extract signals at 8 points of coordinates [pixels]
                        % S1(101,450)
                        % S2(400,450)
                        % S3(450,400)
                        % S4(450,450)
                        % S5(400,51)
                        % S6(101,51)
                        % S7(51,101)
                        % S8(51,400)
                        S(c,1,frame)=Data_frame_interp(101,450);
                        S(c,2,frame)=Data_frame_interp(400,450);
                        S(c,3,frame)=Data_frame_interp(450,450);
                        S(c,4,frame)=Data_frame_interp(450,450);
                        S(c,5,frame)=Data_frame_interp(400,51);
                        S(c,6,frame)=Data_frame_interp(101,51);
                        S(c,7,frame)=Data_frame_interp(51,101);
                        S(c,8,frame)=Data_frame_interp(51,400);   
                        t_frames(c,frame) = time(frame);
            end                  
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
save([dataset_output_path,filesep,modelname],'S');
save([dataset_output_path,filesep,'t_frames'],'t_frames');

