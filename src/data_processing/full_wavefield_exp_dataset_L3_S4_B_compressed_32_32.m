clear all;close all;   warning off;clc;

load project_paths projectroot src_path;
%% Prepare output directories
% allow overwriting existing results if true
overwrite=false;
overwrite=true;
test_case=[1]; % select file numbers for processing
%% Processing parameters
Nx = 32;   % number of points after interpolation in X direction
Ny = 32;   % number of points after interpolation in Y direction
Nmed = 3;   % median filtering window size e.g. Nmed = 2 gives 2 by 2 points window size

%%
% create path to the experimental raw data folder
%raw_data_path = fullfile( projectroot, 'data','raw','exp', filesep );
specimen_folder = 'L3_S4_B';
raw_data_path = ['/pkudela_odroid_laser/aidd/data/raw/exp/',specimen_folder,'/compressed/'];
%raw_data_path = ['\\odroid-laser\laser\aidd\data\raw\exp\',specimen_folder,'\compressed\'];
% create path to the experimental interim data folder
interim_data_path = fullfile( projectroot, 'data','interim','exp', filesep,specimen_folder,'compressed',filesep); % local
%interim_data_path = ['/pkudela_odroid_sensors/aidd/data/interim/exp/compressed/']; % NAS
%interim_data_path = ['\\odroid-sensors\sensors\aidd\data\interim\exp\compressed\']; % NAS
% check if folder exist, if not create it
if ~exist([interim_data_path], 'dir')
    mkdir([interim_data_path]);
end
% full field measurements
list = {'32x32p_50kHz_5HC_14Vpp_x10'  % 1             
        
        };
                 

disp('Interpolation and full wavefield to image calculation');
folder  = raw_data_path;
nFile   = length(test_case);
success = false(1, nFile);
for k = test_case
    filename = list{k};
    processed_filename = filename; % filename of processed .mat data
    % check if folder exist, if not create it
    if ~exist([interim_data_path,processed_filename], 'dir')
        mkdir([interim_data_path,processed_filename]);
    end
    frame_filename=[interim_data_path,processed_filename,filesep,'frame_1_',processed_filename];
              
    % check if already exist
    if(overwrite||(~overwrite && ~exist([frame_filename,'.png'], 'file')))
        try 
            % load raw experimental data file
            disp('loading data');
            load([raw_data_path,filename]); % Data, (time XI YI ZI)
            
            %% PROCESS DATA
            fprintf('Processing:\n%s\n',filename);
            % exclude points at the boundary
%             Data=Data(2:end-2,2:end-2,:);
             [nx,ny,nft]=size(Data);
%             [X,Y] = meshgrid(1:ny,1:nx);                                        % original value grid
%             [XI,YI] = meshgrid(1:(ny-1)/(Ny-1):ny,1:(nx-1)/(Nx-1):nx);          % new value grid
%             %% Median filtering
%              if Nmed > 1      
%                  for frame = 1:nft/2
% %                      Data(:,:,frame) = medfilt2(Data(:,:,frame),[Nmed Nmed],'symmetric');  
%                        Data(:,:,frame) = mymedian3x3(Data(:,:,frame)); % 3x3 median filtering
%                  end
%              end
%             %% make interpolation of full wavefield     
            for frame = 1:nft/2
                %Data_frame_interp = squeeze(interp2(X,Y,Data(:,:,frame),XI,YI,'spline'));
                % convert to image    
                frame_filename=[interim_data_path,processed_filename,filesep,'frame_',num2str(frame),'_',processed_filename];
                %F=frame2image(Data_frame_interp, frame_filename);  
                F=frame2image(Data(:,:,frame), frame_filename);  
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



