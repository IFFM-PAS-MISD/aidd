clear all;close all;   warning off;clc;

load project_paths projectroot src_path;
%% Prepare output directories
% allow overwriting existing results if true
overwrite=false;
overwrite=true;
test_case=[1]; % select file numbers for processing
%% Processing parameters
Nx = 512;   % number of points after interpolation in X direction
Ny = 512;   % number of points after interpolation in Y direction
Nmed = 3;   % median filtering window size e.g. Nmed = 2 gives 2 by 2 points window size
nft = 512;
%%
% create path to the experimental raw data folder
%raw_data_path = fullfile( projectroot, 'data','raw','exp', filesep );
specimen_folder = 'L3_S4_B';
raw_data_path = ['/home/pkudela/work/projects/opus16/aidd/src/data_processing/compressive/'];
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
list = {'389286p'  % 1             
        
        };
                 

disp('Interpolation and full wavefield to image calculation');
folder  = raw_data_path;
nFile   = length(test_case);
success = false(1, nFile);
for k = test_case
    filename = list{k};
    processed_filename = [filename,'_',num2str(Nx),'x',num2str(Ny)]; % filename of processed .mat data
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
             
%             %% make interpolation of full wavefield     
            for frame = 1:nft
                Data_frame_interp = regInterp(Data,XYZ,Nx,Ny,frame);
                %Data_frame_interp = squeeze(interp2(X,Y,Data(:,:,frame),XI,YI,'spline'));
                % convert to image    
                frame_filename=[interim_data_path,processed_filename,filesep,'frame_',num2str(frame),'_',processed_filename];  
                F=frame2image(Data_frame_interp, frame_filename);  
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

