clear all;close all;   warning off;clc;

load project_paths projectroot src_path;
%% Prepare output directories
% allow overwriting existing results if true
overwrite=false;
%overwrite=true;
test_case=[1:3]; % select file numbers for processing
%% Processing parameters
Nx = 500;   % number of points after interpolation in X direction
Ny = 500;   % number of points after interpolation in Y direction
N=Nx;
%%
% create path to the experimental raw data folder
%raw_data_path = fullfile( projectroot, 'data','raw','exp', filesep );
raw_data_path = '/pkudela_odroid_laser/CFRP_Jochen/';

% create path to the experimental interim data folder
interim_data_path = fullfile( projectroot, 'data','interim','exp', filesep );

% full field measurements
list = {'483x483p_50kHz_5HC_11pp_x10_1st_impact', ...  % 1  Length = ?;Width = ?;           
        '483x483p_100kHz_5HC_9Vpp_x10_1st_impact', ... % 2
        '483x483p_200kHz_5HC_4Vpp_x10_1st_impact'};    % 3 
% manual characterization of defects

xCenter=[396];
yCenter=[321];
a=[10/2];
b=[10/2];
rotAngle=[0,0,0];
for k=test_case
    label_data(k).xCenter=xCenter;
    label_data(k).yCenter=yCenter;
    label_data(k).a=a;
    label_data(k).b=b;
    label_data(k).rotAngle=rotAngle;
    label_data(k).type=["ellipse","ellipse","ellipse"];
end

folder  = raw_data_path;
nFile   = length(test_case);
success = false(1, nFile);

for k = test_case
    filename = list{k};
    processed_filename = ['label_CFRP_Jochen_',filename]; % filename of processed .mat data
    % check if already exist
    if(overwrite||(~overwrite && ~exist([interim_data_path,processed_filename,'.png'], 'file')))
        try        
            %% PROCESS DATA
            fprintf('Processing:\n%s\n',filename);
            multiple_delam_image_label(N,label_data(k).xCenter,label_data(k).yCenter,label_data(k).a,label_data(k).b,label_data(k).rotAngle,label_data(k).type,[interim_data_path,processed_filename]);
            
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



