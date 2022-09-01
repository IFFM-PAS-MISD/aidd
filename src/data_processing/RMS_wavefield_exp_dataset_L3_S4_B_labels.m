clear all;close all;   warning off;clc;

load project_paths projectroot src_path;
%% Prepare output directories
% allow overwriting existing results if true
overwrite=false;
%overwrite=true;
test_case=[1:8]; % select file numbers for processing
%% Processing parameters
Nx = 500;   % number of points after interpolation in X direction
Ny = 500;   % number of points after interpolation in Y direction
N=Nx;
%%
% create path to the experimental raw data folder
%raw_data_path = fullfile( projectroot, 'data','raw','exp', filesep );
raw_data_path = '/pkudela_odroid_laser/aidd/data/raw/exp/L3_S4_B/';

% create path to the experimental interim data folder
interim_data_path = fullfile( projectroot, 'data','interim','exp', filesep );

% full field measurements
list = {'333x333p_16_5kHz_5HC_18Vpp_x10_pzt', ...    % 1             
        '333x333p_16_5kHz_10HC_18Vpp_x10_pzt', ... % 2
        '333x333p_50kHz_5HC_18Vpp_x10_pzt', ... % 3
        '333x333p_50kHz_10HC_18Vpp_x10_pzt', ... % 4
        '333x333p_75kHz_10HC_18Vpp_x10_pzt', ... % 5
        '333x333p_75kHz_10HC_18Vpp_x10_pzt_no_filter',...%6
        '333x333p_100kHz_5HC_14Vpp_x10_pzt',...%7
        '333x333p_100kHz_10HC_14Vpp_x20_pzt',...%8
        };
% manual characterization of defects

% xCenter=[250,100,356];
% yCenter=[400,250,144];
% a=[20/2,20/2,20/2];
% b=[10/2,10/2,10/2];
% correction because of shifted teflon inserts
xCenter=[250,100-3,356+2];
yCenter=[400+8,250+6,144];
a=[20/2,20/2,20/2];
b=[10/2,10/2,10/2];
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
    processed_filename = ['label_L3_S4_B_',filename]; % filename of processed .mat data
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



