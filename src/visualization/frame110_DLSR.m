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
figure_output_path = prepare_figure_paths(modelname);
%% figure parameters
fig_width =5; % figure widht in cm
fig_height=5; % figure height in cm
zoom_y = 300:400;
zoom_x = 256-50:256+50;
x_points = 128;
y_points = 128;
Lx=0.5; % plate length
Ly=0.5; % plate width
cmap = 'default';
caxis_cut = 0.6;
%% Processing parameters

%%
% create path to the experimental raw data folder
input_data_path = '/pkudela_odroid_sensors/aidd/data/processed/exp/model_Abdalraheem/';

% files for processing
list = {'SR_Pred_output_110_frame_UNIFORM_MESH_16th_pixel.png'}; 



disp('Interpolation and RMS calcualation');
folder  = input_data_path;
nFile   = length(list);
success = false(1, nFile);
for k = 1:nFile
    filename = list{k};
    processed_filename = ['frame110_DLSR_model_',num2str(k)]; % filename of processed .png data
    % check if already exist
    if(overwrite||(~overwrite && ~exist([figure_output_path,processed_filename,'.png'], 'file')))
        try 
            % load experimental data files
            disp('loading data');
            imdata = rgb2gray(imread([input_data_path,filename]));    
            %% PROCESS DATA
            fprintf('Processing:\n%s\n',filename);
            int_recon_image = im2double(imdata);
            int_recon_image = flipud(int_recon_image)-mean(mean(int_recon_image));
            % reconstructed
            Smax = max(max(int_recon_image));
            Smin = -Smax;
            figure;
            imagesc(int_recon_image);colormap(cmap);
            rectangle('Position',[206 300 101 101]);
            run fig_param4;
            caxis([caxis_cut*Smin,caxis_cut*Smax]);
            print([figure_output_path,'frame110_DLSR_model_',num2str(k),'.png'],'-dpng','-r600');

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



