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

%% Processing parameters

%%
% create path to the experimental raw data folder
input_data_path = '/pkudela_odroid_sensors/aidd/data/processed/exp/model_Abdalraheem/';

% files for processing
list1 = {'PSNR_exp_512_frames.csv'}; 
list2 = {'Pearson_CC_exp_512_frames.csv'};


disp('Interpolation and RMS calcualation');
folder  = input_data_path;
nFile   = length(list1);
success = false(1, nFile);
for k = 1:nFile
    filename1 = list1{k};
    filename2 = list2{k};
    processed_filename = ['frame_metrics_DLSR_model_',num2str(k)]; % filename of processed .mat data
    % check if already exist
    if(overwrite||(~overwrite && ~exist([figure_output_path,processed_filename,'.png'], 'file')))
        try 
            % load experimental data files
            disp('loading data');
            T1 = readtable([input_data_path,filename1]);
            PEARSON_metric = T1{2,:};
            parameter_frames = T1{1,:};
            T2 = readtable([input_data_path,filename2]);
            PSNR_metric = T2{2,:};
            
            %% PROCESS DATA
            fprintf('Processing:\n%s\n',filename1);
            figure;
            yyaxis left;
            plot(parameter_frames,PSNR_metric,'LineWidth',1);
            yyaxis right;
            plot(parameter_frames,PEARSON_metric,'LineWidth',1);
            run font_param;
            legend('PSNR','PEARSON CC','Location','northeast','Fontsize',legend_font_size,'FontName','Times');
            title('DLSR model I: 1024 points','Fontsize',title_font_size,'FontName','Times');
            xlabel({'$N_f$'},'Fontsize',label_font_size,'interpreter','latex');
            set(gcf,'color','white');
            run fig_param2;
            print([figure_output_path,processed_filename],'-dpng','-r600');
        
            %% END OF PROCESSING
            [filepath,name,ext] = fileparts(filename1);
            fprintf('Successfully processed:\n%s\n', name);% successfully processed
        catch
            fprintf('Failed: %s\n', filename1);
        end
    else
        fprintf('Filename: \n%s \nalready exist\n', processed_filename);
    end
end



