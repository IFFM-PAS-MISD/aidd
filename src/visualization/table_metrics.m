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
paper_folder = 'Compressive_Sensing_paper';
paper_path=[projectroot,'reports',filesep,'journal_papers',filesep,paper_folder,filesep];
%% Processing parameters

% Nyquist experimental
L=500; % plate length [mm]
Wavelength = 19.5; % [mm]
dx=Wavelength/(2*sqrt(2));
xp=round(L/dx);
Nyquist_points = xp*xp;
% table filling
Method={'DLSR Model I', 'DLSR Model II', 'CS: jitter', 'CS: random', 'CS: jitter', 'CS: random', 'CS: jitter', 'CS: random'}';
% measurement points
Number_of_measurement_points = [1024,1024,1024,1024,3000,3000,4000,4000]';
% Compression ratio
CR_=Number_of_measurement_points/Nyquist_points*100; % [%]

Np=cellstr(num2str(Number_of_measurement_points));
CR=cellstr(num2str(CR_ ,'%.1f'));

%varNames = {'Method','$N_p$','CR [%]','PSNR','PEARSON CC','PSNR (delam)','PEARSON CC (delam)'};
varNames = {'Method','Np','CR','PSNR','PEARSON','PSNR1','PEARSON1'};
PSNR_plate_ = zeros(8,1)+20.456;
PSNR_plate = cellstr(num2str(PSNR_plate_,'%.1f'));
PEARSON_plate_ = zeros(8,1)+0.8973;
PEARSON_plate = cellstr(num2str(PEARSON_plate_,'%.2f'));
PSNR_delam_ = zeros(8,1)+20.456;
PSNR_delam = cellstr(num2str(PSNR_delam_,'%.1f'));
PEARSON_delam_ = zeros(8,1)+0.8973;
PEARSON_delam = cellstr(num2str(PEARSON_delam_,'%.2f'));
T = table(Method,Np,CR,PSNR_plate,PEARSON_plate,PSNR_delam,PEARSON_delam,'VariableNames',varNames);


writetable(T,[paper_path,'table_metrics.csv']);