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
frame_no=110;
% Nyquist experimental
L=500; % plate length [mm]
Wavelength = 19.5; % [mm]
dx=Wavelength/(2*sqrt(2));
xp=round(L/dx);
Nyquist_points = xp*xp;
%% load data
% create path to the experimental raw data folder
input_data_path = '/pkudela_odroid_sensors/aidd/data/processed/exp/model_Abdalraheem/';
T1 = readtable([input_data_path,'Pearson_CC_exp_512_frames.csv']);
DLSR_METHOD_I_PEARSON = T1{2,frame_no};
T2 = readtable([input_data_path,'PSNR_exp_512_frames.csv']);
DLSR_METHOD_I_PSNR = T2{2,frame_no};
T3 = readtable([input_data_path,'Pearson_CC_exp_512_frames_delamination_region.csv']);
DLSR_METHOD_I_PEARSON_delam = T3{2,frame_no};
T4 = readtable([input_data_path,'PSNR_exp_512_frames_delamination_region.csv']);
DLSR_METHOD_I_PSNR_delam = T4{2,frame_no};

foldername = 'compressive';
modelname = 'compressive_sensing_all_frames';
input_data_path3 = prepare_data_processing_paths('processed','exp',foldername,modelname);
load([input_data_path3,filesep,'frame_metrics_128x128_points_1024_jitter']);
CS_PEARSON_1024_jitter = PEARSON_metric(frame_no);
CS_PSNR_1024_jitter = PSNR_metric(frame_no);
CS_PEARSON_1024_jitter_delam = PEARSON_metric_delam(frame_no);
CS_PSNR_1024_jitter_delam = PSNR_metric_delam(frame_no);

load([input_data_path3,filesep,'frame_metrics_128x128_points_1024_random']);
CS_PEARSON_1024_random = PEARSON_metric(frame_no);
CS_PSNR_1024_random = PSNR_metric(frame_no);
CS_PEARSON_1024_random_delam = PEARSON_metric_delam(frame_no);
CS_PSNR_1024_random_delam = PSNR_metric_delam(frame_no);

load([input_data_path3,filesep,'frame_metrics_128x128_points_3000_jitter']);
CS_PEARSON_3000_jitter = PEARSON_metric(frame_no);
CS_PSNR_3000_jitter = PSNR_metric(frame_no);
CS_PEARSON_3000_jitter_delam = PEARSON_metric_delam(frame_no);
CS_PSNR_3000_jitter_delam = PSNR_metric_delam(frame_no);

load([input_data_path3,filesep,'frame_metrics_128x128_points_3000_random']);
CS_PEARSON_3000_random = PEARSON_metric(frame_no);
CS_PSNR_3000_random = PSNR_metric(frame_no);
CS_PEARSON_3000_random_delam = PEARSON_metric_delam(frame_no);
CS_PSNR_3000_random_delam = PSNR_metric_delam(frame_no);

load([input_data_path3,filesep,'frame_metrics_128x128_points_4000_jitter']);
CS_PEARSON_4000_jitter = PEARSON_metric(frame_no);
CS_PSNR_4000_jitter = PSNR_metric(frame_no);
CS_PEARSON_4000_jitter_delam = PEARSON_metric_delam(frame_no);
CS_PSNR_4000_jitter_delam = PSNR_metric_delam(frame_no);

load([input_data_path3,filesep,'frame_metrics_128x128_points_3000_random']);
CS_PEARSON_4000_random = PEARSON_metric(frame_no);
CS_PSNR_4000_random = PSNR_metric(frame_no);
CS_PEARSON_4000_random_delam = PEARSON_metric_delam(frame_no);
CS_PSNR_4000_random_delam = PSNR_metric_delam(frame_no);

%% create table
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
PSNR_plate_ = zeros(8,1);
PSNR_plate_(1,1) = DLSR_METHOD_I_PSNR;
PSNR_plate_(3,1) = CS_PSNR_1024_jitter;
PSNR_plate_(4,1) = CS_PSNR_1024_random;
PSNR_plate_(5,1) = CS_PSNR_3000_jitter;
PSNR_plate_(6,1) = CS_PSNR_3000_random;
PSNR_plate_(7,1) = CS_PSNR_4000_jitter;
PSNR_plate_(8,1) = CS_PSNR_4000_random;
PSNR_plate = cellstr(num2str(PSNR_plate_,'%.1f'));
PEARSON_plate_ = zeros(8,1);
PEARSON_plate_(1,1) = DLSR_METHOD_I_PEARSON;
PEARSON_plate_(3,1) = CS_PEARSON_1024_jitter;
PEARSON_plate_(4,1) = CS_PEARSON_1024_random;
PEARSON_plate_(5,1) = CS_PEARSON_3000_jitter;
PEARSON_plate_(6,1) = CS_PEARSON_3000_random;
PEARSON_plate_(7,1) = CS_PEARSON_4000_jitter;
PEARSON_plate_(8,1) = CS_PEARSON_4000_random;
PEARSON_plate = cellstr(num2str(PEARSON_plate_,'%.2f'));
PSNR_delam_ = zeros(8,1);
PSNR_delam_(1,1) = DLSR_METHOD_I_PSNR_delam;
PSNR_delam_(3,1) = CS_PSNR_1024_jitter_delam;
PSNR_delam_(4,1) = CS_PSNR_1024_random_delam;
PSNR_delam_(5,1) = CS_PSNR_3000_jitter_delam;
PSNR_delam_(6,1) = CS_PSNR_3000_random_delam;
PSNR_delam_(7,1) = CS_PSNR_4000_jitter_delam;
PSNR_delam_(8,1) = CS_PSNR_4000_random_delam;
PSNR_delam = cellstr(num2str(PSNR_delam_,'%.1f'));
PEARSON_delam_ = zeros(8,1);
PEARSON_delam_(1,1) = DLSR_METHOD_I_PEARSON_delam;
PEARSON_delam_(3,1) = CS_PEARSON_1024_jitter_delam;
PEARSON_delam_(4,1) = CS_PEARSON_1024_random_delam;
PEARSON_delam_(5,1) = CS_PEARSON_3000_jitter_delam;
PEARSON_delam_(6,1) = CS_PEARSON_3000_random_delam;
PEARSON_delam_(7,1) = CS_PEARSON_4000_jitter_delam;
PEARSON_delam_(8,1) = CS_PEARSON_4000_random_delam;

PEARSON_delam = cellstr(num2str(PEARSON_delam_,'%.2f'));
T = table(Method,Np,CR,PSNR_plate,PEARSON_plate,PSNR_delam,PEARSON_delam,'VariableNames',varNames);


writetable(T,[paper_path,'table_metrics.csv']);