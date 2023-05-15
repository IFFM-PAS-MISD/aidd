%% script for computing root mean square (rms)
clear all;
% extract name of the running script
currentFile = mfilename('fullpath');
[pathstr,name,ext] = fileparts( currentFile );

% load data
filename = '50kHz_pzt';
specimen_name = 'specimen_1';
load(['../../data/raw/',specimen_name,'/',filename,'.mat']); % Data, time, L
%
%% process data - calculate root mean square value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Data_rms = sqrt(mean(Data.^2,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% check if directory exist; if not, create it
data_output_path = ['../../data/processed/',specimen_name,'/',name,'/'];
if ~exist(data_output_path, 'dir')
    mkdir(data_output_path);
end

% save processed data
out_filename = [filename, '_', name];
save([data_output_path,out_filename,'.mat'],'Data_rms','L');
