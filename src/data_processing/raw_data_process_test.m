clc;close all;clear all;
% process raw data test script template

% load projectroot path
load project_paths;

% create path to the experimental raw data folder
raw_data_path=fullfile( projectroot, 'data','raw','exp', filesep );

% filename of data to be processed
filename = 'raw_data_testfile.txt';

% load raw experimental data file
raw_data_exp = load([raw_data_path,filename],'-ascii');

%% START DATA PROCESSING
%
interim_data_exp = 2*raw_data_exp;
%% END DATA PROCESSING

% create path to the experimental interim data folder
interim_data_path=fullfile( projectroot, 'data','interim','exp', filesep );

% filename of processed data
interim_filename = 'interim_data_testfile.txt';

% save processed data to interim (intermidiate) data folder
save([interim_data_path,interim_filename],'interim_data_exp','-ascii');
