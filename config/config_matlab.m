% configuration file for Matlab paths

% retrieve path of currently running file
currentFile=mfilename('fullpath');
[pathstr,~,~]=fileparts( currentFile );
config_path=fullfile( pathstr );
% extract projectroot path
projectroot=fullfile(pathstr(1:end-6));
% create src_path based on standard folder structure
src_path = fullfile( projectroot, 'src', filesep);
% add src_path to the matlab path with subfolders
addpath(genpath(src_path));
disp(['projectroot path: ',projectroot]);
disp(['added src_path with subfolders: ', src_path]);
filename = fullfile(src_path,'project_paths.mat');
% save path definitions to mat file at src_path for future use
save(filename,'projectroot','src_path');

