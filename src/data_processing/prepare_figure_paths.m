function figure_output_path = prepare_figure_paths(foldername,modelname)
% PREPARE_FIGURE_PATHS   creates figure_output_path 
%     Also set current directory in folder related to the model
% Syntax: figure_output_path = prepare_figure_paths(foldername,modelname)
% 
% Inputs:   
%    foldername -  name of folder in which model resides, string
%    modelname -  name of model, string
% 
% Outputs:
%    figure_output_path - path to store the result, string
%
% Example: 
%    figure_output_path = prepare_figure_paths(foldername,modelname) 
%    figure_output_path = prepare_figure_paths(foldername) 
% 
% Other m-files required: none 
% Subfunctions:  
% MAT-files required: none 
% See also: CHECK_INPUTS_OUTPUTS
% 

% Author: Pawel Kudela, D.Sc., Ph.D., Eng. 
% Institute of Fluid Flow Machinery Polish Academy of Sciences 
% Mechanics of Intelligent Structures Department 
% email address: pk@imp.gda.pl 
% Website: https://www.imp.gda.pl/en/research-centres/o4/o4z1/people/ 

%---------------------- BEGIN CODE---------------------- 

% load projectroot path
load project_paths projectroot src_path;

if nargin == 1
    modelname = foldername;
    % set current path to figure folder
    figure_output_path = [projectroot,'reports',filesep,'figures',filesep,[modelname,'_out'],filesep];
end
if nargin == 2
    % set current path to figure folder
    figure_output_path = [projectroot,'reports',filesep,'figures',filesep,foldername,filesep,[modelname,'_out'],filesep];
end

% check if folder exist, if not create it
if ~exist(figure_output_path, 'dir')
    mkdir(figure_output_path);
end
end

%---------------------- END OF CODE---------------------- 

% ================ [prepare_model_paths.m] ================  