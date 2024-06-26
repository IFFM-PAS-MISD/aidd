clear all;close all;   warning off;clc;

load project_paths projectroot src_path;
%% Prepare output directories
% allow overwriting existing results if true
overwrite=false;
%overwrite=true;
test_case=[1:12,15:27]; % select file numbers for processing
%% Processing parameters
Nx = 500;   % number of points after interpolation in X direction
Ny = 500;   % number of points after interpolation in Y direction
Nmed = 3;   % median filtering window size e.g. Nmed = 2 gives 2 by 2 points window size
m = 2.5;    % weight scale for wieghted RMS

trs = 0.95; % 
thrs = 20;  % if energy drops below x% stop processing ERMS
%%
% create path to the experimental raw data folder
raw_data_path = fullfile( projectroot, 'data','raw','exp', filesep );
specimen_folder = 'dataset2';
%raw_data_path = ['/pkudela_odroid_laser/aidd/data/raw/exp/',specimen_folder,'/'];
%raw_data_path = ['\\odroid-laser\laser\aidd\data\raw\exp\',specimen_folder,'\'];
% create path to the experimental interim data folder
%interim_data_path = fullfile( projectroot, 'data','interim','exp', filesep); % local
%interim_data_path = ['/pkudela_odroid_sensors/aidd/data/interim/exp/']; % NAS
interim_data_path = ['\\odroid-sensors\sensors\aidd\data\interim\exp/']; % NAS
% check if folder exist, if not create it
if ~exist([interim_data_path,specimen_folder], 'dir')
    mkdir([interim_data_path,specimen_folder]);
end
% full field measurements
list = {'GFRP_nr6_50kHz_5HC_8Vpp_x20_10avg_110889', ...          % 1  Length = ?;Width = ?;           
        'GFRP_nr6_100kHz_5HC_8Vpp_x20_10avg_110889', ... % 2
        'GFRP_nr_6_333x333p_5HC_150kHz_20vpp_x10', ... % 3
        'GFRP_nr6_200kHz_5HC_20vpp_x20xx17avg_110889', ... % 4
        'GFRP_nr1_333x333p_5HC_200kHz_20vpp_x20', ... % 5
        'GFRP_nr_1_333x333p_5HC_150kHz_20vpp_x10', ... % 6
        'GFRP_nr1_333x333p_5HC_100kHz_20vpp_x10', ... % 7
        'GFRP_nr4_50kHz_5HC_8Vpp_x20_10avg_110889', ... % 8
        'GFRP_nr4_100kHz_5HC_8Vpp_x20_10avg_110889', ... % 9
        'GFRP_nr4_150kHz_5HC_20Vpp_x20_5avg_110889', ... % 10
        'GFRP_nr4_200kHz_5HC_20Vpp_x20_12avg_110889', ... % 11
        'CFRP_teflon_impact_375x375p_5HC_100kHz__6Vpp_x10', ... %12
        'sf_30p_5mm_45mm_251x251p_50kHz_5HC_x3_15Vpp_norm', ... %13
        'Alu_2_54289p_16,5kHz_5T_x50_moneta2', ... %14
        'Alu_2_138383p_100kHz_5T_x50_moneta2', ... %15
        'Alu_2_77841p_35kHz_5T_x30_moneta', ... %16
        '93011_7uszk', ... %17
        '93011_2uszk', ... % 18
        'CFRPprepreg_41615p_teflon2cm_3mm_100kHz_20avg_15vpp_prostokatne', ... %19
        'CFRP_50kHz_10Vpp_x10_53261p_strona_oklejona_plastelina_naciecia_prostokatne_256_256', ... %20
        'CFRP_100kHz_20Vpp_x10_53261p_strona_oklejona_plastelina_naciecia_prostokatne_256_256', ... %21
        'CFRP3_5_teflon10x10mm_50kHz_47085p_20Vppx20_10avg_prostokatne', ... %22
        'CFRP3_5_teflon10x10mm_100kHz_47085p_20Vppx20_20avg_prostokatne', ... %23
        'CFRP3_5_teflon15x15mm_50kHz_47085p_20Vppx20_10avg_prostokatne',... %24           
         'CFRP_teflon_3_375_375p_50kHz_5HC_x3_15Vpp',...%25
         'CFRP_teflon_3c_375_375p_50kHz_5HC_x3_15Vpp',...%26
         'CFRP_teflon_3o_375_375p_50kHz_5HC_x12_15Vpp'};%27  
                 

disp('Interpolation and full wavefield to image calculation');
folder  = raw_data_path;
nFile   = length(test_case);
success = false(1, nFile);
for k = test_case
    filename = list{k};
    processed_filename = filename; % filename of processed .mat data
    % check if folder exist, if not create it
    if ~exist([interim_data_path,specimen_folder,filesep,processed_filename], 'dir')
        mkdir([interim_data_path,specimen_folder,filesep,processed_filename]);
    end
    frame_filename=[interim_data_path,specimen_folder,filesep,processed_filename,filesep,'frame_1_',processed_filename];
              
    % check if already exist
    if(overwrite||(~overwrite && ~exist([frame_filename,'.png'], 'file')))
        try 
            % load raw experimental data file
            disp('loading data');
            load([raw_data_path,filename]); % Data, (time XI YI ZI)
            
            %% PROCESS DATA
            fprintf('Processing:\n%s\n',filename);
            % exclude points at the boundary
            Data=Data(2:end-2,2:end-2,:);
            [nx,ny,nft]=size(Data);
            [X,Y] = meshgrid(1:ny,1:nx);                                        % original value grid
            [XI,YI] = meshgrid(1:(ny-1)/(Ny-1):ny,1:(nx-1)/(Nx-1):nx);          % new value grid
            %% Median filtering
             if Nmed > 1      
                 for frame = 1:nft/2
%                      Data(:,:,frame) = medfilt2(Data(:,:,frame),[Nmed Nmed],'symmetric');  
                       Data(:,:,frame) = mymedian3x3(Data(:,:,frame)); % 3x3 median filtering
                 end
             end
            %% make interpolation of full wavefield     
            for frame = 1:nft/2
                Data_frame_interp = squeeze(interp2(X,Y,Data(:,:,frame),XI,YI,'spline'));
                % convert to image    
                frame_filename=[interim_data_path,specimen_folder,filesep,processed_filename,filesep,'frame_',num2str(frame),'_',processed_filename];
                F=frame2image(Data_frame_interp, frame_filename);           
            end
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



