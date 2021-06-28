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
Nmed = 3;   % median filtering window size e.g. Nmed = 2 gives 2 by 2 points window size
m = 2.5;    % weight scale for wieghted RMS

trs = 0.95; % 
thrs = 20;  % if energy drops below x% stop processing ERMS
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
       
disp('Interpolation and RMS calcualation');
folder  = raw_data_path;
nFile   = length(test_case);
success = false(1, nFile);
for k = test_case
    filename = list{k};
    processed_filename = ['RMS_L3_S4_B_',filename]; % filename of processed .mat data
    % check if already exist
    if(overwrite||(~overwrite && ~exist([interim_data_path,processed_filename,'.png'], 'file')))
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
                 for frame = 1:nft
%                      Data(:,:,frame) = medfilt2(Data(:,:,frame),[Nmed Nmed],'symmetric');  
                       Data(:,:,frame) = mymedian3x3(Data(:,:,frame)); % 3x3 median filtering
                 end
             end
            
            %% Attenuation compensation
            E = zeros(nft,1);
            E2 = ones(nft,1);

            % Energy
            for frame = 1:nft
                E(frame) = sqrt(sum(sum(abs(Data(:,:,frame).^2))));
            end
            [maxx ~] = max(E);

            strt = 1;
            while strt < nft
                if E(strt) > trs*maxx 
                    break
                else
                 strt = strt + 1;
                end
            end

            E2(strt:end) = E(strt:end)./maxx;

            endd = strt;

            while endd < nft
                if E(endd) < thrs/100*maxx
                    break
                else
                 endd = endd + 1;
                end
            end
            E2(endd+1:end) = [];

            EData = zeros(nx,ny,endd-strt);
            for frame = strt:endd
                EData(:,:,frame-strt+1) = Data(:,:,frame)/E2(frame);
            end

            %% RMS
            % interpolated RMS
            RMS_small = abs(sqrt(sum(Data.^2,3)));
            RMS_interp = interp2(X,Y,RMS_small,XI,YI,'spline');

            % interpolated ERMS
            ERMS_small = abs(sqrt(sum(EData.^2,3)));
            ERMS_interp = interp2(X,Y,ERMS_small,XI,YI,'spline');

            % interpolated WRMS
            Weighted_Data_small = zeros([nx,ny,nft]);
            for frame=1:nft
                Weighted_Data_small(:,:,frame) = Data(:,:,frame)*sqrt(frame^m);
            end
            WRMS_small = abs(sqrt(sum(Weighted_Data_small(:,:,1:end).^2,3)));
            WRMS_interp = interp2(X,Y,WRMS_small,XI,YI,'spline');
            %% images
            %A = rms2image(RMS_interp, [interim_data_path,processed_filename]);
            B = rms2image(ERMS_interp, [interim_data_path,'E',processed_filename]);
            C = rms2image(WRMS_interp, [interim_data_path,'W',processed_filename]);
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



