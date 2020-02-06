clear all;close all;   warning off;clc;

load project_paths projectroot src_path;
%% Prepare output directories
% allow overwriting existing results if true
%overwrite=false;
overwrite=true;
test_case=[1:12,15:24]; % select file numbers for processing
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

% create path to the experimental interim data folder
interim_data_path = fullfile( projectroot, 'data','interim','exp', filesep );

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
        'CFRP3_5_teflon15x15mm_50kHz_47085p_20Vppx20_10avg_prostokatne' %24
         };              


disp('Interpolation and RMS calcualation');
folder  = raw_data_path;
nFile   = length(test_case);
success = false(1, nFile);
for k = test_case
    filename = list{k};
    processed_filename = ['RMS_',filename]; % filename of processed .mat data
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



