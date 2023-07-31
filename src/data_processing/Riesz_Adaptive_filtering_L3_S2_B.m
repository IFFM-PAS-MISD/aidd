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
dataset_output_path = prepare_data_processing_paths('processed','exp',modelname);
figure_output_path = prepare_figure_paths(modelname);
%image_label_path=fullfile(projectroot,'data','interim','exp',filesep);
image_label_path='/pkudela_odroid_sensors/aidd/data/interim/exp/new_exp/';

test_case=[1,3:8]; % select file numbers for processing

%% input for figures
Cmap = jet(256); 
caxis_cut = 0.8;
fig_width =6; % figure widht in cm
fig_height=6; % figure height in cm
%% Input for signal processing

WL = [0.5 0.5];
%mask_thr = 5;% percentage of points removed by filter mask, should be in range 0.5 - 5 
%mask_thr = 1;
%mask_thr = 0.5;
mask_thr = 2;
%PLT = 0.5;% if PLT = 0 do no create plots; 0<PLT<=0.5 only ERMSF plot ;0.5<PLT<=1 - RMS plots;  1<PLT - all plots
PLT = 0.6;
%threshold = 0.0018; % threshold for binarization
%% Processing parameters
Nx = 500;   % number of points after interpolation in X direction
Ny = 500;   % number of points after interpolation in Y direction
Nmed = 3;   % median filtering window size e.g. Nmed = 2 gives 2 by 2 points window size
selected_frames=50:200; % selected frames for Riesz transform
% create path to the experimental raw data folder
specimen_folder = 'L3_S2_B';

raw_data_path = ['/pkudela_odroid_laser/aidd/data/raw/exp/',specimen_folder,'/'];

% create path to the experimental interim data folder
interim_data_path = fullfile( projectroot, 'data','interim','exp', filesep );

% create path to the experimental processed data folder
processed_data_path = fullfile( projectroot, 'data','processed','exp', filesep );

% full field measurements
list = {'333x333p_16_5kHz_5HC_18Vpp_x10_pzt', ...          % 1  Length = ?;Width = ?;           
        '333x333p_50kHz_5HC_15Vpp_x10_pzt', ... % 2
        '333x333p_100kHz_5HC_10Vpp_x10_pzt', ... % 3
        '333x333p_100kHz_10HC_10Vpp_x10_pzt', ... % 4
        '333x333p_100kHz_20HC_10Vpp_x10_pzt', ... % 5
        '333x333p_150kHz_10HC_10Vpp_x20_pzt', ... % 6
        '333x333p_200kHz_10HC_10Vpp_x20_pzt', ... % 7
        '497x497p_100kHz_10HC_10Vpp_x25_pzt'};% 8



disp('Adaptive filtering calcualation');
folder  = raw_data_path;
nFile   = length(test_case);
success = false(1, nFile);
for k = test_case
    filename = list{k};
    processed_filename = ['ERMSF_',filename]; % filename of processed .mat data
    % check if already exist
    if(overwrite||(~overwrite && ~exist([figure_output_path,processed_filename,'.png'], 'file')))
       
            % load raw experimental data file
            disp('loading data');
            load([raw_data_path,filename]); % Data, (time XI YI ZI)
            
            %% PROCESS DATA
            fprintf('Processing:\n%s\n',filename);
            % exclude points at the boundary
            %Data=Data(2:end-2,2:end-2,:);
            [nx,ny,nft]=size(Data);
%             Data2=Data(200:299,45:144,:);
%             Data=Data2;
%             [nx,ny,nft]=size(Data);
            %% Median filtering
            if Nmed > 1      
                 for frame = 1:nft
%                      Data(:,:,frame) = medfilt2(Data(:,:,frame),[Nmed Nmed],'symmetric');  
                       Data(:,:,frame) = mymedian3x3(Data(:,:,frame)); % 3x3 median filtering
                 end
            end
            %% Adaptive filtering
            [FilterMask,RMSF,ERMSF,WRMSF] = AdaptiveFilteringMask(Data,time,WL,mask_thr,PLT);
            ERMSF = ERMSF(2:end-2,2:end-2,:);
            ReverseFilterMask = -1*(FilterMask - 1);

            M=nx;
            N=ny;
            
           
            Fs =  1/(time(3)-time(2));                % sampling frequency
            f_vec = Fs*linspace(0,1,nft);         % frequency vector
%             if(k == 3) % 75 kHz
%                 % 1D FFT
%                 Dataf = fft(Data,nft,3);
%                 % extract 75 kHz
%                 f_selected = 75000;
%                 [~,I] = min(abs(f_vec-f_selected));
%                 % 3D mask
%                 mask3D = zeros(nx,ny,nft);
%                 mask3D(:,:,I) = 1;
%                 mask3D(:,:,nft-I+2) = 1;
%                 Dataf=Dataf.*mask3D;
%                 Datanew = ifft(Dataf,nft,3);
%                 Data=Datanew;
%             end
%             
            nsX = 1024;
            
            Length = WL(1)*1e3; % [mm]
            dx = Length/(nx-1);
            dkx = 1/(nx*dx);
            kxmax = 1/(2*dx)-dkx/2;
            %kx_vec = 2*pi*linspace(0,kxmax,nx);    % rad/m
            kx_vec = 2*pi*linspace(-kxmax/2,kxmax/2,nx);    % rad/m
            % Frequency coordinates:
%             [W1,W2] = ndgrid(  (-floor(M/2):ceil(M/2)-1)/M  ,...
%                                (-floor(N/2):ceil(N/2)-1)/N  );
             [W1,W2] = ndgrid(  kx_vec  ,...
                                kx_vec  );  % rad/mm             
%              Wx=-W1./(2*pi*(sqrt(W1.^2+W2.^2)).^3);
%              Wy=-W2./(2*pi*(sqrt(W1.^2+W2.^2)).^3);
            
            % ----------------------------------------------------------------- %
            % Perform Riesz transform on the input image.
            % ----------------------------------------------------------------- %
            
            % Numerical approximation of the Riesz transform in the FFT domain:
            %RZ = ifftshift(  -1i * exp(  1i*atan2(W2,W1)  )    ); 
            RZ = (  -1i * exp(  1i*atan2(W2,W1)  )    ); 
            % Riesz operator x direction
            %figure;surf(W2,W1,real(fftshift(RZ)));shading interp;  axis square;xlabel('kx');ylabel('ky');
            % Riesz operator y direction
            %figure;surf(W2,W1,imag(fftshift(RZ)));shading interp;  axis square;xlabel('kx');ylabel('ky');
            A=zeros(nx,ny);
            A1=ones(nx,ny);
            A_all = zeros(nx,ny,length(selected_frames));
            phz_all = zeros(nx,ny,length(selected_frames));
            figure(10);
            c=0;
            for frame=selected_frames
                c=c+1;
                [frame]
            
            %SP = fft2(squeeze(Data(:,:,frame))); % input image FFT
            SP = fftshift(fft2(squeeze(Data(:,:,frame)))); % input image FFT
            %prim = real(ifft2( SP .* mask )); % bandpass filtering
            %prim = real(ifft2( SP )); %
            prim = Data(:,:,frame);
            %riz = ifft2( SP .* RZ ); % Riesz transform
            riz = ifft2(ifftshift( SP .* RZ )); % Riesz transform

            riz1 = real(riz);  % Riesz x-component
            riz2 = imag(riz);  % Riesz y-component
            rizN = abs(riz);   % Riesz norm
            rizA = angle(riz); % Riesz direction

            amp = abs(prim + 1i*rizN); % mistake here; it should be without complex ? It is ok - error in the text
            %amp2 = sqrt(prim.^2+riz1.^2+riz2.^2); % another definition for amplitude (the same results)
            
            %phz = angle(prim + 1i*rizN);
            phz = angle(prim + rizN);
            phz2 = -sign(riz1).*atan2(rizN,prim);

            % rescale to [-pi:pi];
            a0 = -pi;
            b0 = pi;
            phz_scaled = a0 + ((b0-a0)/(max(max(phz))-min(min(phz))))*phz;
            
            
%             ired = (phz_scaled>0.7*2*pi);
%             iblu = (phz_scaled<0.3*2*pi);
%             phz_scaled2=phz_scaled;
%             phz_scaled2(ired)=2*pi;
%             phz_scaled2(iblu)=0;
            % phase unwrap
            %res_img = unwrap_phase(phz_scaled);
            res_img = unwrap_phase(2*phz-pi);
            res_img2 = unwrap_phase(phz2);
            % ----------------------------------------------------------------- %
            % Generate illustrations of the result.
            % ----------------------------------------------------------------- %
            % second order derivative
%             xd = diff([x(3),x,x(n-2)]);  % <-- Corrected
%             ud = diff([u(3),u,u(n-2)]);  % <-- Corrected
%             dudx = (ud(1:end-1)./xd(1:end-1).*xd(2:end) ...
%             + ud(2:end)./xd(2:end).*xd(1:end-1)) ...
%             ./ (xd(2:end)+xd(1:end-1));
            % first order derivative
            %[px,py] = gradient(phzwrap);
%             [px,py] = gradient(res_img2);
%             figure;
%             surf(sqrt(px.^2+py.^2));shading interp;axis square;view(2);
%             figure;
%             imshow(  1-amp/max(amp(:))  );
               surf(amp);shading interp;  axis square;view(2);drawnow;
               
               A=A+amp.^2;
               A1 = A1.*amp;
               A_all(:,:,c) = amp;
               phz_all(:,:,c) = phz2;
            end
            figure;surf(A);shading interp;  axis square;view(2);drawnow;
%             A_all_diff = diff(A_all,1,3); % first derivative
%             A_all_diff2 = diff(A_all,2,3); % second derivative
            %phz_all_diff = diff(phz_all,1,3); % first derivative
            figure;surf(squeeze(mean(A_all,3)));shading interp;view(2);axis square
            figure;surf(squeeze(median(A_all,3)));shading interp;view(2);axis square
%             [px,py] = gradient(squeeze(mean(A_all,3)));
%             [px,py] = gradient(squeeze(A));
%             figure;
%             surf(sqrt(px.^2+py.^2));shading interp;axis square;view(2);
%[FilterMask2,RMSF2,ERMSF2,WRMSF2] = AdaptiveReverseFilteringMask(A_all,time(selected_frames),WL,mask_thr,PLT);
           
 
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Adaptive filtering mask + Riesz
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  Aadapt=zeros(nx,ny);
            figure(12);
            for frame=selected_frames
                [frame]
            
            %SP = fft2(squeeze(Data(:,:,frame))); % input image FFT
            SP = fftshift(fft2(squeeze(Data(:,:,frame)))); % input image FFT
            prim = real(ifft2( ifftshift(SP .* FilterMask) )); % bandpass filtering
            %riz = ifft2( SP .* FilterMask.* RZ ); % Riesz transform
            %riz = ifft2( SP .* RZ ); % Riesz transform
            %riz = ifft2( ifftshift(SP .* RZ) ); % Riesz transform
            riz = ifft2( ifftshift(SP .* FilterMask.* RZ) ); % Riesz transform
            
            riz1 = real(riz);  % Riesz x-component
            riz2 = imag(riz);  % Riesz y-component
            rizN = abs(riz);   % Riesz norm
            rizA = angle(riz); % Riesz direction

            amp = abs(prim + 1i*rizN); % 
            amp2 = sqrt(prim.^2+riz1.^2+riz2.^2);
           
            phz3 = angle(prim + 1i*rizN);
            phz4 = -sign(riz1).*atan2(rizN,prim);
%             phz = atan2(sqrt(riz1.^2+riz2.^2),prim);
%             phz = atan2(prim,sqrt(riz1.^2+riz2.^2));
            % rescale to [-pi:pi];
            a0 = -pi;
            b0 = pi;
            phz_scaled = a0 + ((b0-a0)/(max(max(phz))-min(min(phz))))*phz;
            
            
%             ired = (phz_scaled>0.7*2*pi);
%             iblu = (phz_scaled<0.3*2*pi);
%             phz_scaled2=phz_scaled;
%             phz_scaled2(ired)=2*pi;
%             phz_scaled2(iblu)=0;
            % phase unwrap
            %res_img = unwrap_phase(phz_scaled);
            res_img3 = unwrap_phase(2*phz3-pi);
            res_img4 = unwrap_phase(phz4);
            % ----------------------------------------------------------------- %
            % Generate illustrations of the result.
            % ----------------------------------------------------------------- %
            % second order derivative
%             xd = diff([x(3),x,x(n-2)]);  % <-- Corrected
%             ud = diff([u(3),u,u(n-2)]);  % <-- Corrected
%             dudx = (ud(1:end-1)./xd(1:end-1).*xd(2:end) ...
%             + ud(2:end)./xd(2:end).*xd(1:end-1)) ...
%             ./ (xd(2:end)+xd(1:end-1));
            % first order derivative
            %[px,py] = gradient(phzwrap);
%             [px,py] = gradient(res_img);
%             figure;
%             surf(sqrt(px.^2+py.^2));shading interp;axis square;view(2);
%             figure;
%             imshow(  1-amp/max(amp(:))  );
               surf(amp);shading interp;  axis square;view(2);drawnow;
               
               Aadapt=Aadapt+amp.^2;
            end
            Aadapt=Aadapt(2:end-2,2:end-2,:);
            figure;surf(Aadapt);shading interp;  axis square;view(2);drawnow;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Adaptive Reverse filtering mask + Riesz
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  Aadapt2=zeros(nx,ny);
            figure(14);
            for frame=selected_frames
                [frame]
            SP = fftshift(fft2(squeeze(Data(:,:,frame)))); % input image FFT
            prim = real(ifft2( ifftshift(SP .* ReverseFilterMask) )); % bandpass filtering
            %riz = ifft2( ifftshift(SP .* RZ) ); % Riesz transform
            riz = ifft2( ifftshift(SP .* FilterMask.* RZ) ); % Riesz transform
            
            riz1 = real(riz);  % Riesz x-component
            riz2 = imag(riz);  % Riesz y-component
            rizN = abs(riz);   % Riesz norm
            rizA = angle(riz); % Riesz direction

            amp = abs(prim + 1i*rizN); % 
            amp2 = sqrt(prim.^2+riz1.^2+riz2.^2);
           
            phz5 = angle(prim + 1i*rizN);
            phz6 = -sign(riz1).*atan2(rizN,prim);

            % rescale to [-pi:pi];
            a0 = -pi;
            b0 = pi;
            phz_scaled = a0 + ((b0-a0)/(max(max(phz))-min(min(phz))))*phz;
            
            
%             ired = (phz_scaled>0.7*2*pi);
%             iblu = (phz_scaled<0.3*2*pi);
%             phz_scaled2=phz_scaled;
%             phz_scaled2(ired)=2*pi;
%             phz_scaled2(iblu)=0;
            % phase unwrap
            %res_img = unwrap_phase(phz_scaled);
            res_img5 = unwrap_phase(2*phz5-pi);
            res_img6 = unwrap_phase(phz6);
            % ----------------------------------------------------------------- %
            % Generate illustrations of the result.
            % ----------------------------------------------------------------- %
            % second order derivative
%             xd = diff([x(3),x,x(n-2)]);  % <-- Corrected
%             ud = diff([u(3),u,u(n-2)]);  % <-- Corrected
%             dudx = (ud(1:end-1)./xd(1:end-1).*xd(2:end) ...
%             + ud(2:end)./xd(2:end).*xd(1:end-1)) ...
%             ./ (xd(2:end)+xd(1:end-1));
            % first order derivative
            %[px,py] = gradient(phzwrap);
%             [px,py] = gradient(res_img);
%             figure;
%             surf(sqrt(px.^2+py.^2));shading interp;axis square;view(2);
%             figure;
%             imshow(  1-amp/max(amp(:))  );
               surf(amp);shading interp;  axis square;view(2);drawnow;
               
               Aadapt2=Aadapt2+amp.^2;
            end
            figure;surf(Aadapt2);shading interp;  axis square;view(2);drawnow;
            pause(1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
                       %% save pictures
                       processed_filename = ['ERMSF_',filename]; % filename of processed .mat data
                       figure;
                       imagesc(ERMSF);
                       colormap(Cmap);
                       set(gca,'YDir','normal');
                       axis square;axis off;
                       set(gcf,'color','white');
                       
                       Smin=0;
                       Smax=max(max(ERMSF));
                       set(gcf,'Renderer','zbuffer');
                       caxis([caxis_cut*Smin,caxis_cut*Smax]);
                        set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
                        set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
                        % remove unnecessary white space
                        %set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
                        set(gcf,'PaperPositionMode','auto');
                        drawnow;
                        print([figure_output_path,processed_filename],'-dpng', '-r600'); 
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                       %% binarization                
                        %Binary = uint8(ERMSF_interp >= threshold);
                        Binary = (ERMSF >= 0.5*max(max(ERMSF)));
                        %save(filename,'ERMSF_interp','WL','mask_thr','Binary','threshold');
                        % plot
%                         whitethreshold = .05;
%                         blackthreshold = .05;
%                         CmapB = 1-([blackthreshold:1/255:1-whitethreshold ; blackthreshold:1/255:1-whitethreshold ; blackthreshold:1/255:1-whitethreshold]');
                         figure
                         imagesc(flipud(Binary));
                         CMap=[0,0,0; 1,1,1];
                         colormap(CMap);
%                         colormap(1-CmapB)
%                         set(gca,'YDir','normal');
                        axis square;axis off;
                        set(gcf,'color','white');
                        set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
                        set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
                        drawnow;
                        print([figure_output_path,'Binary_',processed_filename],'-dpng', '-r600'); 
                        %imwrite(flipud(Binary),[figure_output_path,'Binary_',processed_filename,'.png'],'png');         
%                         %% intersection over union
%                         labelname=[image_label_path,'label_',specimen_folder,'_',filename,'.png'];
%                         A=imread(labelname)/255;
%                         IoU=intersect_over_union_fun(flipud(Binary),logical(A));
%                         area=sum(sum(Binary))/(Nx*Ny)*WL(1)*1e3*WL(2)*1e3; % [mm^2]
%                         disp('Intersection over union: ');IoU
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        processed_filename = ['Riesz_',filename]; % filename of processed .mat data
                        figure;
                       imagesc(A);
                       colormap(Cmap);
                       set(gca,'YDir','normal');axis square;axis off;
                       set(gcf,'color','white');
                       
                       Smin=0;
                       Smax=max(max(A));
                       set(gcf,'Renderer','zbuffer');
                       caxis([caxis_cut*Smin,caxis_cut*Smax]);
                        set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
                        set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
                        % remove unnecessary white space
                        %set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
                        set(gcf,'PaperPositionMode','auto');
                        drawnow;
                        print([figure_output_path,processed_filename],'-dpng', '-r600'); 
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        processed_filename = ['Riesz_adapt_',filename]; % filename of processed .mat data
                        figure;
                       imagesc(Aadapt);
                       colormap(Cmap);
                       set(gca,'YDir','normal');axis square;axis off;
                       set(gcf,'color','white');
                       
                       Smin=0;
                       Smax=max(max(Aadapt));
                       set(gcf,'Renderer','zbuffer');
                       caxis([caxis_cut*Smin,caxis_cut*Smax]);
                        set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
                        set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
                        % remove unnecessary white space
                        %set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
                        set(gcf,'PaperPositionMode','auto');
                        drawnow;
                        print([figure_output_path,processed_filename],'-dpng', '-r600'); 
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        processed_filename = ['Riesz_adapt2_',filename]; % filename of processed .mat data
                        figure;
                       imagesc(Aadapt2);
                       colormap(Cmap);
                       set(gca,'YDir','normal');axis square;axis off;
                       set(gcf,'color','white');
                       
                       Smin=0;
                       Smax=max(max(Aadapt2));
                       set(gcf,'Renderer','zbuffer');
                       caxis([caxis_cut*Smin,caxis_cut*Smax]);
                        set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
                        set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
                        % remove unnecessary white space
                        %set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
                        set(gcf,'PaperPositionMode','auto');
                        drawnow;
                        print([figure_output_path,processed_filename],'-dpng', '-r600'); 
                        
                       close all;
                        % 
            %% END OF PROCESSING
            [filepath,name,ext] = fileparts(filename);
            fprintf('Successfully processed:\n%s\n', name);% successfully processed
        
    else
        fprintf('Filename: \n%s \nalready exist\n', processed_filename);
    end
end



