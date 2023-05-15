%% script for computing root mean square (rms)
clear all; close all;
% extract name of the run script
currentFile = mfilename('fullpath');
[pathstr,name,ext] = fileparts( currentFile );

%s
specimen_name = 'specimen_1';
processing_name = {'rms', 'rms_norm'};
filename = '50kHz_pzt';
for proc_no=1:length(processing_name)
    % load data
    load(['../../data/processed/',specimen_name,'/',processing_name{proc_no},'/',filename,'_',processing_name{proc_no},'.mat']);
    %% visualize data
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % figure parameters
    fig_w = 14; % figure width in cm
    fig_h = 12; % figure height in cm
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fig1 = figure;
    figname = [name,'_',specimen_name,'_',filename,'_',processing_name{proc_no}];
    switch proc_no
        case 1
            plot(linspace(0,L,length(Data_rms)),Data_rms,'r-','LineWidth',1);
            ylabel('RMS');
        case 2
            plot(linspace(0,L,length(Data_rms_norm)),Data_rms_norm,'b-','LineWidth',1);
            ylabel('RMSnorm');
    end
    xlabel('x [m]');
    
    xlim([0 L]);
    set(gca,'linewidth',1);
    set(fig1,'color','white');
    set(fig1, 'Units','centimeters', 'Position',[10 10 fig_w fig_h]); 
    set(fig1,'PaperPositionMode','auto');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%
    % check if directory exist; if not, create it
    figure_output_path = ['../../reports/figures/',name,'/'];
    if ~exist(figure_output_path, 'dir')
        mkdir(figure_output_path);
    end
    
    % save figure
    exportgraphics(fig1,[figure_output_path,figname,'.png'], 'Resolution',600);
    close(fig1);
end