%% copy figures to mssp_paper folder and rename

% retrieve path of currently running file
currentFile=mfilename('fullpath');
[pathstr,~,~]=fileparts( currentFile );
% extract projectroot path
projectroot=fullfile(pathstr(1:end-9));% we cut 'src/tools'

paper_folder = '2023_mssp_paper';
% we have to use full paths here instead of relative paths
fig_destination=[projectroot,'reports/journal_papers/',paper_folder,'/figs/'];
figs_source_folder=[projectroot,'reports/figures/'];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure 1
specimen_name = 'specimen_1';
visualization_name = 'plot_rms';
filename = '50kHz_pzt';
figname = [visualization_name,'_',specimen_name,'_',filename,'.png'];
fig_source=[figs_source_folder,visualization_name,filesep,figname];

copyfile(fig_source,fig_destination);
% rename
movefile([fig_destination,figname],[fig_destination,'figure1.png'],'f');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
