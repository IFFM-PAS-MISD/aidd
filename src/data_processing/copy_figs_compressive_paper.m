% copy figures to paper folder and rename

load project_paths projectroot src_path;

figs_source_folder=[projectroot,'reports',filesep,'figures',filesep,'compressive',filesep];
paper_folder = 'Compressive_Sensing_paper';
fig_destination=[projectroot,'reports',filesep,'journal_papers',filesep,paper_folder,filesep,'figs',filesep];

modelname='compressive_sensing_all_frames';

% figure 10a
figname='ref_rect_128x128p_siatka_1024_klatka_110_default_random.png';
fig_source=[figs_source_folder,modelname,'_out',filesep,'random',filesep,'1024p',filesep,figname];
copyfile(fig_source,fig_destination);
% rename
movefile([fig_destination,figname],[fig_destination,'figure10a.png'],'f');

% figure 10b
figname='recon_rect_128x128p_siatka_1024_klatka_110_default_random.png';
fig_source=[figs_source_folder,modelname,'_out',filesep,'random',filesep,'1024p',filesep,figname];
copyfile(fig_source,fig_destination);
% rename
movefile([fig_destination,figname],[fig_destination,'figure10b.png'],'f');

% figure 10c
figname='recon_rect_128x128p_siatka_3000_klatka_110_default_random.png';
fig_source=[figs_source_folder,modelname,'_out',filesep,'random',filesep,'3000p',filesep,figname];
copyfile(fig_source,fig_destination);
% rename
movefile([fig_destination,figname],[fig_destination,'figure10c.png'],'f');

% figure 10d
figname='recon_rect_128x128p_siatka_4000_klatka_110_default_random.png';
fig_source=[figs_source_folder,modelname,'_out',filesep,'random',filesep,'3000p',filesep,figname];
copyfile(fig_source,fig_destination);
% rename
movefile([fig_destination,figname],[fig_destination,'figure10c.png'],'f');

