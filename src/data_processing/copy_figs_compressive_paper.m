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
fig_source=[figs_source_folder,modelname,'_out',filesep,'random',filesep,'4000p',filesep,figname];
copyfile(fig_source,fig_destination);
% rename
movefile([fig_destination,figname],[fig_destination,'figure10d.png'],'f');

% figure 10e
figs_source_folder=[projectroot,'reports',filesep,'figures',filesep];
modelname='frame110_DLSR';
figname='frame110_DLSR_model_1.png';
fig_source=[figs_source_folder,modelname,'_out',filesep,figname];
copyfile(fig_source,fig_destination);
% rename
movefile([fig_destination,figname],[fig_destination,'figure10e.png'],'f');

figs_source_folder=[projectroot,'reports',filesep,'figures',filesep,'compressive',filesep];
modelname='compressive_sensing_all_frames';

% figure 11a
figname='ref_delam_128x128p_siatka_1024_klatka_110_default_random.png';
fig_source=[figs_source_folder,modelname,'_out',filesep,'random',filesep,'1024p',filesep,figname];
copyfile(fig_source,fig_destination);
% rename
movefile([fig_destination,figname],[fig_destination,'figure11a.png'],'f');

% figure 11b
figname='recon_delam_128x128p_siatka_1024_klatka_110_default_random.png';
fig_source=[figs_source_folder,modelname,'_out',filesep,'random',filesep,'1024p',filesep,figname];
copyfile(fig_source,fig_destination);
% rename
movefile([fig_destination,figname],[fig_destination,'figure11b.png'],'f');

% figure 11c
figname='recon_delam_128x128p_siatka_3000_klatka_110_default_random.png';
fig_source=[figs_source_folder,modelname,'_out',filesep,'random',filesep,'3000p',filesep,figname];
copyfile(fig_source,fig_destination);
% rename
movefile([fig_destination,figname],[fig_destination,'figure11c.png'],'f');

% figure 11d
figname='recon_delam_128x128p_siatka_4000_klatka_110_default_random.png';
fig_source=[figs_source_folder,modelname,'_out',filesep,'random',filesep,'4000p',filesep,figname];
copyfile(fig_source,fig_destination);
% rename
movefile([fig_destination,figname],[fig_destination,'figure11d.png'],'f');

% figure 11e
figs_source_folder=[projectroot,'reports',filesep,'figures',filesep];
modelname='frame110_delam_DLSR';
figname='frame110_delam_DLSR_model_1.png';

fig_source=[figs_source_folder,modelname,'_out',filesep,figname];
copyfile(fig_source,fig_destination);
% rename
movefile([fig_destination,figname],[fig_destination,'figure11e.png'],'f');

figs_source_folder=[projectroot,'reports',filesep,'figures',filesep,'compressive',filesep];
modelname='compressive_sensing_all_frames';

% figure 12a
figname='frame_metrics_128x128_points_3000_random.png';
fig_source=[figs_source_folder,modelname,'_out',filesep,'random',filesep,'3000p',filesep,figname];
copyfile(fig_source,fig_destination);
% rename
movefile([fig_destination,figname],[fig_destination,'figure12a.png'],'f');

% figure 12b
figname='frame_metrics_128x128_points_4000_random.png';
fig_source=[figs_source_folder,modelname,'_out',filesep,'random',filesep,'4000p',filesep,figname];
copyfile(fig_source,fig_destination);
% rename
movefile([fig_destination,figname],[fig_destination,'figure12b.png'],'f');

% figure 12c
figs_source_folder=[projectroot,'reports',filesep,'figures',filesep];
modelname='frame_metrics_DLSR';
figname='frame_metrics_DLSR_model_1.png';
fig_source=[figs_source_folder,modelname,'_out',filesep,figname];
copyfile(fig_source,fig_destination);
% rename
movefile([fig_destination,figname],[fig_destination,'figure12c.png'],'f');


