% copy figures to paper folder and rename

load project_paths projectroot src_path;


paper_folder = 'Compressive_Sensing_paper';
fig_destination=[projectroot,'reports',filesep,'journal_papers',filesep,paper_folder,filesep,'figs',filesep];

figs_source_folder=[projectroot,'reports',filesep,'figures',filesep,'compressive',filesep];
modelname='compressive_sensing_frame110_points';
% figure 9a
figname='points_metrics_128x128_frame_110_random.png';
fig_source=[figs_source_folder,modelname,'_out',filesep,figname];
copyfile(fig_source,fig_destination);
% rename
movefile([fig_destination,figname],[fig_destination,'figure9a.png'],'f');

% figure 9b
figname='points_metrics_delam_128x128_frame_110_random.png';
fig_source=[figs_source_folder,modelname,'_out',filesep,figname];
copyfile(fig_source,fig_destination);
% rename
movefile([fig_destination,figname],[fig_destination,'figure9b.png'],'f');

% stitched figures 9 together
I9a=imread([fig_destination,'figure9a.png']);
I9b=imread([fig_destination,'figure9b.png']);
I9=[I9a(1:1190,1:1417,:),I9b(1:1190,1:1417,:)];
I1info=imfinfo([fig_destination,'figure9a.png']);
imwrite(I9,[fig_destination,'figure9.png'],'png','ResolutionUnit','meter','XResolution',I1info.XResolution,'YResolution',I1info.YResolution);

figs_source_folder=[projectroot,'reports',filesep,'figures',filesep,'compressive',filesep];
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

% stitched figures 12 together
I12a=imread([fig_destination,'figure12a.png']);
I12b=imread([fig_destination,'figure12b.png']);
I12c=imread([fig_destination,'figure12c.png']);
I12=[I12a(1:1181,1:3306,:);I12b(1:1181,1:3306,:);I12c(1:1181,1:3306,:)];
I1info=imfinfo([fig_destination,'figure12a.png']);
imwrite(I12,[fig_destination,'figure12.png'],'png','ResolutionUnit','meter','XResolution',I1info.XResolution,'YResolution',I1info.YResolution);

