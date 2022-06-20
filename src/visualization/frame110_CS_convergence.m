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
figure_output_path = prepare_figure_paths(modelname);

model_output_path = prepare_data_processing_paths('processed','exp','compressive','compressive_sensing_frame110_points');
x_points = 128;
y_points = 128;
p_frame = 110;
mask_name = 'random';
load([model_output_path,filesep,'points_metrics_',num2str(x_points), 'x', num2str(y_points),'_frane_',num2str(p_frame),'_',mask_name],'PSNR_metric','SSIM_metric','PEARSON_metric','MSE_metric','PSNR_metric_delam','SSIM_metric_delam','PEARSON_metric_delam','MSE_metric_delam','parameter_points');


figure;
yyaxis left;
plot(parameter_points,PSNR_metric,'o-','LineWidth',1);
ylim([10 30]);
yyaxis right;
plot(parameter_points,PEARSON_metric,'v-','LineWidth',1);
ylim([0 1]);
run font_param;
leg1=legend('PSNR','PEARSON CC','Location','southoutside','Fontsize',legend_font_size,'FontName','Times');
set(leg1,'Box','off');
title('CS: frame 110','Fontsize',title_font_size,'FontName','Times');
xlabel({'$N_p$'},'Fontsize',label_font_size,'interpreter','latex');
set(gcf,'color','white');
run fig_param5;
print([figure_output_path,'points_metrics_',num2str(x_points), 'x', num2str(y_points),'_frame_',num2str(p_frame),'_',mask_name],'-dpng','-r600');

% metrics at delamination


figure;
yyaxis left;
plot(parameter_points,PSNR_metric_delam,'o-','LineWidth',1);
ylim([10 30]);
yyaxis right;
plot(parameter_points,PEARSON_metric_delam,'v-','LineWidth',1);
ylim([0 1]);
run font_param;
leg2=legend('PSNR','PEARSON CC','Location','southoutside','Fontsize',legend_font_size,'FontName','Times');
set(leg2,'Box','off');
title('CS: frame 110 (delam)','Fontsize',title_font_size,'FontName','Times');
xlabel({'$N_p$'},'Fontsize',label_font_size,'interpreter','latex');
run fig_param5;
print([figure_output_path,'points_metrics_delam_',num2str(x_points), 'x', num2str(y_points),'_frame_',num2str(p_frame),'_',mask_name],'-dpng','-r600');

