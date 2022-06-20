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

model_output_path = prepare_data_processing_paths('processed','exp','compressive','compressive_sensing_all_frames');
x_points = 128;
y_points = 128;
mask_name = 'random';
for points = [3000,4000]

    load([model_output_path,filesep,'frame_metrics_',num2str(x_points), 'x', num2str(y_points),'_points_',num2str(points),'_',mask_name],'PSNR_metric','SSIM_metric','PEARSON_metric','MSE_metric','PSNR_metric_delam','SSIM_metric_delam','PEARSON_metric_delam','MSE_metric_delam','parameter_frames');


    figure;
    yyaxis left;
    plot(parameter_frames,PSNR_metric,'LineWidth',1);
    ylim([10 60]);
    yyaxis right;
    plot(parameter_frames,PEARSON_metric,'LineWidth',1);
    ylim([-0.5 1]);
    run font_param;
    legend('PSNR','PEARSON CC','Location','east','Fontsize',legend_font_size,'FontName','Times');
    title(['CS: ', num2str(points),' points'],'Fontsize',title_font_size,'FontName','Times');
    xlabel({'$N_f$'},'Fontsize',label_font_size,'interpreter','latex');
    set(gcf,'color','white');
    run fig_param2;
    print([figure_output_path,'frame_metrics_',num2str(x_points), 'x', num2str(y_points),'_points_',num2str(points),'_',mask_name],'-dpng','-r600');

    % metrics at delamination

    figure;
    yyaxis left;
    plot(parameter_frames,PSNR_metric_delam,'LineWidth',1);
    ylim([10 60]);
    yyaxis right;
    plot(parameter_frames,PEARSON_metric_delam,'LineWidth',1);
    ylim([-0.5 1]);
    run font_param;
    legend('PSNR','PEARSON CC','Location','southeast','Fontsize',legend_font_size,'FontName','Times');
    title(['CS: ', num2str(points),' points (delam)'],'Fontsize',title_font_size,'FontName','Times');
    xlabel({'$N_f$'},'Fontsize',label_font_size,'interpreter','latex');
    run fig_param2;
print([figure_output_path,'frame_metrics_delam_',num2str(x_points), 'x', num2str(y_points),'_points_',num2str(points),'_',mask_name],'-dpng','-r600');
end