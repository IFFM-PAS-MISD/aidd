clear all; close all;

% load projectroot path
load project_paths projectroot src_path;
output_path = fullfile(projectroot,'reports','beamer_presentations','Lectures','Wave_Propagation_Fundamentals_1','figs',filesep);
fig_width = 10; fig_height = 6; 
logoblue=[1,67,140]/255; % line color
darkblue=[0,0,139]/255;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% waves of group and phase velocities of opposite sign
x=0:0.002:2;
c=0;
for t=0:1/50:1-1/50
    c=c+1;
    y=(5/8-3/8*cos(2*pi*(x-t))).*cos(2*pi*(10*x+4*t));
    figure;
    figfilename=['wave_opposite_group_phase_velocity_',num2str(c)];
    plot(x,y,'Color',logoblue,'Linewidth',1);

    ax = gca;
    ax.XAxisLocation = 'origin';
    ax.YAxisLocation = 'origin';

    set(gca,'Fontsize',10,'linewidth',1);
    fig = gcf;
    set(fig,'Color','w');
    set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
    % remove unnecessary white space
    set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
    fig.PaperPositionMode   = 'auto';
    print([output_path,figfilename],'-dpng', '-r600'); 
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c=0;
for t=0:1/50:1-1/50
    c=c+1;
    y=(5/8-3/8*cos(2*pi*(x-t))).*cos(2*pi*(10*x-4*t));
    figure;
    figfilename=['wave_phase_slower_than_group_velocity_',num2str(c)];
    plot(x,y,'Color',logoblue,'Linewidth',1);

    ax = gca;
    ax.XAxisLocation = 'origin';
    ax.YAxisLocation = 'origin';

    set(gca,'Fontsize',10,'linewidth',1);
    fig = gcf;
    set(fig,'Color','w');
    set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
    % remove unnecessary white space
    set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
    fig.PaperPositionMode   = 'auto';
   print([output_path,figfilename],'-dpng', '-r600'); 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c=0;
for t=0:1/50:1-1/50
    c=c+1;
    y=(5/8-3/8*cos(2*pi*(x-t))).*cos(2*pi*(10*x-10*t));
    figure;
    figfilename=['wave_phase_equal_group_velocity_',num2str(c)];
    plot(x,y,'Color',logoblue,'Linewidth',1);

    ax = gca;
    ax.XAxisLocation = 'origin';
    ax.YAxisLocation = 'origin';

    set(gca,'Fontsize',10,'linewidth',1);
    fig = gcf;
    set(fig,'Color','w');
    set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
    % remove unnecessary white space
    set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
    fig.PaperPositionMode   = 'auto';
   print([output_path,figfilename],'-dpng', '-r600'); 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c=0;
for t=0:1/50:1-1/50
    c=c+1;
    y=(5/8-3/8*cos(2*pi*(x-t))).*cos(2*pi*(12*x-20*t));
    figure;
    figfilename=['wave_phase_faster_than_group_velocity_',num2str(c)];
    plot(x,y,'Color',logoblue,'Linewidth',1);

    ax = gca;
    ax.XAxisLocation = 'origin';
    ax.YAxisLocation = 'origin';

    set(gca,'Fontsize',10,'linewidth',1);
    fig = gcf;
    set(fig,'Color','w');
    set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
    % remove unnecessary white space
    set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
    fig.PaperPositionMode   = 'auto';
   print([output_path,figfilename],'-dpng', '-r600'); 
end