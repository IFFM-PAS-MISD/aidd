clear all; close all;

% load projectroot path
load project_paths projectroot src_path;
output_path = fullfile(projectroot,'reports','beamer_presentations','Lectures','Wave_Propagation_Fundamentals_1','figs',filesep);
fig_width = 10; fig_height = 6; 
logoblue=[1,67,140]/255; % line color
darkblue=[0,0,139]/255;

x=-1:0.1:12;
y=1/3*x;
figure;
figfilename='travelling_line_1';
plot(x,y,'Color',logoblue,'Linewidth',1);
xlabel('x');
ylabel('y');
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';
axis([-1 12 -3 4]);
txt='y=1/3 x';
text(7,-2,txt);
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600'); 

%
t=0;
y=1/3*(x-6*t);
figure;
figfilename='travelling_line_2';
plot(x,y,'Color',logoblue,'Linewidth',1);
xlabel('x');
ylabel('y');
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';
axis([-1 12 -3 4]);
txt = ['t=0'];
text(7,3,txt,'Color',logoblue)
txt='y=1/3 (x - 6t)';
text(7,-2,txt)
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600'); 

%
t1=1;
y1=1/3*(x-6*t1);
figure;
figfilename='travelling_line_3';
plot(x,y,'Color',logoblue,'Linewidth',1);
hold on;
plot(x,y1,'Color','g','Linewidth',1);
xlabel('x');
ylabel('y');
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';
axis([-1 12 -3 4]);
txt = ['t=0'];
text(7,3,txt,'Color',logoblue);
txt = ['t=1'];
text(7,1,txt,'Color','g');
txt='y=1/3 (x - 6t)';
text(7,-2,txt);
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600'); 