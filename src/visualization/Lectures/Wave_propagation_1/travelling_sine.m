clear all; close all;

% load projectroot path
load project_paths projectroot src_path;
output_path = fullfile(projectroot,'reports','beamer_presentations','Lectures','Wave_Propagation_Fundamentals_1','figs',filesep);
fig_width = 10; fig_height = 6; 
logoblue=[1,67,140]/255; % line color
darkblue=[0,0,139]/255;

x=-pi/6:0.01:pi+0.2;
y=2*sin(3*x);
figure;
figfilename='travelling_sine_1';
plot(x,y,'Color',logoblue,'Linewidth',1);
xlabel('x');
ylabel('y');
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';
axis([-pi/6 pi+0.2 -2.1 3.3]);
xticks([0,pi/3,2*pi/3]);
xticklabels({'0','\pi/3','2\pi/3'});
hold on;
hq1=quiver(pi/3,2.2,pi/3,0,0,'Color','k');
hq2=quiver(pi/3,2.2,-pi/3,0,0,'Color','k');

%get the data from regular quiver
U1 = hq1.UData;
V1 = hq1.VData;
X1 = hq1.XData;
Y1 = hq1.YData;
U2 = hq2.UData;
V2 = hq2.VData;
X2 = hq2.XData;
Y2 = hq2.YData;
% modify arrow heads by using annotation
headWidth = 5;
headLength = 8;
LineLength = 1;%0.08;
for ii = 1:length(X1)
    for ij = 1:length(X1)
        ah = annotation('arrow',...
            'headStyle','cback1','HeadLength',headLength,'HeadWidth',headWidth);
        set(ah,'parent',gca);
        set(ah,'position',[X1(ii,ij) Y1(ii,ij) LineLength*U1(ii,ij) LineLength*V1(ii,ij)]);
    end
end
for ii = 1:length(X2)
    for ij = 1:length(X2)
        ah = annotation('arrow',...
            'headStyle','cback1','HeadLength',headLength,'HeadWidth',headWidth);
        set(ah,'parent',gca);
        set(ah,'position',[X2(ii,ij) Y2(ii,ij) LineLength*U2(ii,ij) LineLength*V2(ii,ij)]);
    end
end
txt={'\lambda = 2\pi/3'};
text(pi/4,2.5,txt);
title('y=2 sin(3x)');
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600'); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t=0;
y=2*sin(3*(x-6*t));
figure;
figfilename='travelling_sine_2';
plot(x,y,'Color',logoblue,'Linewidth',1);
xlabel('x');
ylabel('y');
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';
axis([-pi/6 pi+0.2 -2.1 3.3]);
xticks([0,pi/3,2*pi/3]);
xticklabels({'0','\pi/3','2\pi/3'});
hold on;
txt='t=0';
text(pi/3-0.1,1,txt,'Color',logoblue);
% wave crest
plot([pi/6,5*pi/6],[2,2],'o','Color',darkblue,'MarkerFaceColor',darkblue);
text(5*pi/6-0.15,2.4,'crest');
% wave trough
plot([-pi/6,3*pi/6],[-2,-2],'o','Color','r','MarkerFaceColor','r');
text(pi/2-0.2,-1.5,'trough');
title('y=2 sin(3(x-6t))');
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600'); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T=2*pi/18;
t=T/6;
y1=2*sin(3*(x-6*t));
figure;
figfilename='travelling_sine_3';
plot(x,y,'Color',logoblue,'Linewidth',1);
hold on
plot(x,y1,'Color','g','Linewidth',1);
xlabel('x');
ylabel('y');
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';
axis([-pi/6 pi+0.2 -2.1 3.3]);
xticks([0,pi/3,2*pi/3]);
xticklabels({'0','\pi/3','2\pi/3'});
hold on;
txt='t=0';
text(pi/3-0.1,1,txt,'Color',logoblue);
txt = ['t=',num2str(t,'%1.3f')];
text(pi/3+0.22,1,txt,'Color','g');
% wave crest
plot([pi/6,5*pi/6],[2,2],'o','Color',darkblue,'MarkerFaceColor',darkblue);
text(5*pi/6-0.15,2.4,'crest');
% wave trough
plot([-pi/6,3*pi/6],[-2,-2],'o','Color','r','MarkerFaceColor','r');
text(pi/2-0.2,-1.5,'trough');
title('y=2 sin(3(x-6t))');
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600'); 