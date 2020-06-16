clear all; close all;

% load projectroot path
load project_paths projectroot src_path;
output_path = fullfile(projectroot,'reports','beamer_presentations','Lectures','Wave_Propagation_Fundamentals_1','figs',filesep);
fig_width = 10; fig_height = 10; 
logoblue=[1,67,140]/255; % line color
darkblue=[0,0,139]/255;

x=2:0.01:4*pi+0.2;
y=2*sin(3*x+pi/6-0.2);
figure;
figfilename='travelling_wave_string';
plot(x,y,'Color',logoblue,'Linewidth',1);
% xlabel('x');
% ylabel('y');
ax = gca;
% ax.XAxisLocation = 'origin';
% ax.YAxisLocation = 'origin';
axis([-2.2 12.5 -2.2 2.2]);
axis off;
hold on;

circle(0,0,2);
axis equal
text(-0.2, 0.8, {'\omega'});
arc(0,0,1.5,45,135);
hold on;
hq1=quiver(-1.061,1.061,-0.2,-0.2,'Color','k');
%get the data from regular quiver
U1 = hq1.UData;
V1 = hq1.VData;
X1 = hq1.XData;
Y1 = hq1.YData;
% modify arrow heads by using annotation
headWidth = 5;
headLength = 8;
LineLength = 1;%0.08;
ah = annotation('arrow',...
            'headStyle','cback1','HeadLength',headLength,'HeadWidth',headWidth);
        set(ah,'parent',gca);
        set(ah,'position',[X1 Y1 LineLength*U1 LineLength*V1]);
hold on
plot(2,0,'ko','MarkerFaceColor','k');
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600'); 

function h = circle(x,y,r)
    hold on
    th = 0:pi/50:2*pi;
    xunit = r * cos(th) + x;
    yunit = r * sin(th) + y;
    h = plot(xunit, yunit);
    hold off
end

function h = arc(x,y,r,th1, th2)
    hold on
    th = [th1:1:th2]*pi/180;
    xunit = r * cos(th) + x;
    yunit = r * sin(th) + y;
    h = plot(xunit, yunit,'k');
    hold off
end
