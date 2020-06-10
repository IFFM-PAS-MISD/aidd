clear all; close all;
% step function approximation by Fourier series

% load projectroot path
load project_paths projectroot src_path;
output_path = fullfile(projectroot,'reports','beamer_presentations','Lectures','Fourier_Transform_1','figs',filesep);
fig_width = 5; fig_height = 5; 
logoblue=[1,67,140]/255; % line color
darkblue=[0,0,139]/255;
x=0:1/250:1;



% y1=4/pi*(cos(pi*x)/1);
% 
% figure;plot(x,y1);
% 
% y2=4/pi*((-1)*cos(3*pi*x)/3);
% 
% figure;plot(x,y1+y2);
% 
% y3=4/pi*((-1)^2*cos(5*pi*x)/5);
% 
% figure;plot(x,y1+y2+y3);

N=25;
y=zeros(N,length(x));
c=1;
for k=1:N
    if(mod(k,2)) p=2; else p=1; end
    y(k,:) = 4/pi*((-1)^p*cos(c*pi*x)/c);
    ysum = sum(y,1);
    figfilename = ['Fourier_series_',num2str(k)];
    figure;
    %plot(x,ysum,'r','Linewidth',1);
    plot(x,ysum,'Color',logoblue,'Linewidth',1);
    title(['Fourier series for n=',num2str(k)],'Fontsize',11);
    set(gca,'Fontsize',10,'linewidth',1);
    fig = gcf;
    set(fig,'Color','w');
    xticks([0,0.5,1]);
    yticks([-1,0,1]); % works in new matlab
    axis([0 1 -1.5,1.5]);
    
        set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
        % remove unnecessary white space
        set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
        fig.PaperPositionMode   = 'auto';
        print([output_path,figfilename],'-dpng', '-r600'); 
    c=c+2;
end

% plot step function
figfilename='step_function';
figure;
x1=0:1/100:0.5;
y1= zeros(1,length(x1))+1;
%plot(x1,y1,'Color',[0.9375, 0, 0],'Linewidth',1); % red
plot(x1,y1,'Color',logoblue,'Linewidth',1); % red
title('Step function','Fontsize',11);
hold on;
plot(0.5,0,'ko','MarkerSize',4,'markerfacecolor','k');
x2=0.5:1/100:1;
y2= zeros(1,length(x1))-1;
%plot(x2,y2,'Color',[0,0,0.625],'Linewidth',1); % blue
plot(x2,y2,'Color',logoblue,'Linewidth',1); % blue
set(gca,'Fontsize',10,'linewidth',1);
fig = gcf;
set(fig,'Color','w');
xticks([0,0.5,1]);
yticks([-1,0,1]); % works in new matlab
axis([0 1 -1.5,1.5]);

set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
fig.PaperPositionMode   = 'auto';
print([output_path,figfilename],'-dpng', '-r600'); 
