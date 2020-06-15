clear all; close all;

% load projectroot path
load project_paths projectroot src_path;
output_path = fullfile(projectroot,'reports','beamer_presentations','Lectures','Wave_Propagation_Fundamentals_1','figs',filesep);
fig_width = 10; fig_height = 6; 
logoblue=[1,67,140]/255; % line color
darkblue=[0,0,139]/255;

y0=2;
L=1;
x=0:0.01:L;
n=1:6; % six first harmonics
omega=18;
T=2*pi/omega;
c=0;
for t=0:T/32:T/2
    c=c+1;
    figfilename = ['standing_wave_',num2str(c)];
    % fundamental, first harmonic
    n=1;
    k=n*pi/L;
    V=omega./k;
    lambda=2*L./n;
    f=n*V/(2*L);
    m=0:n;
    xnodes=m*L/(n);
    y=2*y0*sin(k*x)*cos(omega*t);
    figure;subplot(2,2,1);
    plot(x,y,'Color',logoblue,'Linewidth',1);
    hold on;
    plot(xnodes,zeros(n+1,1),'ko','MarkerFaceColor','k');
    axis([0 L -2*y0-y0/10 2*y0+y0/10]);
    set(gca,'Fontsize',10,'linewidth',1);
   
    
    % second harmonic
    n=2;
    k=n*pi/L;
    V=omega./k;
    lambda=2*L./n;
    f=n*V/(2*L);
    m=0:n;
    xnodes=m*L/(n);
    y=2*y0*sin(k*x)*cos(omega*t);
    subplot(2,2,2);
    plot(x,y,'Color',logoblue,'Linewidth',1);
    hold on;
    plot(xnodes,zeros(n+1,1),'ko','MarkerFaceColor','k');
    axis([0 L -2*y0-y0/10 2*y0+y0/10]);
    
    % third harmonic
    n=3;
    k=n*pi/L;
    V=omega./k;
    lambda=2*L./n;
    f=n*V/(2*L);
    m=0:n;
    xnodes=m*L/(n);
    y=2*y0*sin(k*x)*cos(omega*t);
    subplot(2,2,3);
    plot(x,y,'Color',logoblue,'Linewidth',1);
    hold on;
    plot(xnodes,zeros(n+1,1),'ko','MarkerFaceColor','k');
    axis([0 L -2*y0-y0/10 2*y0+y0/10]);
    
    % fourth harmonic
    n=4;
    k=n*pi/L;
    V=omega./k;
    lambda=2*L./n;
    f=n*V/(2*L);
    m=0:n;
    xnodes=m*L/(n);
    y=2*y0*sin(k*x)*cos(omega*t);
    subplot(2,2,4);
    plot(x,y,'Color',logoblue,'Linewidth',1);
    hold on;
    plot(xnodes,zeros(n+1,1),'ko','MarkerFaceColor','k');
    axis([0 L -2*y0-y0/10 2*y0+y0/10]);
    
    %
    set(gca,'Fontsize',10,'linewidth',1);
    fig = gcf;
    set(fig,'Color','w');
    set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
    % remove unnecessary white space
    set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
    fig.PaperPositionMode   = 'auto';
    print([output_path,figfilename],'-dpng', '-r600'); 
end
close all;