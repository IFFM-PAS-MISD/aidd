clear all; close all;
% Heat transfer problem of two connected rods of different temperatures
% solutions given by Fourier series

% load projectroot path
load project_paths projectroot src_path;
output_path = fullfile(projectroot,'reports','beamer_presentations','Lectures','Fourier_Transform_1','figs',filesep);
fig_width = 6; fig_height = 5; 
logoblue=[1,67,140]/255; % line color
darkblue=[0,0,139]/255;
xrod=0:1/250:1;
yrod=[-0.1:0.2/10:0.1];

x_hotrod_1=xrod/2 -0.1; x_coldrod_1=0.5+xrod/2 +0.1; % position 1
x_hotrod_2=xrod/2 -0.05; x_coldrod_2=0.5+xrod/2 +0.05; % position 2
x_hotrod_3=xrod/2 -0.02; x_coldrod_3=0.5+xrod/2 +0.02; % position 3

N=250;
NumberOfTimeSteps=25;

[X,Y] = meshgrid(xrod,yrod);
[X_hotrod_1,Y_hotrod_1]=meshgrid(x_hotrod_1,yrod);[X_coldrod_1,Y_coldrod_1]=meshgrid(x_coldrod_1,yrod);
[X_hotrod_2,Y_hotrod_2]=meshgrid(x_hotrod_2,yrod);[X_coldrod_2,Y_coldrod_2]=meshgrid(x_coldrod_2,yrod);
[X_hotrod_3,Y_hotrod_3]=meshgrid(x_hotrod_3,yrod);[X_coldrod_3,Y_coldrod_3]=meshgrid(x_coldrod_3,yrod);
Thot=zeros(length(yrod),length(xrod))+1;
Tcold=zeros(length(yrod),length(xrod))-1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% two rods coming into contact
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% position 1
    counter=1;t=0;
    figfilename = ['Fourier_series_rod_temperatures_',num2str(counter)];
    figure;   
    surf([X_hotrod_1],Y_hotrod_1,Thot);shading flat; view(0,90);
    
    hold on;
    surf(X_coldrod_1,Y_coldrod_1,Tcold);shading flat; view(0,90);
    %caxis([-1 1]);
    colormap jet;
    cmap = colormap;
    cmap_new=cmap(2:57,:);
    colormap(cmap_new);
    
    txt = ['T(x,t=',num2str(t,'%1.3f'),')'];
    text(0.02,1.3,txt)
    txt1='Hot';txt2='Cold';
    text(0.1,-0.3,txt1);text(0.7,-0.3,txt2);
    txt3=['\rightarrow']; txt4=['\leftarrow'];
    text(0.1,0.4,txt3,'FontSize', 18);text(0.7,0.4,txt4,'FontSize', 18);
    set(gca,'Fontsize',10,'linewidth',1);
    fig = gcf;
    set(fig,'Color','w');
    xticks([0,0.5,1]);
    yticks([-1,0,1]); % works in new matlab
    axis([0 1 -1.5,1.5]);
   
    x1=0:1/100:0.5;
    y1= zeros(1,length(x1))+1;
    %plot(x1,y1,'Color',cmap_new(end,:),'Linewidth',1);
    plot(x1,y1,'Color',logoblue,'Linewidth',1);
    axis off;
    hold on;
    x2=0.5:1/100:1;
    y2= zeros(1,length(x1))-1;
    %plot(x2,y2,'Color',cmap_new(1,:),'Linewidth',1);
    plot(x2,y2,'Color',logoblue,'Linewidth',1);
    line([0,1],[1.5 1.5],'Color','k','Linewidth',1);
    line([1,1],[1.5 -1.5],'Color','k','Linewidth',1);
    ax = gca;
    ax.YAxisLocation = 'origin';
    ax.XLim = [-0.1 1.1];
    ax.YLim = [-1.5  1.5];
    set(gca,'Fontsize',10,'Linewidth',1);
    axis on;
    set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
    % remove unnecessary white space
    set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
    fig.PaperPositionMode   = 'auto';
    print([output_path,figfilename],'-dpng', '-r600'); 

% position 2
    counter=2;t=0;
    figfilename = ['Fourier_series_rod_temperatures_',num2str(counter)];
    figure;   
    surf([X_hotrod_2],Y_hotrod_2,Thot);shading flat; view(0,90);
    
    hold on;
    surf(X_coldrod_2,Y_coldrod_2,Tcold);shading flat; view(0,90);
    %caxis([-1 1]);
    colormap jet;
    cmap = colormap;
    cmap_new=cmap(2:57,:);
    colormap(cmap_new);
    
    txt = ['T(x,t=',num2str(t,'%1.3f'),')'];
    text(0.02,1.3,txt)
    txt1='Hot';txt2='Cold';
    text(0.1,-0.3,txt1);text(0.7,-0.3,txt2);
    txt3=['\rightarrow']; txt4=['\leftarrow'];
    text(0.1,0.4,txt3,'FontSize', 18);text(0.7,0.4,txt4,'FontSize', 18);
    set(gca,'Fontsize',10,'linewidth',1);
    fig = gcf;
    set(fig,'Color','w');
    xticks([0,0.5,1]);
    yticks([-1,0,1]); % works in new matlab
    axis([0 1 -1.5,1.5]);
   
    x1=0:1/100:0.5;
    y1= zeros(1,length(x1))+1;
    %plot(x1,y1,'Color',cmap_new(end,:),'Linewidth',1);
    plot(x1,y1,'Color',logoblue,'Linewidth',1);
    axis off;
    hold on;
    x2=0.5:1/100:1;
    y2= zeros(1,length(x1))-1;
    %plot(x2,y2,'Color',cmap_new(1,:),'Linewidth',1);
    plot(x2,y2,'Color',logoblue,'Linewidth',1);
    line([0,1],[1.5 1.5],'Color','k','Linewidth',1);
    line([1,1],[1.5 -1.5],'Color','k','Linewidth',1);
    ax = gca;
    ax.YAxisLocation = 'origin';
    ax.XLim = [-0.1 1.1];
    ax.YLim = [-1.5  1.5];
    set(gca,'Fontsize',10,'Linewidth',1);
    axis on;
    set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
    % remove unnecessary white space
    set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
    fig.PaperPositionMode   = 'auto';
    print([output_path,figfilename],'-dpng', '-r600');
    
    % position 3
    counter=3;t=0;
    figfilename = ['Fourier_series_rod_temperatures_',num2str(counter)];
    figure;   
    surf([X_hotrod_3],Y_hotrod_3,Thot);shading flat; view(0,90);
    
    hold on;
    surf(X_coldrod_3,Y_coldrod_3,Tcold);shading flat; view(0,90);
    %caxis([-1 1]);
    colormap jet;
    cmap = colormap;
    cmap_new=cmap(2:57,:);
    colormap(cmap_new);
    
    txt = ['T(x,t=',num2str(t,'%1.3f'),')'];
    text(0.02,1.3,txt)
    txt1='Hot';txt2='Cold';
    text(0.1,-0.3,txt1);text(0.7,-0.3,txt2);
    txt3=['\rightarrow']; txt4=['\leftarrow'];
    text(0.1,0.4,txt3,'FontSize', 18);text(0.7,0.4,txt4,'FontSize', 18);
    set(gca,'Fontsize',10,'linewidth',1);
    fig = gcf;
    set(fig,'Color','w');
    xticks([0,0.5,1]);
    yticks([-1,0,1]); % works in new matlab
    axis([0 1 -1.5,1.5]);
   
    x1=0:1/100:0.5;
    y1= zeros(1,length(x1))+1;
    %plot(x1,y1,'Color',cmap_new(end,:),'Linewidth',1);
    plot(x1,y1,'Color',logoblue,'Linewidth',1);
    axis off;
    hold on;
    x2=0.5:1/100:1;
    y2= zeros(1,length(x1))-1;
    %plot(x2,y2,'Color',cmap_new(1,:),'Linewidth',1);
    plot(x2,y2,'Color',logoblue,'Linewidth',1);
    line([0,1],[1.5 1.5],'Color','k','Linewidth',1);
    line([1,1],[1.5 -1.5],'Color','k','Linewidth',1);
    ax = gca;
    ax.YAxisLocation = 'origin';
    ax.XLim = [-0.1 1.1];
    ax.YLim = [-1.5  1.5];
    set(gca,'Fontsize',10,'Linewidth',1);
    axis on;
    set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
    % remove unnecessary white space
    set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
    fig.PaperPositionMode   = 'auto';
    print([output_path,figfilename],'-dpng', '-r600');
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% temperature evolution over time
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
a=1;

for t=[0:1/(NumberOfTimeSteps-1):1]
    counter=counter+1;
    y=zeros(N,length(xrod));
    c=1;
    for k=1:N
        if(mod(k,2)) p=2; else p=1; end
        y(k,:) = 4/pi*((-1)^p*cos(c*pi*xrod)/c*exp(-a*c^2*t));
        c=c+2;
    end
    ysum = sum(y,1);
    figfilename = ['Fourier_series_rod_temperatures_',num2str(counter)];
    figure;
    T=repmat(ysum',[1,length(yrod)])';
    surf(X,Y,T);shading interp; view(0,90);caxis([-1 1]);
    colormap jet;
    cmap = colormap;
    cmap_new=cmap(2:57,:);
    colormap(cmap_new);
    hold on;
    %plot(xrod,ysum,'r','Linewidth',1);
    plot3(xrod,ysum,zeros(1,length(xrod))+1,'Color',logoblue,'Linewidth',1);
    txt = ['T(x,t=',num2str(t,'%1.3f'),')'];
    text(0.02,1.3,txt)
    box off;
    set(gca,'Fontsize',10,'linewidth',1);
    fig = gcf;
    set(fig,'Color','w');
    xticks([0,0.5,1]);
    yticks([-1,0,1]); % works in new matlab
    axis([0 1 -1.5,1.5]);
    ax = gca;
    ax.YAxisLocation = 'origin';
    ax.XLim = [-0.1 1.1];
    ax.YLim = [-1.5  1.5];
    line([0,1],[1.5 1.5],'Color','k','Linewidth',1);
    line([1,1],[1.5 -1.5],'Color','k','Linewidth',1);
    

    set(fig, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
    % remove unnecessary white space
    set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02));
    fig.PaperPositionMode   = 'auto';
    print([output_path,figfilename],'-dpng', '-r600'); 
    

end

