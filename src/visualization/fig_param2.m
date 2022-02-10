
drawnow;
%fig_w = 7;  % one column
fig_w = 14; % two columns
fig_h = 5;
set(gcf,'color','white');
set(gcf,'Renderer','zbuffer');
%set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
set(gcf, 'Units','centimeters', 'Position',[10 10 fig_w fig_h]); 
set(gcf,'PaperPositionMode','auto');