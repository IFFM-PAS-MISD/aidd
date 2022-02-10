axis square;axis off;
drawnow;
set(gcf,'color','white');

set(gcf,'Renderer','zbuffer');

set(gca, 'Position',[0 0 1. 1.]); % figure without axis and white border
set(gcf, 'Units','centimeters', 'Position',[10 10 fig_width fig_height]); 
% remove unnecessary white space
%set(gca,'LooseInset', max(get(gca,'TightInset'), 0.));
%set(gca, 'LooseInset', [0,0,0,0]);
set(gca,'LooseInset',get(gca,'TightInset'));
set(gcf,'PaperPositionMode','auto');