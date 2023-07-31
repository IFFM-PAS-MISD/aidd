function [Data_polar,R,TH]=Cartesian2PolarWarped(Data)
% convert spatial wavefield to polar coordinates by using interpolation
% input Data of dimensions nx, ny, nt 
% output: Data_polar of dimensions theta, radius, frame_no
%             figure;
%             surf(R(1:1496,:),TH(1:1496,:),squeeze(Data_polar(1:1496,:,180)));
%             shading interp; view(2);
%             set(gcf,'Renderer','zbuffer');
%% check NAN
[m1,n1,p1]=size(Data);
for i=1:m1
    for j=1:n1
        for k=1:p1
            if(isnan(Data(i,j,k)))
                Data(i,j,k)=0;
            end
        end
    end
end
[m1,n1,NoOfFrames]=size(Data);

%%
% input
 
 alpha=0.5; % caxis scaling
 % real area

 lxmax=0.5/2; % dlugosc obszaru
 lymax=0.5/2; % szerokosc obszaru
 lxmin=-0.5/2;
 lymin=-0.5/2;
 % Define the resolution of the grid:
 N=max([m1,n1]); % # no of grid points for R coordinate
 if(mod(N,2)) N=N-1; end;

 %%
 disp('Preliminary calculation...');
 % Polar data allocation: angle, radius, time
 Data_polar=zeros(4*N,N+1,NoOfFrames);
 %%
 [XI,YI] = meshgrid(linspace(lxmin,lxmax,n1),linspace(lymin,lymax,m1)); % due to columnwise plotting n1 is for x coordinates and m1 is for y coordinates
%% or alternatively:
%  XI=zeros(m1,n1);
%  YI=zeros(m1,n1);
%  xi=linspace(lxmin,lxmax,n1);
%  yi=linspace(lymin,lymax,m1);
%  for i=1:m1 % y coordinate
%      for j=1:n1 % x coordinate
%          XI(i,j)=xi(j);
%          YI(i,j)=yi(i);
%      end
%  end

%%
 X=reshape(XI,[],1);
 Y=reshape(YI,[],1);

%  
%  
%  minx=min(X);
%  miny=min(Y);
%  maxx=max(X);
%  maxy=max(Y);
%  X=lxmin+(X-minx)*(lxmax-lxmin)/(maxx-minx); % skalowanie przedzialu
%  Y=lymin+(Y-miny)*(lymax-lymin)/(maxy-miny); % skalowanie przedzialu

%lBV=-N:N-1;
lBV=-N:N;
mBV=-N/2:N/2-1;
xBV=zeros(length(lBV),length(mBV));
yBV=zeros(length(lBV),length(mBV));
for i=1:length(lBV)
    for j=1:length(mBV)
        yBV(i,j)=lBV(i)/N;
        xBV(i,j)=yBV(i,j).*2*mBV(j)/N;
    end
end
%lBH=-N:N-1;
lBH=-N:N;
mBH=-N/2+1:N/2;
xBH=zeros(length(lBH),length(mBH));
yBH=zeros(length(lBH),length(mBH));
for i=1:length(lBH)
    for j=1:length(mBH)
        xBH(i,j)=lBH(i)/N;
        yBH(i,j)=xBH(i,j).*2*mBH(j)/N;
    end
end

 xBH=xBH*(lxmax-lxmin)/2;
 xBV=xBV*(lxmax-lxmin)/2;

yBH=yBH*(lymax);
yBV=yBV*(lymax);
% figure;
% plot(xBH,yBH,'r.');
% hold on;
% plot(xBV,yBV,'b.');

 %%
 % convert Data from Cartesian to polar coordinates
 %%

TH=zeros(4*length(mBH),N+1);
R=zeros(4*length(mBH),N+1);
% adjust theta to interval [-180:180]
I=[4*N-N/2+2:4*N,1:4*N-N/2+1];
c=0;
for k=1:length(mBH)
    c=c+1;
    th=atan2(yBH(1,k),xBH(1,k))*180/pi;
    TH(I(c),:)=zeros(1,N+1)+th;
    R(I(c),:)=sqrt(xBH(N+1:-1:1,k).^2+yBH(N+1:-1:1,k).^2);
end
for k=length(mBV):-1:1
    c=c+1;
    th=atan2(yBV(1,k),xBV(1,k))*180/pi;
    TH(I(c),:)=zeros(1,N+1)+th;
    R(I(c),:)=sqrt(xBV(N+1:-1:1,k).^2+yBV(N+1:-1:1,k).^2);
end
for k=1:length(mBH)
    c=c+1;
    th=atan2(yBH(2*N+1,k),xBH(2*N+1,k))*180/pi;
    TH(I(c),:)=zeros(1,N+1)+th;
    R(I(c),:)=sqrt(xBH(N+1:2*N+1,k).^2+yBH(N+1:2*N+1,k).^2);
end
for k=length(mBV):-1:1
    c=c+1;
    th=atan2(yBV(2*N+1,k),xBV(2*N+1,k))*180/pi;
    TH(I(c),:)=zeros(1,N+1)+th;
    R(I(c),:)=sqrt(xBV(N+1:2*N+1,k).^2+yBV(N+1:2*N+1,k).^2);
end
%%
 disp('Data conversion...');
% loop through time frames
for frame=1:NoOfFrames
    [frame NoOfFrames]
    ZI=Data(:,:,frame);
    Z=reshape(ZI,[],1);
    %F = TriScatteredInterp(rho,theta,Z,'linear');
    F = TriScatteredInterp(X,Y,Z,'linear');
    %Evaluate the interpolant at the locations (xBH, yBH).
    %The corresponding value at these locations is ZH:
    ZBH = F(xBH,yBH);
    ZBV = F(xBV,yBV);
    Zpolar=zeros(4*length(mBH),N+1); % Zpolar(theta,radius);
    c=0;
    for k=1:length(mBH)
        c=c+1;
        Zpolar(I(c),:)=ZBH(N+1:-1:1,k);
    end
    for k=length(mBV):-1:1
        c=c+1;
        Zpolar(I(c),:)=ZBV(N+1:-1:1,k);
    end
    for k=1:length(mBH)
        c=c+1;
        Zpolar(I(c),1:N)=ZBH(N+1:2*N,k);
    end
    for k=length(mBV):-1:1
        c=c+1;
        Zpolar(I(c),1:N)=ZBV(N+1:2*N,k);
    end
    % store data
    Data_polar(:,:,frame)=Zpolar;
end

%% check NAN
[m1,n1,p1]=size(Data_polar);
for i=1:m1
    for j=1:n1
        for k=1:p1
            if(isnan(Data_polar(i,j,k)))
                Data_polar(i,j,k)=0;
            end
        end
    end
end