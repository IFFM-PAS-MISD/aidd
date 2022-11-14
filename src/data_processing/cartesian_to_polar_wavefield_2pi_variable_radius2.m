function [Data_polar,number_of_points,radius_max] = cartesian_to_polar_wavefield_2pi_variable_radius2(Data,Lx,Ly,beta)
% CARTESIAN_TO_POLAR_WAVEFIELD   transform wavefield to polar coordinates 
%    Data is interpolated at given angles beta 
%    The same number of points is used for interpolation at each angle 
%    radius is variable at each angle (radius spacing is angle dependent)
% 
% Syntax: [Data_polar,number_of_points,radius_max] = cartesian_to_polar_wavefield_2pi_variable_radius(Data,Lx,Ly,beta) 
% 
% Inputs: 
%    Data - Wavefield in space domain, double, dimensions [nPointsx, nPointsy, number_of_time_steps], Units:  
%    Lx - length (in x direction), double, Units: m 
%    Ly - width (in y direction), double, Units: m 
%    beta - list of angles in range 0:360, double, Units: deg
% 
% Outputs: 
%    Data_polar - Data transformed to polar coordinates, double 
%    dimensions [number_of_angles,number_of_points,number_of_time_steps], Units: - 
%    number_of_points - integer 
%    radius_max - variable radius at each angle, double, Units: m
% 
% Example: 
%    [Data_polar,number_of_points,radius_max] = cartesian_to_polar_wavefield_2pi_variable_radius(Data,Lx,Ly,beta) 
%    [Data_polar,number_of_points,radius_max] = cartesian_to_polar_wavefield_2pi_variable_radius(Data,Lx,Ly,[0:15:360])  
% 
% Other m-files required: none 
% Subfunctions: none 
% MAT-files required: none 
% See also: 
% 

% Author: Pawel Kudela, D.Sc., Ph.D., Eng. 
% Institute of Fluid Flow Machinery Polish Academy of Sciences 
% Mechanics of Intelligent Structures Department 
% email address: pk@imp.gda.pl 
% Website: https://www.imp.gda.pl/en/research-centres/o4/o4z1/people/ 

%---------------------- BEGIN CODE---------------------- 

b = beta*pi/180;
number_of_angles = length(beta);

%% check NAN
[nPointsx,nPointsy,number_of_time_steps]=size(Data); % number_of_time_steps is equal to number_of_frequencies
% for i=1:nPointsx
%     for j=1:nPointsy
%         for k=1:number_of_time_steps
%             if(isnan(Data(i,j,k)))
%                 Data(i,j,k)=0;
%             end
%         end
%     end
% end
Data(isnan(Data))=0;
%%
% input
lxmax=Lx/2; % length
lymax=Ly/2; % width
lxmin=-Lx/2; % quarter
lymin=-Ly/2; % quarter
% Define the resolution of the grid:
%number_of_points=round(sqrt(2)*max([nPointsx,nPointsy])); % # no of grid points for R coordinate
number_of_points=round(sqrt(nPointsx^2+nPointsy^2)); % # no of grid points for R coordinate
if(mod(number_of_points,2)) % only even numbers
    number_of_points=number_of_points-1; 
end
%%
% Polar data allocation: angle, radius(wavenumbers), time(frequency)
Data_polar=zeros(number_of_angles,number_of_points,number_of_time_steps);
 %%
[XI,YI] = meshgrid(linspace(lxmin,lxmax,nPointsx),linspace(lymin,lymax,nPointsy)); % due to columnwise plotting nPointsx is for x coordinates and nPointsy is for y coordinates
%X=reshape(XI,[],1);
%Y=reshape(YI,[],1);
 %% 

x=zeros(number_of_angles,number_of_points);
y=zeros(number_of_angles,number_of_points);
radius_max=zeros(number_of_angles ,1);
for k=1:number_of_angles 
    % first qauadrant
    if(b(k)<=pi/2)
        if(b(k) <= atan(lymax/lxmax))
            radius_max(k,1)=sqrt((lxmax*tan(b(k))).^2+lxmax^2);
        else
            radius_max(k,1)=sqrt((lymax*tan(pi/2-b(k))).^2+lymax^2);
        end
    end
    % second qauadrant
    if( b(k)>pi/2 && b(k)<=pi)
        if(b(k)-pi/2 <= atan(lxmax/lymax))
            radius_max(k,1)=sqrt((lymax*tan(b(k)-pi/2)).^2+lymax^2);
            
        else
            radius_max(k,1)=sqrt((lxmax*tan(pi-b(k))).^2+lxmax^2);
        end
    end
    % third qauadrant
    if( b(k)>pi && b(k)<=3*pi/2)
        if(b(k)-pi <= atan(lymax/lxmax))
            radius_max(k,1)=sqrt((lymax*tan(b(k)-pi)).^2+lymax^2);
            
        else
            radius_max(k,1)=sqrt((lxmax*tan(3*pi/2-b(k))).^2+lxmax^2);
        end
    end
    % fourth qauadrant
    if( b(k)>3*pi/2 && b(k)<=2*pi)
        if(b(k)-3*pi/2 <= atan(lymax/lxmax))
            radius_max(k,1)=sqrt((lymax*tan(b(k)-3*pi/2)).^2+lymax^2);
            
        else
            radius_max(k,1)=sqrt((lxmax*tan(2*pi-b(k))).^2+lxmax^2);
        end
    end
    
end
%figure;polar(b',radius_max);
for k=1:number_of_angles 
    R=linspace(0,radius_max(k),number_of_points);
    x(k,:) = R*cos(b(k));
    y(k,:) = R*sin(b(k));
end
%plot(x,y,'.');
 %%
 % convert Data from Cartesian to polar coordinates
 %%

 disp('Cartesian to polar coordinates transformation and interpolation');
% loop through time (frequency) frames
frame=1;
ZI=Data(:,:,frame);
%Z=reshape(ZI,[],1);
%F = TriScatteredInterp(X,Y,Z,'linear');
%F = scatteredInterpolant(X,Y,Z,'linear');
F = griddedInterpolant(XI',YI',ZI','linear'); % requires ndgrid format
%F = griddedInterpolant(XI',YI',ZI','cubic'); % requires ndgrid format
%Evaluate the interpolant at the locations (x, y).
% store data
Data_polar(:,:,frame)=F(x,y);
for frame=2:number_of_time_steps
    [frame,number_of_time_steps]
    ZI=Data(:,:,frame);
    %Z=reshape(ZI,[],1);
    %F = TriScatteredInterp(X,Y,Z,'linear');
    F.Values = ZI';
    %Evaluate the interpolant at the locations (x, y).
    % store data
    Data_polar(:,:,frame)=F(x,y);
end

%% check NAN
%[nPointsy,nPointsx,p1]=size(Data_polar);
% for i=1:nPointsy
%     for j=1:nPointsx
%         for k=1:p1
%             if(isnan(Data_polar(i,j,k)))
%                 Data_polar(i,j,k)=0;
%             end
%         end
%     end
% end 
Data_polar(isnan(Data_polar))=0;
%---------------------- END OF CODE---------------------- 

% ================ [cartesian_to_polar_wavefield_2pi_variable_radius.m] ================  
