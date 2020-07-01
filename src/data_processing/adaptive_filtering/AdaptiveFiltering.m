function [RMSF,ERMSF,WRMSF] = AdaptiveFiltering(Data,time,WL,mask_thr,PLT)
%WL - e.g. [0.5 0.5] Width and Lenght in meters
%mask_thr - percentage of points removed by filter mask, should be in range 0.5 - 5 
%PLT - if PLT = 0 do no create plots; 0<PLT<=0.5 only ERMSF plot ;0.5<PLT<=1 - RMS plots;  1<PLT - all plots

%% processing parameters
Nmed = 1;          % median filtering window size
wgr = 1;           % Weighed RMS weight 
blur = 1;          % bluring filter mask (0 0.5 1 2) 
Vh = 1000;         % group velocity 
Vv = 1000;
HW = 0.05;         % 2D Hanning window size %the bigger the value the more points around the egdes will be removed

%% Initialization
[rows cols samples] = size(Data);
if HW > 0 
    [H2D] = Hann2D(size(Data,2),size(Data,1),HW);  % Hanning 2D to remove high edge values
    H2D = H2D/3*2+1/3;
else
    H2D = ones(size(Data,2),size(Data,1));         
end

Cmap = jet(256); 

whitethreshold = .05;
blackthreshold = .05;
CmapB = 1-([blackthreshold:1/255:1-whitethreshold ; blackthreshold:1/255:1-whitethreshold ; blackthreshold:1/255:1-whitethreshold]');

%% Unit vectors
X = 0:WL(2)/(cols-1):WL(2);
Y = 0:WL(1)/(rows-1):WL(1);

deltaR = WL(1)/(rows-1);
deltaC = WL(2)/(cols-1);

KX1 = (0:cols-1)/(cols)/deltaC;  
KX = (KX1-1/2*KX1(end));               % wavenumber centering [1/m]
KY1 = (0:rows-1)/(rows)/deltaR;  
KY = (KY1-1/2*KY1(end));               % wavenumber centering [1/m]
clear KX1 KY1 deltaT deltaC

%% Median filtering in space domain
if Nmed > 1      
    for k = 1:samples
         Data(:,:,k) = medfilt2(Data(:,:,k),[Nmed Nmed],'symmetric');  %#ok<SAGROW>
    end
end

%% Attenuation and geometrical spread amplitude decrease compensation
end_thrs = 20;                             % level of energy (percent) below processing stops
E = zeros(samples,1);
P = zeros(samples,1);
EData = zeros(rows,cols,samples);

for k = 1:samples
    E(k) = sqrt(sum(sum(abs(Data(:,:,k).^2))));   % signal energy
    P(k) = pi*time(k)*(3/2*(Vh+Vv)-sqrt(Vv*Vh));  % wavefront circumference
end

[maxx ~] = max(E);
strt = 1;

while strt < samples
    if E(strt) > 0.92*maxx
        break
    else
     strt = strt + 1;
    end
end

E2 = E;
E2(1:strt) = maxx;

for k = 1:samples
    EData(:,:,k) = Data(:,:,k)/E2(k)*sqrt(P(k));   % compensated data 
end

endd = strt;
while endd < samples
    if E2(endd) < end_thrs/100*maxx
        break
    else
     endd = endd + 1;
    end
end
disp(['start end frames:   ',num2str(strt),'  ',num2str(endd)])

if PLT > 1
    figure('Name','Energy')
    plot(time,E/max(E2),'black','LineWidth',2); hold on
    plot(time,E2/max(E2),'LineWidth',2,'Color',[0.5 .5 .5],'LineStyle','- -'); hold on
    line([time(strt) time(strt)], [0 1],'LineWidth',2,'LineStyle','--','Color',[0.2 .2 1]);
    line([time(endd) time(endd)], [0 E2(endd)/maxx],'LineWidth',2,'LineStyle','--','Color','red' )

    xlabel('Time [s]','FontSize',16,'FontName','Cambria'); 
    ylabel('E/E_m_a_x','FontSize',16,'FontName','Arial');
    ylim([0 1.05]); xlim([time(1),time(end)]);
    legend('E/E_m_a_x','E_r','n_a','n_b');
end

%% 2D FFT
FFT_Data = zeros(rows,cols,samples);
for k = 1:samples
     FFT_Data(:,:,k) = fftshift(fftn(EData(:,:,k)));  % data for filter determination       
end

%% Filter Mask
AvgFrames = zeros(rows,cols);
for k = strt:endd
    AvgFrames = AvgFrames + abs(FFT_Data(:,:,k));   
end

FilterMask = ones(rows,cols); 

MinX = min(min(abs(AvgFrames)));
MaxX = max(max(abs(AvgFrames)));
CDF_x = MinX:(MaxX-MinX)/(1000-1):MaxX;
A = reshape(AvgFrames,cols*rows,1);
n_elements = histc(abs(A),CDF_x);
c_elements = cumsum(n_elements);
CDF = c_elements/cols/rows;

mask_thr_value = (100-mask_thr)/100*cols*rows;
n = 1;
while c_elements(n) < mask_thr_value
    n = n + 1;
end
threshold = CDF_x(n);

if PLT > 1
    figure('Name','CFD')
    plot(CDF_x,CDF,'black','LineWidth',2);
    axis tight
    xlabel('Magnitude')
    ylabel('Cumulative Distribution Function')
    ylim([0.5 1.05])
    xlim([-max(CDF_x)*0.02 max(CDF_x)]);
    hold on
    line([threshold threshold], [0 (100-thresh)/100],'LineWidth',2,'LineStyle','--','Color',[0.2 .2 1]);
    line([-max(CDF_x)*0.01 threshold], [(100-thresh)/100 (100-thresh)/100],'LineWidth',2,'LineStyle','--','Color',[0.2 .2 1])
end

FilterMask = double(AvgFrames <= threshold);

% Filter mask gaussian blurring
switch blur
    case 0.5
        [H] = Gauss(5,1);        
        FilterMask = filter2(H,FilterMask);
        FilterMask(1:2,:) = 1;
        FilterMask(end-1:end,:) = 1; 
        FilterMask(:,1:2) = 1;
        FilterMask(:,end-1:end) = 1;        
    case 1
        [H] = Gauss(10,2);   
        FilterMask = filter2(H,FilterMask);
        FilterMask(1:4,:) = 1;
        FilterMask(end-3:end,:) = 1; 
        FilterMask(:,1:4) = 1;
        FilterMask(:,end-3:end) = 1;        
    case 2
        [H] = Gauss(20,4);        
        FilterMask = filter2(H,FilterMask);
        FilterMask(1:9,:) = 1;
        FilterMask(end-8:end,:) = 1; 
        FilterMask(:,1:9) = 1;
        FilterMask(:,end-8:end) = 1;               
end

if PLT > 0.5
    figure('Name','Filter Mask')
    imagesc(KX,KY,FilterMask)
    colormap(1-CmapB)
    xlabel('k_x [rad/m]')
    ylabel('k_y [rad/m]')
    Percentage = (cols*rows-sum(sum(FilterMask)))/cols/rows*100;  %#ok<NASGU>
end

%% Filtering wavefield images
Filtered_FFTData = zeros(rows,cols,samples);
for k = 1:samples
     Filtered_FFTData(:,:,k) = FFT_Data(:,:,k).*FilterMask;
end

%% Inverse 2D FFT
Filtered_Data = zeros(rows,cols,samples);
for k = 1:samples
     Filtered_Data(:,:,k) = real(ifftn(ifftshift(Filtered_FFTData(:,:,k)))); %sprawdziæ dlaczego zostaje czêœæ urojona  
end


%% RMS
RMS = abs(sqrt(sum(Data(:,:,strt:endd).^2,3)))/samples;

if PLT > 0.5
    figure('Name','RMS')
    imagesc(X,Y,RMS);
    xlabel('x (m)')
    ylabel('y (m)')
    colormap(Cmap)
end

%% WRMS
WRMS = zeros(rows,cols);

for k = strt:endd
    WRMS = WRMS + (Data(:,:,k)*k^wgr).^2;
end
WRMS = sqrt(WRMS)/(endd-strt);

if PLT > 0.5
    figure('Name','WRMS')
    imagesc(X,Y,WRMS);
    xlabel('x (m)')
    ylabel('y (m)')
    colormap(Cmap)
end

%% ERMS
ERMS = abs(sqrt(sum(EData(:,:,strt:endd).^2,3)))/samples;

if PLT > 0.5
    figure('Name','ERMS')
    imagesc(X,Y,ERMS);
    xlabel('x (m)')
    ylabel('y (m)')
end

%% RMSF
RMSF = abs(sqrt(sum(Filtered_Data(:,:,strt:endd).^2,3))).*H2D/(endd-strt);
if PLT > 0.5
    figure('Name','RMSF')
    imagesc(X,Y,RMSF);
    xlabel('x (m)')
    ylabel('y (m)')
    colormap(Cmap)
end

%% WRMSF
WRMSF = zeros(rows,cols);
for k = strt:endd
    WRMSF = WRMSF + (Filtered_Data(:,:,k)*(k^wgr)).^2;
end
WRMSF = sqrt(WRMSF).*H2D/samples;

if PLT > 0.5
    figure('Name','WRMSF')
    imagesc(X,Y,WRMSF);
    xlabel('x (m)')
    ylabel('y (m)')
    colormap(Cmap)    
end

%% ERMSF
for k = 1:samples
    EDataF(:,:,k) = Filtered_Data(:,:,k)/E2(k)*sqrt(P(k));   % compensated filtered data 
end

ERMSF = abs(sqrt(sum(EDataF(:,:,strt:endd).^2,3))).*H2D/(endd-strt);

if PLT > 0
    figure('Name','ERMSF')
    imagesc(X,Y,ERMSF);
    xlabel('x (m)')
    ylabel('y (m)')
    colormap(Cmap)
    set(gca,'YDir','normal') ;
end

end


function [h] = Gauss(hsize,sigma)

siz   = (hsize-1)/2;
     
[x,y] = meshgrid(-siz(1):siz(1),-siz(1):siz(1));
arg   = -(x.*x + y.*y)/(2*sigma*sigma);

h     = exp(arg);
h(h<eps*max(h(:))) = 0;

sumh = sum(h(:));
if sumh ~= 0,
       h  = h/sumh;
end;
end

function [H2D] = Hann2D(cols,rows,mrg)

n1 = floor(rows*mrg);
n2 = floor(cols*mrg);
H = sym_hanning(n1);
W = sym_hanning(n2); 

A = [H(1:floor(size(H,1)/2))' ones(rows-size(H,1),1)' H(floor(size(H,1)/2)+1:end)']';
B = [W(1:floor(size(W,1)/2))' ones(cols-size(W,1),1)' W(floor(size(W,1)/2)+1:end)'];
H2D = A*B;
end

function w = sym_hanning(n)
if ~rem(n,2)
   % Even length window
   half = n/2;
   w = calc_hanning(half,n);
   w = [w; w(end:-1:1)];
else
   % Odd length window
   half = (n+1)/2;
   w = calc_hanning(half,n);
   w = [w; w(end-1:-1:1)];
end
end

function w = calc_hanning(m,n)
w = .5*(1 - cos(2*pi*(1:m)'/(n+1))); 
end

function [threshold] = mythresholding(Data,thrs)

[rows cols ~] = size(Data);
AData = sum(Data,3);
MinX = min(min((AData)));
MaxX = max(max((AData)));
x = MinX:(MaxX-MinX)/(1000-1):MaxX;
A = reshape(AData,cols*rows,1);
n_elements = histc(abs(A),x);
c_elements = cumsum(n_elements);
CDF = c_elements/cols/rows;
CDF_x = x;
    
thrs_value = (100-thrs)/100*cols*rows;
n = 1;
while c_elements(n) < thrs_value
    n = n + 1;
end
threshold = x(n);
end