function [x,fs] = load_sound(name, n0, options)

% load_sound - load a sound from a file.
%
%   [x,fs] = load_sound(name, n0, options);
%
%   Load from a .wav (windows), .au (mac) or .asc (Wavelab)
%   sound file.
%
%   Some cool built in examples include
%   spectrogram.wav, tiger.au, bell.wav
%
% Copyright (c) 2006 Gabriel Peyr�

options.null = [];
if nargin<2
    n0 = [];
end

name = lower(name);

% find extension 
I = strfind(name, '.');
if isempty(I)
    ext = [];
    extlist = {'wav' 'asc' 'au'};
    for i=1:length(extlist)
        if exist([name '.' extlist{i}])==2
            ext = extlist{i};
            break;
        end
    end
    if isempty(ext)
        error('Unable to determine extension');
    end
else
    I = I(end);
    ext = name(I+1:end);
    name = name(1:I-1);
end



options.null = 0;

rep = getoptions(options, 'rep', '');


fs = 22000/2; % sampling rate
switch ext
    case 'asc'
        fid = fopen([rep name], 'r');
        if fid<0
            error(['Unknown file ' name '.' ext]);
        end
        x = fscanf(fid,'%g');
        fclose(fid);
    case 'wav'
        % MS files
        [x,fs,nbits] = wavread([rep name '.' ext]);
    case 'au'
        % Sun files
        [x,fs,nbits] = auread([rep name '.' ext]);
        
    otherwise
        x = load_signal(name,n);
end

if size(x,1)<size(x,2)
    x = x';
end
if size(x,2)>1
    % mono signal
    x = x(:,1);
end
x = x(:);

% specific cropping to remov blank

if strcmp(name, 'bird')
%    sel = [1:9400 6500:9000 16500:18700 26000:28500];
    sel = [1:6000 12500:15000 22500:24000 32500:3400];
    x(sel) = [];
end
if strcmp(name, 'spectrogram')
    x = x(5800:end);
end
if strcmp(name, 'aeio')
    x = x(15000:end);
end
if strcmp(name, 'acha')
    x = x(14000:end);
end
if strcmp(name,'aherohasfallen')
    x = x(5000:2^14+5000);
end
if strcmp(name,'sentence')
    x = x(3300:end);
end
% specific cropping
if strcmp(name, 'tiger')
    x = x(2000:end);
end
if strcmp(name, 'drums')
    x = x(4800:12500);
end

n = size(x,1);
subsampling = getoptions(options, 'subsampling', 1);
if subsampling~=1    
    % sub-sampling 1/10
    fs = fs*subsampling;
    t = linspace(0,1,n);
    ti = linspace(0,1,round(n*subsampling));
    x = interp1( t,x,ti );     
    x = x(:);
end

if not(isempty(n0)) && n0<n
    x = x(1:n0);
end

% rescale to [-1,1]
x = x/max(abs(x));