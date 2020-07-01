clc;clear;close all
load 'l:\Praca\4Metody\wavefield_rand\1_output\flat_shell_Vz_1_500x500top.mat';
load 'l:\Praca\4Metody\wavefield_rand\1_output\t_frames.mat';
time = t_frames(1:512);
clear t_frames;

WL = [0.5 0.5];
mask_thr = 1;
PLT = 0.5;

[RMSF,ERMSF,WRMSF] = AdaptiveFiltering(Data,time,WL,mask_thr,PLT);

threshold = 0.012;
Binary = uint8(ERMSF >= threshold);

whitethreshold = .05;
blackthreshold = .05;
CmapB = 1-([blackthreshold:1/255:1-whitethreshold ; blackthreshold:1/255:1-whitethreshold ; blackthreshold:1/255:1-whitethreshold]');
figure
imagesc(Binary)
colormap(1-CmapB)