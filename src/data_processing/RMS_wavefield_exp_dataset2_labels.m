clear all;close all;   warning off;clc;

load project_paths projectroot src_path;
%% Prepare output directories
% allow overwriting existing results if true
%overwrite=false;
overwrite=true;
%test_case=[1:12,15:24]; % select file numbers for processing
test_case=[25:27]; % select file numbers for processing
%% Processing parameters
Nx = 500;   % number of points after interpolation in X direction
Ny = 500;   % number of points after interpolation in Y direction
N=Nx;
%%
% create path to the experimental raw data folder
raw_data_path = fullfile( projectroot, 'data','raw','exp', filesep );

% create path to the experimental interim data folder
interim_data_path = fullfile( projectroot, 'data','interim','exp', filesep );

% full field measurements
list = {'GFRP_nr6_50kHz_5HC_8Vpp_x20_10avg_110889', ...          % 1  Length = ?;Width = ?;           
        'GFRP_nr6_100kHz_5HC_8Vpp_x20_10avg_110889', ... % 2
        'GFRP_nr_6_333x333p_5HC_150kHz_20vpp_x10', ... % 3
        'GFRP_nr6_200kHz_5HC_20vpp_x20xx17avg_110889', ... % 4
        'GFRP_nr1_333x333p_5HC_200kHz_20vpp_x20', ... % 5
        'GFRP_nr_1_333x333p_5HC_150kHz_20vpp_x10', ... % 6
        'GFRP_nr1_333x333p_5HC_100kHz_20vpp_x10', ... % 7
        'GFRP_nr4_50kHz_5HC_8Vpp_x20_10avg_110889', ... % 8
        'GFRP_nr4_100kHz_5HC_8Vpp_x20_10avg_110889', ... % 9
        'GFRP_nr4_150kHz_5HC_20Vpp_x20_5avg_110889', ... % 10
        'GFRP_nr4_200kHz_5HC_20Vpp_x20_12avg_110889', ... % 11
        'CFRP_teflon_impact_375x375p_5HC_100kHz__6Vpp_x10', ... %12
        'sf_30p_5mm_45mm_251x251p_50kHz_5HC_x3_15Vpp_norm', ... %13
        'Alu_2_54289p_16,5kHz_5T_x50_moneta2', ... %14
        'Alu_2_138383p_100kHz_5T_x50_moneta2', ... %15
        'Alu_2_77841p_35kHz_5T_x30_moneta', ... %16
        '93011_7uszk', ... %17
        '93011_2uszk', ... % 18
        'CFRPprepreg_41615p_teflon2cm_3mm_100kHz_20avg_15vpp_prostokatne', ... %19
        'CFRP_50kHz_10Vpp_x10_53261p_strona_oklejona_plastelina_naciecia_prostokatne_256_256', ... %20
        'CFRP_100kHz_20Vpp_x10_53261p_strona_oklejona_plastelina_naciecia_prostokatne_256_256', ... %21
        'CFRP3_5_teflon10x10mm_50kHz_47085p_20Vppx20_10avg_prostokatne', ... %22
        'CFRP3_5_teflon10x10mm_100kHz_47085p_20Vppx20_20avg_prostokatne', ... %23
        'CFRP3_5_teflon15x15mm_50kHz_47085p_20Vppx20_10avg_prostokatne',... %24
         'CFRP_teflon_3_375_375p_50kHz_5HC_x3_15Vpp',...%25
         'CFRP_teflon_3c_375_375p_50kHz_5HC_x3_15Vpp',...%26 quarter only
         'CFRP_teflon_3o_375_375p_50kHz_5HC_x12_15Vpp'};%27                  
     
% manual characterization of defects
% 'GFRP_nr6_50kHz_5HC_8Vpp_x20_10avg_110889', ...1
xCenter=[122,375,248,248];
yCenter=[250,250,124,375];
a=[30/2,20,30,20/2];
b=[20/2,20,20,20/2];
rotAngle=[0,0,0,0];
label_data(1).xCenter=xCenter;
label_data(1).yCenter=yCenter;
label_data(1).a=a;
label_data(1).b=b;
label_data(1).rotAngle=rotAngle;
label_data(1).type=["ellipse","rectangle","rectangle","ellipse"];
% 'GFRP_nr6_100kHz_5HC_8Vpp_x20_10avg_110889', ...2
xCenter=[122,375,248,248];
yCenter=[250,250,124,375];
a=[30/2,20,30,20/2];
b=[20/2,20,20,20/2];
rotAngle=[0,0,0,0];
label_data(2).xCenter=xCenter;
label_data(2).yCenter=yCenter;
label_data(2).a=a;
label_data(2).b=b;
label_data(2).rotAngle=rotAngle;
label_data(2).type=["ellipse","rectangle","rectangle","ellipse"];
% 'GFRP_nr_6_333x333p_5HC_150kHz_20vpp_x10', ...3
xCenter=[122,375,248,248];
yCenter=[250,250,124,375];
a=[30/2,20,30,20/2];
b=[20/2,20,20,20/2];
rotAngle=[0,0,0,0];
label_data(3).xCenter=xCenter;
label_data(3).yCenter=yCenter;
label_data(3).a=a;
label_data(3).b=b;
label_data(3).rotAngle=rotAngle;
label_data(3).type=["ellipse","rectangle","rectangle","ellipse"];
% 'GFRP_nr6_200kHz_5HC_20vpp_x20xx17avg_110889', ...4
xCenter=[122,375,248,248];
yCenter=[250,250,124,375];
a=[30/2,20,30,20/2];
b=[20/2,20,20,20/2];
rotAngle=[0,0,0,0];
label_data(4).xCenter=xCenter;
label_data(4).yCenter=yCenter;
label_data(4).a=a;
label_data(4).b=b;
label_data(4).rotAngle=rotAngle;
label_data(4).type=["ellipse","rectangle","rectangle","ellipse"];

% 'GFRP_nr1_333x333p_5HC_200kHz_20vpp_x20', ...5
xCenter=[125,125,125,250,375,375];
yCenter=[375,250,125,125,125,375];
a=[10,10,10,10,10,10,10];
b=[10,10,10,10,10,10,10];
rotAngle=[0,0,0,0,0,0,0];
label_data(5).xCenter=xCenter;
label_data(5).yCenter=yCenter;
label_data(5).a=a;
label_data(5).b=b;
label_data(5).rotAngle=rotAngle;
label_data(5).type=["ellipse","ellipse","ellipse","ellipse","ellipse","ellipse"];

% 'GFRP_nr_1_333x333p_5HC_150kHz_20vpp_x10', ...6
xCenter=[125,125,125,250,375,375];
yCenter=[375,250,125,125,125,375];
a=[10,10,10,10,10,10,10];
b=[10,10,10,10,10,10,10];
rotAngle=[0,0,0,0,0,0,0];
label_data(6).xCenter=xCenter;
label_data(6).yCenter=yCenter;
label_data(6).a=a;
label_data(6).b=b;
label_data(6).rotAngle=rotAngle;
label_data(6).type=["ellipse","ellipse","ellipse","ellipse","ellipse","ellipse"];

% 'GFRP_nr1_333x333p_5HC_100kHz_20vpp_x10', ...7
xCenter=[125,125,125,250,375,375];
yCenter=[375,250,125,125,125,375];
a=[10,10,10,10,10,10,10];
b=[10,10,10,10,10,10,10];
rotAngle=[0,0,0,0,0,0,0];
label_data(7).xCenter=xCenter;
label_data(7).yCenter=yCenter;
label_data(7).a=a;
label_data(7).b=b;
label_data(7).rotAngle=rotAngle;
label_data(7).type=["ellipse","ellipse","ellipse","ellipse","ellipse","ellipse"];

% 'GFRP_nr4_50kHz_5HC_8Vpp_x20_10avg_110889', ...8
xCenter=[125,250,375,250];
yCenter=[252,125,250,375];
a=[10,10,10,10];
b=[10,10,10,10];
rotAngle=[0,0,0,0];
label_data(8).xCenter=xCenter;
label_data(8).yCenter=yCenter;
label_data(8).a=a;
label_data(8).b=b;
label_data(8).rotAngle=rotAngle;
label_data(8).type=["ellipse","ellipse","ellipse","ellipse"];

% 'GFRP_nr4_100kHz_5HC_8Vpp_x20_10avg_110889', ...9
xCenter=[125,250,375,250];
yCenter=[252,125,250,375];
a=[10,10,10,10];
b=[10,10,10,10];
rotAngle=[0,0,0,0];
label_data(9).xCenter=xCenter;
label_data(9).yCenter=yCenter;
label_data(9).a=a;
label_data(9).b=b;
label_data(9).rotAngle=rotAngle;
label_data(9).type=["ellipse","ellipse","ellipse","ellipse"];

% 'GFRP_nr4_150kHz_5HC_20Vpp_x20_5avg_110889', ...10
xCenter=[125,250,375,250];
yCenter=[252,125,250,375];
a=[10,10,10,10];
b=[10,10,10,10];
rotAngle=[0,0,0,0];
label_data(10).xCenter=xCenter;
label_data(10).yCenter=yCenter;
label_data(10).a=a;
label_data(10).b=b;
label_data(10).rotAngle=rotAngle;
label_data(10).type=["ellipse","ellipse","ellipse","ellipse"];

% 'GFRP_nr4_200kHz_5HC_20Vpp_x20_12avg_110889', ...11
xCenter=[125,250,375,250];
yCenter=[252,125,250,375];
a=[10,10,10,10];
b=[10,10,10,10];
rotAngle=[0,0,0,0];
label_data(11).xCenter=xCenter;
label_data(11).yCenter=yCenter;
label_data(11).a=a;
label_data(11).b=b;
label_data(11).rotAngle=rotAngle;
label_data(11).type=["ellipse","ellipse","ellipse","ellipse"];

% 'CFRP_teflon_impact_375x375p_5HC_100kHz__6Vpp_x10', ...12
xCenter=[125,258,375,375,375,250];
yCenter=[382,125,125,248,378,378];
a=[15,16/2,10/2,10/2,10/2,10/2];
b=[15,10/2,10/2,10/2,10/2,10/2];
rotAngle=[0,0,0,0,0,0];
label_data(12).xCenter=xCenter;
label_data(12).yCenter=yCenter;
label_data(12).a=a;
label_data(12).b=b;
label_data(12).rotAngle=rotAngle;
label_data(12).type=["rectangle","ellipse","ellipse","ellipse","ellipse","ellipse"];

% 'Alu_2_138383p_100kHz_5T_x50_moneta2', ...15
xCenter=[418];
yCenter=[243];
a=[10/2];
b=[10/2];
rotAngle=[0];
label_data(15).xCenter=xCenter;
label_data(15).yCenter=yCenter;
label_data(15).a=a;
label_data(15).b=b;
label_data(15).rotAngle=rotAngle;
label_data(15).type=["ellipse"];

% 'Alu_2_77841p_35kHz_5T_x30_moneta', ...16
xCenter=[110];
yCenter=[128];
a=[10/2];
b=[10/2];
rotAngle=[0];
label_data(16).xCenter=xCenter;
label_data(16).yCenter=yCenter;
label_data(16).a=a;
label_data(16).b=b;
label_data(16).rotAngle=rotAngle;
label_data(16).type=["ellipse"];

% '93011_7uszk', ...17 (visible 6)
xCenter=[87,273,422,263,410,172];
yCenter=[255,150,250,286,344,428];
a=[10/2,10/2,2/2,10/2,10/2,10/2];
b=[10/2,10/2,23/2,10/2,10/2,10/2];
rotAngle=[0,0,40,0,0,0];
label_data(17).xCenter=xCenter;
label_data(17).yCenter=yCenter;
label_data(17).a=a;
label_data(17).b=b;
label_data(17).rotAngle=rotAngle;
label_data(17).type=["ellipse","ellipse","ellipse","ellipse","ellipse","ellipse"];

% 93011_2uszk', ...18 
xCenter=[94,418];
yCenter=[245,241];
a=[20/2,10/2];
b=[20/2,10/2];
rotAngle=[0,0];
label_data(18).xCenter=xCenter;
label_data(18).yCenter=yCenter;
label_data(18).a=a;
label_data(18).b=b;
label_data(18).rotAngle=rotAngle;
label_data(18).type=["ellipse","ellipse"];

% 'CFRPprepreg_41615p_teflon2cm_3mm_100kHz_20avg_15vpp_prostokatne', ...19
xCenter=[94,418];
yCenter=[245,241];
a=[20,10];
b=[20,10];
rotAngle=[0,0];
label_data(19).xCenter=xCenter;
label_data(19).yCenter=yCenter;
label_data(19).a=a;
label_data(19).b=b;
label_data(19).rotAngle=rotAngle;
label_data(19).type=["rectangle","rectangle"];

% 'CFRP_50kHz_10Vpp_x10_53261p_strona_oklejona_plastelina_naciecia_prostokatne_256_256', ...20
xCenter=[94,418];
yCenter=[245,241];
a=[20,10];
b=[20,10];
rotAngle=[0,0];
label_data(20).xCenter=xCenter;
label_data(20).yCenter=yCenter;
label_data(20).a=a;
label_data(20).b=b;
label_data(20).rotAngle=rotAngle;
label_data(20).type=["rectangle","rectangle"];

% 'CFRP_100kHz_20Vpp_x10_53261p_strona_oklejona_plastelina_naciecia_prostokatne_256_256', ...21
xCenter=[94,418];
yCenter=[245,241];
a=[20,10];
b=[20,10];
rotAngle=[0,0];
label_data(21).xCenter=xCenter;
label_data(21).yCenter=yCenter;
label_data(21).a=a;
label_data(21).b=b;
label_data(21).rotAngle=rotAngle;
label_data(21).type=["rectangle","rectangle"];

% 'CFRP3_5_teflon10x10mm_50kHz_47085p_20Vppx20_10avg_prostokatne', ...22
xCenter=[94,418];
yCenter=[245,241];
a=[20,10];
b=[20,10];
rotAngle=[0,0];
label_data(22).xCenter=xCenter;
label_data(22).yCenter=yCenter;
label_data(22).a=a;
label_data(22).b=b;
label_data(22).rotAngle=rotAngle;
label_data(22).type=["rectangle","rectangle"];

% 'CFRP3_5_teflon10x10mm_100kHz_47085p_20Vppx20_20avg_prostokatne', ...23
xCenter=[94,418];
yCenter=[245,241];
a=[20,10];
b=[20,10];
rotAngle=[0,0];
label_data(23).xCenter=xCenter;
label_data(23).yCenter=yCenter;
label_data(23).a=a;
label_data(23).b=b;
label_data(23).rotAngle=rotAngle;
label_data(23).type=["rectangle","rectangle"];

% 'CFRP3_5_teflon15x15mm_50kHz_47085p_20Vppx20_10avg_prostokatne', ...24
xCenter=[94,418];
yCenter=[245,241];
a=[20,10];
b=[20,10];
rotAngle=[0,0];
label_data(24).xCenter=xCenter;
label_data(24).yCenter=yCenter;
label_data(24).a=a;
label_data(24).b=b;
label_data(24).rotAngle=rotAngle;
label_data(24).type=["rectangle","rectangle"];

% 'CFRP_teflon_3_375_375p_50kHz_5HC_x3_15Vpp', ...25
xCenter=[128];
yCenter=[128];
a=[15];
b=[15];
rotAngle=[0];
label_data(25).xCenter=xCenter;
label_data(25).yCenter=yCenter;
label_data(25).a=a;
label_data(25).b=b;
label_data(25).rotAngle=rotAngle;
label_data(25).type=["rectangle"];

% 'CFRP_teflon_3c_375_375p_50kHz_5HC_x3_15Vpp', ...26
xCenter=[130];
yCenter=[131];
a=[30];
b=[30];
rotAngle=[0];
label_data(26).xCenter=xCenter;
label_data(26).yCenter=yCenter;
label_data(26).a=a;
label_data(26).b=b;
label_data(26).rotAngle=rotAngle;
label_data(26).type=["rectangle"];

% 'CFRP_teflon_3o_375_375p_50kHz_5HC_x12_15Vpp', ...27
xCenter=[369];
yCenter=[129];
a=[15];
b=[15];
rotAngle=[0];
label_data(27).xCenter=xCenter;
label_data(27).yCenter=yCenter;
label_data(27).a=a;
label_data(27).b=b;
label_data(27).rotAngle=rotAngle;
label_data(27).type=["rectangle"];


folder  = raw_data_path;
nFile   = length(test_case);
success = false(1, nFile);

for k = test_case
    filename = list{k};
    processed_filename = ['label_',filename]; % filename of processed .mat data
    % check if already exist
    if(overwrite||(~overwrite && ~exist([interim_data_path,processed_filename,'.png'], 'file')))
        try        
            %% PROCESS DATA
            fprintf('Processing:\n%s\n',filename);
            multiple_delam_image_label(N,label_data(k).xCenter,label_data(k).yCenter,label_data(k).a,label_data(k).b,label_data(k).rotAngle,label_data(k).type,[interim_data_path,processed_filename]);
            
            %% END OF PROCESSING
            [filepath,name,ext] = fileparts(filename);
            fprintf('Successfully processed:\n%s\n', name);% successfully processed
        catch
            fprintf('Failed: %s\n', filename);
        end
    else
        fprintf('Filename: \n%s \nalready exist\n', processed_filename);
    end
end



