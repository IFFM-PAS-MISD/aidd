function newfun(fcnname)
% NEWFUN   creates new M-file with function header template
%   NEWFCN creates a M-File having the entered filename and a specific
%   structure which helps for creating the main function structure. It
%   would be opened and the starting line for writting will be highlighted
%   (Only available in R14 or higher). The actual working MATLAB Version
%   will be also captured. If user forgot to enter code and execute the
%   function, he will get a reminder to enter code in the function.
% 
% Syntax: newfun(fcnname) 
% 
% Inputs: 
%    fcnname - string, filename
% 
% Outputs: none
% 
% Example: 
%    newfun('my_function_name') 
%    newfun my_function_name
% 
% Other m-files required: none 
% Subfunctions: none 
% MAT-files required: none 
% See also: NEWFUN_RENAME  

% Author: Pawel Kudela, D.Sc., Ph.D., Eng. 
% Institute of Fluid Flow Machinery Polish Academy of Sciences 
% Mechanics of Intelligent Structures Department 
% email address: pk@imp.gda.pl 
% Website: https://www.imp.gda.pl/en/research-centres/o4/o4z1/people/ 
%
% Inspired by newfcn at MATLABcentral by Frank González-Morphy
% frank.gonzalez-morphy@mathworks.de
% Inspired by template_header at MATLABcentral by Denis Gilbert
% gilbertd@dfo-mpo.gc.ca

%---------------------- BEGIN CODE---------------------- 

if nargin == 0, help(mfilename); return; end
if nargin > 1, error('  MSG: Only one Parameter accepted!'); end


ex = exist(fcnname);  % does M-Function already exist ? Loop statement
while ex == 2         % rechecking existence
    overwrite = 0;    % Creation decision
    msg = sprintf(['Sorry, but Function -< %s.m >- does already exist!\n', ...
        'Do you wish to Overwrite it ?'], fcnname);
    % Action Question: Text, Title, Buttons and last one is the Default
    action = questdlg(msg, ' Overwrite Function?', 'Yes', 'No','No');
    if strcmp(action,'Yes') == 1
        ex = 0; % go out of While Loop, set breaking loop statement
    else
        % Dialog for new Functionname
        fcnname = char(inputdlg('Enter new Function Name ... ', 'NEWFCN - New Name'));
        if isempty(fcnname) == 1  % {} = Cancel Button => "1"
            disp('   MSG: User decided to Cancel !')
            return
        else
            ex = exist(fcnname);  % does new functionname exist ?
        end
    end
end

overwrite = 1;

if overwrite == 1
    CreationMsg = CreateFcn(fcnname);   % Call of Sub-Function
    disp(['   MSG: <' fcnname '.m> ' CreationMsg])
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%   CREATEFCN   %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function s = CreateFcn(name)
% Sub-Function will write the M-File, open it and mark the starting write line

ext = '.m';  % Default extension for a FUNCTION !!
filename = [name ext];

fid = fopen(filename,'w');

line_1 = ['function [output1,output2] = ',name,'(input1,input2,input3)']; % Function Header

str_tmp1 = upper(name);
h1 = ['% ', str_tmp1, '   One line description of what the function or script performs (H1 line)'];    % HELP-Line's will be preset
h2 = '%    optional: more details about the function than in the H1 line';
syntax = ['% Syntax: [output1,output2] = ',name,'(input1,input2,input3)'];
h0 = '%';
inputs = '% Inputs:';
input1_descr = '%    input1 - Description, string, dimensions [m, n], Units: ms';
input2_descr = '%    input2 - Description, logical, dimensions [m, n], Units: m';
input3_descr = '%    input3 - Description, double, dimensions [m, n], Units: N';
outputs = '% Outputs:';
output1_descr = '%    output1 - Description, integer, dimensions [m, n], Units: -';
output2_descr = '%    output2 - Description, double, dimensions [m, n], Units: m/s^2';
example = '% Example:';
example1 = ['%    [output1,output2] = ',name,'(input1,input2,input3)'];
example2 = ['%    [output1,output2] = ',name,'(input1,input2)'];
example3 = ['%    [output1] = ',name,'(input1,input2,input3)'];
other = '% Other m-files required: none';
subfunc = '% Subfunctions: none';
matreq = '% MAT-files required: none';
see_also = '% See also: OTHER_FUNCTION_NAME1,  OTHER_FUNCTION_NAME2';

fprintf(fid,'%s\n', line_1);      % Write header to file
fprintf(fid,'%s \n', h1);         %   
fprintf(fid,'%s \n', h2);         %   
fprintf(fid,'%s \n', h2);         %   
fprintf(fid,'%s \n', h2);  
fprintf(fid,'%s \n', h0);
fprintf(fid,'%s \n', syntax); 
fprintf(fid,'%s \n', h0); 
fprintf(fid,'%s \n', inputs); 
fprintf(fid,'%s \n', input1_descr); 
fprintf(fid,'%s \n', input2_descr); 
fprintf(fid,'%s \n', input3_descr); 
fprintf(fid,'%s \n', h0);
fprintf(fid,'%s \n', outputs);
fprintf(fid,'%s \n', output1_descr);
fprintf(fid,'%s \n', output2_descr);
fprintf(fid,'%s \n', h0);
fprintf(fid,'%s \n', example);
fprintf(fid,'%s \n', example1);
fprintf(fid,'%s \n', example2);
fprintf(fid,'%s \n', example3);
fprintf(fid,'%s \n', h0);
fprintf(fid,'%s \n', other);
fprintf(fid,'%s \n', subfunc);
fprintf(fid,'%s \n', matreq);
fprintf(fid,'%s \n', see_also);
fprintf(fid,'%s \n\n', h0);       %   

%% Personalise template
% Writer settings will be consructed ...
author = '% Author: Pawel Kudela, D.Sc., Ph.D., Eng.';
institute_line1 = '% Institute of Fluid Flow Machinery Polish Academy of Sciences';
institute_line2 = '% Mechanics of Intelligent Structures Department';
email = '% email address: pk@imp.gda.pl';
website = '% Website: https://www.imp.gda.pl/en/research-centres/o4/o4z1/people/';
% dt = datestr(now);                date = ['%%        $DATE: ', dt, ' $'];
% rev = ['%        $Revision: 1.00 $'];
% devel = ['% Matlab version: ',version];
%%
filenamesaved = filename;         fns = ['% FILENAME  : ', filenamesaved];

% Personalised template will be write in File ...
fprintf(fid,'%s \n', author);
fprintf(fid,'%s \n', institute_line1);
fprintf(fid,'%s \n', institute_line2);
fprintf(fid,'%s \n', email);
fprintf(fid,'%s \n\n', website);
%fprintf(fid,'%s \n\n', fns);

begin_code = '%---------------------- BEGIN CODE----------------------';
fprintf(fid,'%s \n\n', begin_code);
% Reminder that user must enter code in created File / Function
lst = 'disp('' !!!  You must enter code into this file <';
lst_3 = '> !!!'')';
fprintf(fid,'%s %s.m %s \n\n', lst, name, lst_3);

end_code =  '%---------------------- END OF CODE----------------------';
fprintf(fid,'%s \n\n', end_code);
%fprintf(fid,'%s \n', devel);
%fprintf(fid,'%s \n', date);
%fprintf(fid,'%s \n', rev);
% Before last line, from where functionality does come
% originl1 = '% Created with NEWFCN.m by Frank González-Morphy ';
% originl2 = '% Contact...: frank.gonzalez-morphy@mathworks.de ';
% fprintf(fid,'\n\n\n\n\n\n\n\n%s \n', originl1);
% fprintf(fid,'%s \n', originl2);

% Last Line in the Fcn
end_of_file = ['% ================ [', filenamesaved, '] ================ '];
                    
fprintf(fid,'%s \n', end_of_file);
    
% Close the written File
st = fclose(fid);

if st == 0  % "0" for successful
    % Open the written File in the MATLAB Editor/Debugger
    v = version;
    if v(1) == '7'                 % R14 Version
        opentoline(filename, 12);  % Open File and highlight the start Line
    else
        % ... for another versions of MATLAB
        edit(filename);
    end
    s = 'successfully done !!';
else
    s = ' ERROR: Problems encounter while closing File!';
end

%---------------------- END OF CODE---------------------- 

% ================ [newfun.m] ================  
