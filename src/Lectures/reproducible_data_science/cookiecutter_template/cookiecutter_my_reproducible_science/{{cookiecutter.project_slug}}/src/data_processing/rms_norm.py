# script for computing normalized root mean square (rms)
import numpy as np
from pathlib import Path
from scipy.io import savemat
from scipy.io import loadmat
import os

# extract name of the running script (Python versions 3.4+)
name = Path(__file__).stem

# load data
filename = '50kHz_pzt'
specimen_name = 'specimen_1'

D = loadmat('../../data/raw/' + specimen_name + '/' + filename + '.mat')  # Data, time, L
Data = np.asarray(D['Data'])
L = D['L']
# process data - calculate root mean square value
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Data_rms = np.sqrt(np.mean(Data**2, axis=1))
Data_rms_norm = Data_rms / Data_rms.max()
Data_dict = {'Data_rms_norm': Data_rms_norm, 'L': L}
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# check if directory exist; if not, create it
data_output_path = '../../data/processed/' + specimen_name + '/' + name + '/'
isExist = os.path.exists(data_output_path)
if not isExist:
    os.makedirs(data_output_path)

# save processed data
out_filename =  filename + '_' + name
savemat(data_output_path + out_filename + '.mat', Data_dict)
