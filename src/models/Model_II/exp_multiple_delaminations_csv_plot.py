import matplotlib.pyplot as plt
import os
import numpy as np

# select case number [0, 1, 2, 3]
n = 0
case = {0: 'CFRP_teflon_3o',
        1: 'L3_S2_B',
        2: 'L3_S3_B',
        3: 'L3_S4_B'}

author = {0: 'Ijjeh',
          1: 'Saeed'}

case_name = {0: '__375_375p_50kHz_5HC_x12_15Vpp_%s_updated_results_.csv' % author[0],
             1: '__375_375p_50kHz_5HC_x12_15Vpp_%s_updated_results_.csv' % author[1],

             2: '_333x333p_50kHz_5HC_15Vpp_x10_pzt_%s_updated_results_.csv' % author[0],
             3: '_333x333p_50kHz_5HC_15Vpp_x10_pzt_%s_updated_results_.csv' % author[1],

             4: '_333x333p_50kHz_5HC_18Vpp_x10_pzt_%s_updated_results_.csv' % author[0],
             5: '_333x333p_50kHz_5HC_18Vpp_x10_pzt_%s_updated_results_.csv' % author[1],

             6: '_333x333p_50kHz_10HC_18Vpp_x10_pzt_%s_updated_results_.csv' % author[0],
             7: '_333x333p_50kHz_10HC_18Vpp_x10_pzt_%s_updated_results_.csv' % author[1],
             }

case_Ijjeh = case[n] + case_name[(2 * n)]
case_Saeed = case[n] + case_name[(2 * n + 1)]

os.chdir('/home/aijjeh/Documents/GitHub/aidd_new/reports/journal_papers/ConvLSTM Paper')

ijjeh = np.genfromtxt(case_Ijjeh, delimiter=',', dtype=float)
saeed = np.genfromtxt(case_Saeed, delimiter=',', dtype=float)

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 10,
        }

plt.matplotlib.rc('font', **font)

############################################################################################################
plt.figure(figsize=(7, 3.5))
plt.grid()

plt.plot(saeed[0, :] + 32, saeed[-1, :], label='Model-I: window size=64')
plt.plot(ijjeh[0, :] + 12, ijjeh[-1, :], label='Model-II: window size=24')
plt.ylabel('IoU', weight="normal", fontsize=12)
plt.xlabel('Centre of the window', weight="normal", fontsize=12)
plt.legend(loc="upper right")

os.chdir('/home/aijjeh/Documents/GitHub/aidd_new/reports/journal_papers/ConvLSTM Paper/Graphics')
# plt.savefig(case[n] + '_plot.png', dpi=600, bbox_inches='tight', transparent="True", pad_inches=0)
plt.show()
plt.close('all')
