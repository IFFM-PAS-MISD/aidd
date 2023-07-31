import os
import csv
import matplotlib.pyplot as plt
import numpy as np

os.chdir('/aijjeh_odroid_sensors/aidd/data/raw/num/wavefield_dataset2_bottom_out/1_output/')
temp_arr = np.zeros((512, 500, 500), dtype=np.float32)
for i in range(1, 513):
    temp_arr[i-1] = plt.imread('%d_flat_shell_Vz_1_500x500bottom.png' % i, 0)

fft3_arr = np.fft.fftn(temp_arr)
fft3_arr = np.fft.fftshift(fft3_arr)
fft3_arr = abs(fft3_arr)

ax = plt.axes(projection='3d')
ax.plot(fft3_arr, cmap='viridis')
plt.show()
