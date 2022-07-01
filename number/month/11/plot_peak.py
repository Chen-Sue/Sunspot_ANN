
import time
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
from numpy.core.shape_base import block
import pandas as pd
import matplotlib.pyplot as plt
import julian
import datetime

import config
import utils 

start_time = time.time()
np.random.seed(config.seed)

file_location = config.file_location

initial_julian = julian.to_jd(datetime.datetime(1740,1,1,0,0,0), fmt='jd')
end_julian = julian.to_jd(datetime.datetime(2030,1,1,0,0,0), fmt='jd')

sunspot = np.loadtxt(file_location+r'\data1749_2021\SN_m_tot_V2.0.txt')
number = np.array(sunspot[:, 3], dtype=np.float32)
jul_date = np.loadtxt(file_location+r'\data1749_2021\sn_jul_date.txt').reshape(-1,1)
print(jul_date.shape)

x_jul, x_tick_time = [], [] # 设置横轴标签
for x_tick in np.arange(1870, 2020+1, config.step): 
    t = datetime.datetime(x_tick,1,1,0,0,0)
    t_jd = julian.to_jd(t, fmt='jd')-initial_julian
    x_jul.append(t_jd)
    x_tick_time.append(x_tick)

peaks,troughs = list(utils.get_peaks_troughs(number,100))
plt.figure(figsize=(18, 5))
plt.grid(True, linestyle='--', linewidth=1.0)
plt.plot(jul_date, number, color='dodgerblue', marker='o', markersize=3)  
x_peaks = []
for x,y in peaks:
    x_peaks.append(jul_date[x])
    plt.text(jul_date[x],y,'({},{})'.format(julian.from_jd(jul_date[x]+initial_julian,fmt='jd'),y),
        fontsize=10,verticalalignment='bottom',horizontalalignment='center',color='b')
x_troughs = []
for x,y in troughs:
    x_troughs.append(jul_date[x])
    plt.text(jul_date[x],y,'({}, {})'.format(julian.from_jd(jul_date[x]+initial_julian,fmt='jd'),y),
        fontsize=10,verticalalignment='top',horizontalalignment='center',color='b')
plt.title(f'All Set', fontproperties='Arial', fontsize=20, color='red')
plt.xlabel('Date', fontproperties='Arial', fontsize=18)         
plt.ylabel('Total Sunspots Number', fontproperties='Arial', fontsize=18)  
plt.xticks(x_jul, x_tick_time, size=14)
plt.yticks(size=14)
plt.xlim(0, end_julian-initial_julian)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
plt.tight_layout()
plt.show()

jul_date_peaks = np.empty(shape=(len(jul_date), len(x_peaks)))
for key, i in enumerate(x_peaks):
    result = []
    for j in np.arange(len(jul_date)):
        result.append(jul_date[j]-int(i))
    jul_date_peaks[:, key] = result
jul_date_peaks0 = []
for i in np.arange(jul_date_peaks.shape[0]):
    for j in np.arange(jul_date_peaks.shape[1]):
        if jul_date_peaks[i,j]>=0:
            L = [jul_date_peaks[i,j]]
    jul_date_peaks0.append(min(L))
jul_date_peaks0 = np.array(jul_date_peaks0).reshape(-1,1)

jul_date_troughs = np.empty(shape=(len(jul_date), len(troughs)))
for key, i in enumerate(x_troughs):
    result = []
    for j in np.arange(len(jul_date)):
        result.append(jul_date[j]-int(i))
    jul_date_troughs[:, key] = result
jul_date_troughs0 = []
for i in np.arange(jul_date_troughs.shape[0]):
    for j in np.arange(jul_date_troughs.shape[1]):
        if jul_date_troughs[i,j]>=0:
            L = [jul_date_troughs[i,j]]
    jul_date_troughs0.append(min(L))
jul_date_troughs0 = np.array(jul_date_troughs0).reshape(-1,1)            

np.savetxt(file_location+r'\data1749_2021\jul_date_lat_upper_lower.txt', 
    np.concatenate((jul_date_peaks0,jul_date_troughs0),axis=1), 
    fmt='%.1f', delimiter=' ')

plt.figure(figsize=(18, 5))
plt.grid(True, linestyle='--', linewidth=1.0)
plt.plot(jul_date, jul_date_troughs0, \
    color='dodgerblue', label='Troughs')    
plt.plot(jul_date, jul_date_peaks0, \
    color='indianred', label='Peaks')    
plt.title(f'Troughs/Peaks', fontproperties='Arial', fontsize=20, color='red')
plt.xlabel('Date', fontproperties='Arial', fontsize=18)         
plt.ylabel('Time Intervals', fontproperties='Arial', fontsize=18)  
plt.xticks(x_jul, x_tick_time, size=14)
plt.yticks(size=14)
plt.xlim(0, end_julian-initial_julian)
plt.legend(loc='best', fontsize=15)
plt.tight_layout()
plt.show()