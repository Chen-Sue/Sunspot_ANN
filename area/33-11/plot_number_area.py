
print('\tBegin gen_i_o.py')

import time
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import julian
import datetime
from matplotlib.font_manager import FontProperties

import config
import utils 

start_time = time.time()
np.random.seed(config.seed)

file_location = config.file_location
fontcn = FontProperties(fname=r"C:\WINDOWS\Fonts\simsun.ttc", size=14)

x_jul, x_tick_time = [], []
for x_tick in np.arange(1870, 2030+1, config.step): 
    t = datetime.datetime(x_tick,1,1,0,0,0)
    t_jd = julian.to_jd(t, fmt='jd')-julian.to_jd(datetime.datetime(1870,1,1,0,0,0), fmt='jd')
    x_jul.append(t_jd)
    x_tick_time.append(x_tick)

x_label2, times = [], []
for t in np.arange(1740, 2030+1, config.step):
    t_jd = julian.to_jd(datetime.datetime(t,1,1,0,0,0), fmt='jd')-\
        julian.to_jd(datetime.datetime(1740,1,1,0,0,0), fmt='jd')
    x_label2.append(t_jd)
    times.append(t)

data = np.loadtxt(file_location+r'\RGO_NOAA1874_2021\sunspot_number_lat{}.txt'.format(
    config.blocks))
area = data[:,-1].reshape(-1,1)
jul_date_area = np.loadtxt(file_location+r'\RGO_NOAA1874_2021\sunspot_number_lat_jul_date{}.txt'.format(
    config.blocks))
for i in np.arange(len(jul_date_area)): 
    jul_date_area[i] = jul_date_area[i]+julian.to_jd(datetime.datetime(1870,1,1,0,0,0), fmt='jd')-\
        julian.to_jd(datetime.datetime(1740,1,1,0,0,0), fmt='jd')
    # x_jul.append(t_jd)
    # x_tick_time.append(x_tick)

# data = np.loadtxt(file_location+r'..\RGO_NOAA1874_2021\sunspot_number_lat{}.txt'.format(
#     config.blocks))
# sunspots = data[:,-1]
file_location = r'D:\sunspot\number'
data = np.loadtxt(file_location+'\RGO_NOAA1874_2021\SN_m_tot_V2.0.txt')
sunspots = data[:, 3].reshape(-1,1)
jul_date_sunspots = np.loadtxt(file_location+'\RGO_NOAA1874_2021\sunspot_number_lat_jul_date{}.txt'.format(
    config.blocks))

fig = plt.figure(figsize=(14, 5))
ax1 = fig.add_subplot(111)
ax1.grid(True, linestyle='--', linewidth=1)
ax1.plot(jul_date_sunspots, sunspots, c='dodgerblue', label=u'太阳黑子数')
ax1.set_ylabel('月平均太阳黑子数', fontproperties=fontcn, fontsize=18)
ax1.set_xlabel('日期', fontproperties=fontcn, fontsize=18)
# ax1.set_ylim(0, 1500)
# ax1.set_xlim(0, 1000)
ax1.set_xticks(x_label2) 
ax1.set_xticklabels(times, fontproperties=fontcn, fontsize=14)
ax1.set_xlim(min(x_label2)-1000, max(x_label2)+1000)
ax1.legend(loc='upper left', prop=fontcn)  
ax2 = ax1.twinx()
ax2.plot(jul_date_area, area, c='indianred', label=u'太阳黑子面积')
ax2.set_ylabel('月平均太阳黑子面积', fontproperties=fontcn, fontsize=18)
ax2.legend(loc='upper right', prop=fontcn)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
plt.tight_layout()
plt.show()


print('\ End gen_i_o.py')
