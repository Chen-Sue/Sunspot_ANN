
from astropy.time import Time
import datetime
from dateutil.relativedelta import relativedelta
from pylab import *
from matplotlib.font_manager import FontProperties

import config
import pandas as pd
import numpy as np
import math
import re
import matplotlib
import matplotlib.pyplot as plt
import julian

file_location = config.file_location
fontcn = FontProperties(fname=r"C:\WINDOWS\Fonts\simsun.ttc", size=14)
initial_julian = julian.to_jd(datetime.datetime(1870,1,1,0,0,0), fmt='jd')

step = 10
initial = julian.to_jd(datetime.datetime(1870,1,1,0,0,0), fmt='jd')

data =np.loadtxt(file_location+r'\RGO_NOAA1874_2021\merge_data.txt', 
    delimiter=' ')
year = np.array(data[:, 0], dtype=np.int)
month = np.array(data[:, 1], dtype=np.int)
day = np.array(data[:, 2], dtype=np.int)
date = pd.DataFrame({'year':year, 'month':month, 'day':day})
date = pd.to_datetime(date, format='%Y%m%d', errors='ignore')

area =  np.array(data[:,-2], dtype=np.float32)
Latitude = np.array(data[:, -1], dtype=np.float32)
print(len(Latitude), len(date))

# 横轴为儒略日
jul_date = []
for i in np.arange(len(date)):
    t = datetime.datetime.strptime(str(date[i]), '%Y-%m-%d %H:%M:%S')
    jd = julian.to_jd(t, fmt='jd')-initial
    jul_date.append(jd)
jul_date = np.array(jul_date)

print(len(jul_date))

x_jul, x_tick_time = [], [] # 设置横轴标签
for x_tick in np.arange(1870, 2030+1, config.step): 
    t = datetime.datetime(x_tick,1,1,0,0,0)
    t_jd = julian.to_jd(t, fmt='jd')-initial_julian
    x_jul.append(t_jd)
    x_tick_time.append(x_tick)

print(max(area))
fig = plt.figure(figsize=(14, 5))
ax = fig.add_subplot(1,1,1)
plt.scatter(jul_date[np.where(area<=80)], Latitude[np.where(area<=80)], 
    c='k', marker='|', s=5, label='$\leq$ 80ppm')
plt.scatter(jul_date[np.where((area>80) & (area<600))], 
    Latitude[np.where((area>80) & (area<600))], 
    c='r', marker='|', s=5, label='(80ppm,600ppm)')
plt.scatter(jul_date[np.where(area>=600)], Latitude[np.where(area>=600)], 
    c='y', marker='|', s=5, label='$\geq$ 600ppm')
# plt.scatter(jul_date[np.where(area<=10)], Latitude[np.where(area<=10)], 
#     c='dodgerblue', marker='s', s=0.8, label='(0,1)')
# plt.scatter(jul_date[np.where((area>10) & (area<=100))], 
#     Latitude[np.where((area>10) & (area<=100))], 
#     c='indianred', marker='s', s=0.8, label='(0.1%,1)')
# plt.scatter(jul_date[np.where(area>100)], Latitude[np.where(area>100)], 
#     c='darkviolet', marker='s', s=0.8, label='(1.0%,1)')
plt.xlabel('日期', fontproperties=fontcn, fontsize=18)
plt.ylabel('纬度', fontproperties=fontcn, fontsize=18)
plt.text(0.1, 92, '可见太阳半球中黑子面积', fontproperties=fontcn, fontsize=16)
plt.xticks(x_jul, x_tick_time, size=14)
# plt.yticks([-90,-60,-30,0,30,60],
#     ['120S','90S','30S','EQ','30N','90N'], size=14)
plt.yticks([-120,-90,-60,-30,0,30,60,90],
    ['120S','90S','60S','30S','EQ','30N','60N','90N'], size=14)
x_min = [1867,1878,1890,1902,\
    1913,1923,1933,1944,1954,1964,1976,1986,1996,2008,2019]
for key, value in enumerate(x_min):
    jul = julian.to_jd(datetime.datetime(value+3,1,1,0,0,0), fmt='jd')-initial_julian
    plt.text(jul, -75, '{}'.format(11+key), fontproperties=fontcn, fontsize=20, 
        c='k', bbox=dict(boxstyle='round, pad=0.5',alpha=0.1))
# for key,value in enumerate(np.arange(1870, 2031, 11)):
#     j = julian.to_jd(datetime.datetime(value,1,1,0,0,0), fmt='jd')-initial
#     plt.text(j, -75, '周期{}'.format(12+key), fontproperties=fontcn, fontsize=16)
plt.xlim(min(x_jul)-1000, max(x_jul)+1000)
plt.ylim(-90, 90)
plt.grid(True, linestyle='--', linewidth=1.0) 
plt.legend(loc='upper left', prop=fontcn, fontsize=16, ncol=1) 
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.yaxis.set_major_locator(plt.MultipleLocator(30))
plt.tight_layout()
plt.show()
