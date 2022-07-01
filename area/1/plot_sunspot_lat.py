
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

# data = pd.read_csv(file_location+r'\RGO_NOAA1874_2016\merge_data.txt', 
#     delimiter=' ', header=None, dtype=np.float32)
# data = np.array(data)
# date = pd.DataFrame({'year':np.array(data[:, 0], dtype=np.int), \
#     'month':np.array(data[:, 1], dtype=np.int), \
#     'day':np.array(data[:, 2], dtype=np.int)})
# date = pd.to_datetime(date, format='%Y%m%d', errors='ignore')
# Latitude = data[:, -1]
# count0 = 0
# with open(file_location+r'\RGO_NOAA1874_2016\sunspot_number_lat{}.txt'.format(
#     config.blocks), mode='w+', encoding='utf-8') as fout:
#     for y in np.arange(config.min_year, config.max_year+1):        
#         print('year:', y)
#         for m in np.arange(1, 12+1):
#             if y==config.min_year and m<5:
#                 continue
#             elif datetime.datetime(y,m,1,0,0,0)+relativedelta(years=1) \
#                 > datetime.datetime(2016,10,1,0,0,0):
#                 break
#             else:
#                 result = []
#                 # count0 += 1
#                 # print(count0)
#                 date_begin = datetime.datetime(y,m,1,0,0,0)
#                 date_end = datetime.datetime(y,m,1,0,0,0)+relativedelta(years=1)
#                 flag_year = np.logical_and(date>date_begin, date<=date_end)
#                 lat = Latitude[flag_year]
#                 result.append(len(lat))
#             fout.write('{} {} 1 {}\n'.format(y, m, result)) 
# with open(file_location+r'\RGO_NOAA1874_2016\sunspot_number_lat{}.txt'.format(
#     config.blocks), 'r') as f:
#     content = f.read()
# content = content.replace('[', '')
# content = content.replace(']', '')
# content = content.replace(',', '')
# with open(file_location+r'\RGO_NOAA1874_2016\sunspot_number_lat{}.txt'.format(
#     config.blocks), 'w') as f:
#     f.write(content)

data = np.loadtxt(file_location+r'\RGO_NOAA1874_2016\sunspot_number_lat{}.txt'.format(
    config.blocks), delimiter=' ')
date = pd.DataFrame({'year':np.array(data[:,0], dtype=np.int), \
    'month':np.array(data[:,1], dtype=np.int), \
    'day':np.array(data[:,2], dtype=np.int)})
date = pd.to_datetime(date, format='%Y%m%d', errors='ignore')
initial_julian = julian.to_jd(datetime.datetime(1870,1,1,0,0,0), fmt='jd')
end_julian = julian.to_jd(datetime.datetime(2020,1,1,0,0,0), fmt='jd')
jul_date = []
for i in np.arange(len(data)): 
    t = datetime.datetime.strptime(str(date[i]), '%Y-%m-%d %H:%M:%S')
    jul_date.append(julian.to_jd(t, fmt='jd')-initial_julian)
number = np.array(data[:,-1], dtype=np.int)
np.savetxt(file_location+r'\RGO_NOAA1874_2016\jul_date_number_lat{}.txt'.format(
    config.blocks), number, fmt='%.1f', delimiter=' ')

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

data = np.loadtxt(file_location+r'\RGO_NOAA1874_2016\sunspot_lat{}.txt'.format(
    config.blocks), delimiter=' ')
jul_date = np.loadtxt(file_location+r'\RGO_NOAA1874_2016\jul_date_lat{}.txt'.format(
    config.blocks), delimiter=' ')
jul_date = jul_date.reshape(config.blocks, len(data)).T[:,0]
print(jul_date.shape)

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

np.savetxt(file_location+r'\RGO_NOAA1874_2016\jul_date_lat_upper_lower{}.txt'.format(
    config.blocks), 
    np.concatenate((jul_date_peaks0,jul_date_troughs0),axis=1), fmt='%.1f', delimiter=' ')

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