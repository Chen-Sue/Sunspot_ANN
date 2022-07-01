print('\tBegin gen_i_o.py')

import time
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import julian
from matplotlib.font_manager import FontProperties

import config
import utils 

start_time = time.time()
np.random.seed(config.seed)

sun_epoch = config.sun_epoch
in_cycle = config.in_cycle

file_location = config.file_location
fontcn = FontProperties(fname=r'C:\WINDOWS\Fonts\simsun.ttc', size=14)
initial_julian = julian.to_jd(datetime.datetime(1870,1,1,0,0,0), fmt='jd')
x_jul, x_tick_time = [], [] 
for x_tick in np.arange(1870, 2030+1, config.step): 
    t = datetime.datetime(x_tick,1,1,0,0,0)
    t_jd = julian.to_jd(t, fmt='jd')-initial_julian
    x_jul.append(t_jd)
    x_tick_time.append(x_tick)

data = np.loadtxt(file_location+r'\RGO_NOAA1874_2021\merge_data.txt')
area = data[:,-2]
year = np.array(data[:, 0],dtype=np.int)
month = np.array(data[:, 1],dtype=np.int)
day = np.array(data[:, 2],dtype=np.int)
date = pd.DataFrame({'year':year,'month':month,'day':day})
date = pd.to_datetime(date,format='%Y%m%d',errors='ignore')

count0 = 0
with open(file_location+r'\RGO_NOAA1874_2021\sa_train.txt', mode='w+', encoding='utf-8') as fout1:
    with open(file_location+r'\RGO_NOAA1874_2021\sa_pre.txt', mode='w+', encoding='utf-8') as fout2:
        for y in np.arange(config.min_year, config.max_year+1):        
        # for y in np.arange(config.min_year, config.min_year+2):        
            print('year:', y)
            for m in np.arange(1, 12+1):
                if y==config.min_year and m<5: continue
                elif datetime.datetime(y,m,1,0,0,0) > datetime.datetime(2021,6,1,0,0,0)\
                        -relativedelta(months=in_cycle*sun_epoch*12): break
                elif datetime.datetime(y,m,1,0,0,0) <= datetime.datetime(2021,6,1,0,0,0)\
                        -relativedelta(months=in_cycle*sun_epoch*12+1) :
                    result = []
                    # count0 += 1
                    # print(count0)
                    for i in np.arange(0,in_cycle*sun_epoch*12+1,1):
                        date_begin = datetime.datetime(y,m,1,0,0,0)+relativedelta(months=i)
                        date_end = datetime.datetime(y,m,1,0,0,0)+relativedelta(months=i+1)
                        days = (date_end - date_begin).days
                        flag_year = np.logical_and(date>=date_begin, date<date_end)
                        # print('\t date_begin:', date_begin, 'date_end', date_end)
                        lat = area[flag_year]
                        days = (date_end - date_begin).days
                        result.append(np.sum(lat)/days)
                        # print(result)
                    fout1.write('{} {} 1 {}\n'.format(y+sun_epoch, m, result)) 
                elif datetime.datetime(y,m,1,0,0,0) > datetime.datetime(2021,6,1,0,0,0)\
                        -relativedelta(months=in_cycle*sun_epoch*12+1) and \
                    datetime.datetime(y,m,1,0,0,0) <= datetime.datetime(2021,6,1,0,0,0)\
                        -relativedelta(months=in_cycle*sun_epoch*12) :
                    result = []
                    for i in np.arange(0,in_cycle*sun_epoch*12,1):
                        date_begin = datetime.datetime(y,m,1,0,0,0)+relativedelta(months=i)
                        date_end = datetime.datetime(y,m,1,0,0,0)+relativedelta(months=i+1)
                        days = (date_end - date_begin).days
                        flag_year = np.logical_and(date>=date_begin, date<date_end)
                        # print('\t date_begin:', date_begin, 'date_end', date_end)
                        lat = area[flag_year]
                        days = (date_end - date_begin).days
                        result.append(np.sum(lat)/days)   
                    fout2.write('{} {} 1 {}\n'.format(y+sun_epoch, m, result)) 


with open(file_location+r'\RGO_NOAA1874_2021\sa_train.txt', 'r') as f:
    content = f.read()
with open(file_location+r'\RGO_NOAA1874_2021\sa_train.txt', 'w') as f:
    content = content.replace('[', '')
    content = content.replace(']', '')
    content = content.replace(',', '')
    content = content.replace('array(', '')
    content = content.replace(')', '') 
    content = content.replace('dtype=float32', '')
    f.write(content)
data = np.loadtxt(file_location+r'\RGO_NOAA1874_2021\sa_train.txt')
year = np.array(data[:, 0], dtype=np.int)
month = np.array(data[:, 1], dtype=np.int)
day = np.array(data[:, 2], dtype=np.int)
date = pd.DataFrame({'year':year, 'month':month, 'day':day})
date = pd.to_datetime(date, format='%Y%m%d', errors='ignore')
jul_data0 = []
for i in np.arange(len(date)): 
    t = datetime.datetime.strptime(str(date[i]), '%Y-%m-%d %H:%M:%S')
    t_jd = julian.to_jd(t,fmt='jd')-initial_julian
    jul_data0.append(t_jd)
np.savetxt(file_location+r'\RGO_NOAA1874_2021\sa_train_jul_date.txt', \
    jul_data0, fmt='%.0f', delimiter=' ')


with open(file_location+r'\RGO_NOAA1874_2021\sa_pre.txt', 'r') as f:
    content = f.read()
with open(file_location+r'\RGO_NOAA1874_2021\sa_pre.txt', 'w') as f:
    content = content.replace('[', '')
    content = content.replace(']', '')
    content = content.replace(',', '')
    content = content.replace('array(', '')
    content = content.replace(')', '') 
    content = content.replace('dtype=float32', '')
    f.write(content)

# with open(file_location+r'\RGO_NOAA1874_2021\sunspot_area_lat{}.txt'.format(
#     config.blocks),mode='w+',encoding='utf-8') as fout:
#     for y in np.arange(config.min_year,config.max_year+1):          
#         print('\tyear:', y)
#         for m in np.arange(1, 12+1):
#             if y==config.min_year and m<5: continue
#             elif datetime.datetime(y,m,1,0,0,0)>=datetime.datetime(2021,6,1,0,0,0):
#                 break
#             else:
#                 result = []
#                 count0 += 1
#                 print('\t count=', count0)
#                 date_begin = datetime.datetime(y,m,1,0,0,0)
#                 date_end = datetime.datetime(y,m,1,0,0,0)+relativedelta(months=1)
#                 flag_year = np.logical_and(date>=date_begin, date<date_end)
#                 print('\t date_begin:', date_begin, ', date_end', date_end)
#                 lat = area[flag_year]
#                 days = (date_end - date_begin).days
#                 print(np.sum(lat)/days, lat)
#                 result.append(np.sum(lat)/days)
#                 fout.write('{} {} 1 {}\n'.format(y, m, result)) 


# with open(file_location+r'\RGO_NOAA1874_2021\sunspot_area_lat{}.txt'.format(
#     config.blocks), 'r') as f:
#     content = f.read()
# content = content.replace('[', '')
# content = content.replace(']', '')
# content = content.replace(',', '')
# with open(file_location+r'\RGO_NOAA1874_2021\sunspot_area_lat{}.txt'.format(
#     config.blocks), 'w') as f:
#     f.write(content)

# data = np.loadtxt(file_location+r'\RGO_NOAA1874_2021\sunspot_area_lat{}.txt'.format(
#     config.blocks))
# year = np.array(data[:, 0],dtype=np.int)
# month = np.array(data[:, 1],dtype=np.int)
# day = np.array(data[:, 2],dtype=np.int)
# date = pd.DataFrame({'year':year,'month':month,'day':day})
# date = pd.to_datetime(date,format='%Y%m%d',errors='ignore')

# jul_date = [] 
# for i in np.arange(len(data)): 
#     t = datetime.datetime.strptime(str(date[i]), '%Y-%m-%d %H:%M:%S')
#     t_jd = julian.to_jd(t, fmt='jd')-initial_julian
#     jul_date.append(t_jd)
# np.savetxt(file_location+r'\RGO_NOAA1874_2021\sunspot_area_lat_jul_date{}.txt'.format(
#     config.blocks), jul_date, fmt='%.0f', delimiter=' ')
# jul_date = np.loadtxt(file_location+r'\RGO_NOAA1874_2021\sunspot_area_lat_jul_date{}.txt'.format(
#     config.blocks))

# fig = plt.figure(figsize=(14, 5))
# ax = fig.add_subplot(1,1,1)
# plt.grid(True, linestyle='--', linewidth=1)
# plt.plot(jul_date, data[:, -1], marker='|', c='black')
# plt.xlabel('日期', fontproperties=fontcn, fontsize=18)
# plt.ylabel('月平均太阳黑子面积', fontproperties=fontcn, fontsize=18)
# plt.xticks(x_jul, x_tick_time, size=14)
# plt.yticks(size=14)
# # for key,value in enumerate(np.arange(1870, 2031, 11)):
# #     j = julian.to_jd(datetime.datetime(value,1,1,0,0,0), fmt='jd')-initial
# #     plt.text(j, -75, '周期{}'.format(12+key), fontproperties=fontcn, fontsize=16)
# plt.xlim(min(x_jul)-1000, max(x_jul)+1000)
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(1.5)
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['top'].set_linewidth(1.5)
# ax.spines['right'].set_linewidth(1.5)
# plt.tight_layout()
# plt.show()

print('\ End gen_i_o.py')
