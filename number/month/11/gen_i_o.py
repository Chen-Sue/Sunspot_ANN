
import time
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import julian
import datetime

import config
import utils 

start_time = time.time()
np.random.seed(config.seed)

in_cycle = config.in_cycle
sun_epoch = config.sun_epoch
file_location = config.file_location
initial_julian = julian.to_jd(datetime.datetime(1740,1,1,0,0,0), fmt='jd')

data = np.loadtxt(file_location+r'\data1749_2021\SN_m_tot_V2.0.txt')

sunspots = np.array(data[:, 3], dtype=np.float32)
year = np.array(data[:, 0], dtype=np.int)
month = np.array(data[:, 1], dtype=np.int)
day = np.ones_like(month)
date = pd.DataFrame({'year':year,'month':month,'day':day})
date = pd.to_datetime(date, format='%Y%m%d', errors='ignore')

with open(file_location+r'\data1749_2021\sn_train_mul.txt', mode='w+', encoding='utf-8') as fout1:
    with open(file_location+r'\data1749_2021\sn_pre_mul.txt', mode='w+', encoding='utf-8') as fout2:
        for y in np.arange(config.min_year, config.max_year+1):        
        # for y in np.arange(config.min_year, config.min_year+2):        
            print('year:', y)
            for m in np.arange(1, 12+1):
                if y==config.min_year and m<5:
                    continue
                elif datetime.datetime(y,m,1,0,0,0) > datetime.datetime(2021,6,1,0,0,0)\
                        -relativedelta(months=in_cycle*sun_epoch*12):
                    break
                elif datetime.datetime(y,m,1,0,0,0) <= datetime.datetime(2021,6,1,0,0,0)\
                        -relativedelta(months=2*in_cycle*sun_epoch*12) :
                    result = []
                    for i in np.arange(0,2*in_cycle*sun_epoch*12,1):
                        date_begin = datetime.datetime(y,m,1,0,0,0)+relativedelta(months=i)
                        date_end = datetime.datetime(y,m,1,0,0,0)+relativedelta(months=i+1)
                        days = (date_end - date_begin).days
                        flag_year = np.logical_and(date>=date_begin, date<date_end)
                        lat = sunspots[flag_year]
                        result.append(lat) 
                    fout1.write('{} {} 1 {}\n'.format(y+in_cycle*sun_epoch, m, result)) 
                elif datetime.datetime(y,m,1,0,0,0) > datetime.datetime(2021,6,1,0,0,0)\
                        -relativedelta(months=2*in_cycle*sun_epoch*12) and \
                    datetime.datetime(y,m,1,0,0,0) <= datetime.datetime(2021,6,1,0,0,0)\
                        -relativedelta(months=in_cycle*sun_epoch*12):
                    result = []
                    for i in np.arange(0,in_cycle*sun_epoch*12,1):
                        date_begin = datetime.datetime(y,m,1,0,0,0)+relativedelta(months=i)
                        date_end = datetime.datetime(y,m,1,0,0,0)+relativedelta(months=i+1)
                        days = (date_end - date_begin).days
                        flag_year = np.logical_and(date>=date_begin, date<date_end)
                        # print('\t date_begin:', date_begin, 'date_end', date_end)
                        lat = sunspots[flag_year]
                        result.append(lat)    
                    fout2.write('{} {} 1 {}\n'.format(y+sun_epoch, m, result)) 
                
with open(file_location+r'\data1749_2021\sn_train_mul.txt', 'r') as f:
    content = f.read()
with open(file_location+r'\data1749_2021\sn_train_mul.txt', 'w') as f:
    content = content.replace('[', '')
    content = content.replace(']', '')
    content = content.replace(',', '')
    content = content.replace('array(', '')
    content = content.replace(')', '') 
    content = content.replace('dtype=float32', '')
    f.write(content)
data = np.loadtxt(file_location+r'\data1749_2021\sn_train_mul.txt')
year = np.array(data[:, 0], dtype=np.int)
month = np.array(data[:, 1], dtype=np.int)
day = np.array(data[:, 2], dtype=np.int)
date = pd.DataFrame({'year':year, 'month':month, 'day':day})
date = pd.to_datetime(date, format='%Y%m%d', errors='ignore')
jul_data0 = np.zeros(shape=(len(date),in_cycle*sun_epoch*12))
for i in np.arange(len(date)):
    for j in np.arange(in_cycle*sun_epoch*12): 
        t = datetime.datetime.strptime(str(date[i]), '%Y-%m-%d %H:%M:%S')\
             + relativedelta(months=j)
        t_jd = julian.to_jd(t,fmt='jd')-initial_julian
        jul_data0[i,j] = t_jd
np.savetxt(file_location+r'\data1749_2021\sn_train_jul_date_mul.txt', \
    jul_data0, fmt='%.0f', delimiter=' ')


with open(file_location+r'\data1749_2021\sn_pre_mul.txt', 'r') as f:
    content = f.read()
with open(file_location+r'\data1749_2021\sn_pre_mul.txt', 'w') as f:
    content = content.replace('[', '')
    content = content.replace(']', '')
    content = content.replace(',', '')
    content = content.replace('array(', '')
    content = content.replace(')', '') 
    content = content.replace('dtype=float32', '')
    f.write(content)
data = np.loadtxt(file_location+r'\data1749_2021\sn_pre_mul.txt')
year = np.array(data[:, 0], dtype=np.int)
month = np.array(data[:, 1], dtype=np.int)
day = np.array(data[:, 2], dtype=np.int)
date = pd.DataFrame({'year':year, 'month':month, 'day':day})
date = pd.to_datetime(date, format='%Y%m%d', errors='ignore')
jul_date_pre = np.zeros(shape=(len(date),in_cycle*sun_epoch*12))
for i in np.arange(len(date)):
    for j in np.arange(in_cycle*sun_epoch*12): 
        t = datetime.datetime.strptime(str(date[i]), '%Y-%m-%d %H:%M:%S')\
             + relativedelta(months=j)
        t_jd = julian.to_jd(t,fmt='jd')-initial_julian
        jul_date_pre[i,j] = t_jd
np.savetxt(file_location+r'\data1749_2021\sn_pre_jul_date_mul.txt', \
    jul_date_pre, fmt='%.0f', delimiter=' ')
