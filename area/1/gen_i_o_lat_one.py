
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

file_location = config.file_location
lat_sep = config.lat_sep

data = np.loadtxt(file_location+r'\RGO_NOAA1874_2021\merge_data.txt')
area = np.array(data[:,-2],dtype=np.float32)
Latitude = np.array(data[:,-1],dtype=np.float32)
year = np.array(data[:, 0],dtype=np.int)
month = np.array(data[:, 1],dtype=np.int)
day = np.array(data[:, 2],dtype=np.int)
date = pd.DataFrame({'year':year,'month':month,'day':day})
date = pd.to_datetime(date,format='%Y%m%d',errors='ignore')

count0 = 0
with open(file_location+r'\RGO_NOAA1874_2021\sunspot_area_lat_one{}.txt'.format(
    config.blocks), mode='w+', encoding='utf-8') as fout:
    for y in np.arange(config.min_year, config.max_year+1):        
    # for y in np.arange(config.min_year, config.min_year+1):        
        print('year:', y)
        for m in np.arange(1, 12+1):
            if y==config.min_year and m<5:
                continue
            elif datetime.datetime(y,m,1,0,0,0)+relativedelta(
                    months=config.input_window*config.sun_epoch*12+1) \
                >= datetime.datetime(2021,6,1,0,0,0):
                break
            else:
                result = []
                # count0 += 1
                # print(count0)
                for i in np.arange(0,config.input_window*config.sun_epoch*12+1,1):
                    date_begin = datetime.datetime(y,m,1,0,0,0)+relativedelta(months=i)
                    date_end = datetime.datetime(y,m,1,0,0,0)+relativedelta(months=i+1)
                    days = (date_end - date_begin).days
                    flag_year = np.logical_and(date>=date_begin, date<date_end)
                    # print('\t date_begin:', date_begin, 'date_end', date_end)
                    for key, value in enumerate(lat_sep):
                        if value == lat_sep[0]:
                            flag = np.logical_and(flag_year, Latitude<lat_sep[1])
                        elif value == lat_sep[-1]:
                            flag = np.logical_and(flag_year, Latitude>=lat_sep[-1])
                        else:
                            flag_loc = np.logical_and(Latitude>=lat_sep[key], Latitude<lat_sep[key+1])
                            flag = np.logical_and(flag_year, flag_loc)
                        lat = area[flag]
                        result.append(np.sum(lat)/days)    
                fout.write('{} {} 1 {}\n'.format(y+12, m, result)) 

with open(file_location+r'\RGO_NOAA1874_2021\sunspot_area_lat_one{}.txt'.format(
    config.blocks), 'r') as f:
    content = f.read()
content = content.replace('[', '')
content = content.replace(']', '')
content = content.replace(',', '')
with open(file_location+r'\RGO_NOAA1874_2021\sunspot_area_lat_one{}.txt'.format(
    config.blocks), 'w') as f:
    f.write(content)

data = np.loadtxt(file_location+r'\RGO_NOAA1874_2021\sunspot_area_lat_one{}.txt'.format(
    config.blocks), delimiter=' ')
month = np.array(data[:, 1], dtype=np.int)
day = np.array(data[:, 2], dtype=np.int)
initial_julian = julian.to_jd(datetime.datetime(1870,1,1,0,0,0), fmt='jd')
end_julian = julian.to_jd(datetime.datetime(2020,1,1,0,0,0), fmt='jd')
jul_date = [] # 横轴为儒略日
for j in np.arange(config.blocks):
    for i in np.arange(len(data)): 
        names = globals()
        names['year_'+str(j+1)] = np.array(data[:, 0], dtype=np.int)
        names['date_'+str(j+1)] = pd.DataFrame({'year':names['year_'+str(j+1)], 'month':month, 'day':day})
        names['date_'+str(j+1)] = pd.to_datetime(names['date_'+str(j+1)], format='%Y%m%d', errors='ignore')
        t = datetime.datetime.strptime(str(names['date_'+str(j+1)][i]), '%Y-%m-%d %H:%M:%S')
        t_jd = julian.to_jd(t, fmt='jd')-initial_julian
        jul_date.append(t_jd)
np.savetxt(file_location+r'\RGO_NOAA1874_2021\sunspot_area_lat_jul_date_one{}.txt'.format(
    config.blocks), jul_date, fmt='%.1f', delimiter=' ')