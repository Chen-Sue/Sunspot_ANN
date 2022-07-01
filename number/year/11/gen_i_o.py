
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
sun_epoch = config.sun_epoch
in_cycle = config.in_cycle

file_location = config.file_location
initial_julian = julian.to_jd(datetime.datetime(1700,1,1,0,0,0), fmt='jd')

data = np.loadtxt(file_location+r'\data1700_2021\SN_y_tot_V2.0.txt')

sunspots =np.array(data[:, 1], dtype=np.float32)
year = np.array(data[:, 0], dtype=np.int)
month = np.ones_like(year)
day = np.ones_like(year)
date = pd.DataFrame({'year':year,'month':month,'day':day})
date = pd.to_datetime(date, format='%Y%m%d', errors='ignore')

with open(file_location+r'\data1700_2021\sn_train{}.txt'.format(in_cycle*sun_epoch), mode='w+', encoding='utf-8') as fout1:
    with open(file_location+r'\data1700_2021\sn_pre{}.txt'.format(in_cycle*sun_epoch), mode='w+', encoding='utf-8') as fout2:
        for y in np.arange(config.min_year, config.max_year+1):          
            print('year:', y)
            if datetime.datetime(y,1,1,0,0,0) > datetime.datetime(2020,1,1,0,0,0)\
                    -relativedelta(years=in_cycle*sun_epoch-1): break
            elif datetime.datetime(y,1,1,0,0,0) <= datetime.datetime(2020,1,1,0,0,0)\
                    -relativedelta(years=(in_cycle+1)*sun_epoch-1):
                result = []
                for i in np.arange(0,(in_cycle+1)*sun_epoch,1):
                    date_begin = datetime.datetime(y,1,1,0,0,0)+relativedelta(years=i)
                    date_end = datetime.datetime(y,1,1,0,0,0)+relativedelta(years=i+1)
                    days = (date_end - date_begin).days
                    flag_year = np.logical_and(date>=date_begin, date<date_end)
                    # print('\t date_begin:', date_begin, 'date_end', date_end)
                    lat = sunspots[flag_year]
                    result.append(lat) 
                fout1.write('{} 1 1 {}\n'.format(y+in_cycle*sun_epoch, result)) 
            elif datetime.datetime(y,1,1,0,0,0) > datetime.datetime(2020,1,1,0,0,0)\
                    -relativedelta(years=(in_cycle+1)*sun_epoch-1) and \
                datetime.datetime(y,1,1,0,0,0) <= datetime.datetime(2020,1,1,0,0,0)\
                    -relativedelta(years=in_cycle*sun_epoch-1) :
                result = []
                for i in np.arange(0,in_cycle*sun_epoch,1):
                    date_begin = datetime.datetime(y,1,1,0,0,0)+relativedelta(years=i)
                    date_end = datetime.datetime(y,1,1,0,0,0)+relativedelta(years=i+1)
                    days = (date_end - date_begin).days
                    flag_year = np.logical_and(date>=date_begin, date<date_end)
                    # print('\t date_begin:', date_begin, 'date_end', date_end)
                    lat = sunspots[flag_year]
                    result.append(lat)    
                fout2.write('{} {} 1 {}\n'.format(y+in_cycle*sun_epoch, 1, result)) 

with open(file_location+r'\data1700_2021\sn_train{}.txt'.format(in_cycle*sun_epoch), 'r') as f:
    content = f.read()
with open(file_location+r'\data1700_2021\sn_train{}.txt'.format(in_cycle*sun_epoch), 'w') as f:
    content = content.replace('[', '')
    content = content.replace(']', '')
    content = content.replace(',', '')
    content = content.replace('array(', '')
    content = content.replace(')', '') 
    content = content.replace('dtype=float32', '')
    f.write(content)
data = np.loadtxt(file_location+r'\data1700_2021\sn_train{}.txt'.format(in_cycle*sun_epoch))
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
np.savetxt(file_location+r'\data1700_2021\sn_train_jul_date{}.txt'.format(in_cycle*sun_epoch), \
    jul_data0, fmt='%.0f', delimiter=' ')


with open(file_location+r'\data1700_2021\sn_pre{}.txt'.format(in_cycle*sun_epoch), 'r') as f:
    content = f.read()
with open(file_location+r'\data1700_2021\sn_pre{}.txt'.format(in_cycle*sun_epoch), 'w') as f:
    content = content.replace('[', '')
    content = content.replace(']', '')
    content = content.replace(',', '')
    content = content.replace('array(', '')
    content = content.replace(')', '') 
    content = content.replace('dtype=float32', '')
    f.write(content)
