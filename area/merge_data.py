print('\tBegin merge_catalog.py')

import config
import pandas as pd
import numpy as np
import math
import re
import matplotlib.pyplot as plt

file_location = config.file_location

area, Latitude = [], []
time = np.empty(shape=(0,3))

for year in np.arange(config.min_year, config.max_year+1, 1):
    print('\tyear:', year)
    try:
        f = open(file_location+r'\RGO_NOAA1874_2021\old\g{}.txt'.format(year), 'r')
    except FileNotFoundError:
        print('File is not found')
    else:
        lines = f.readlines()
        for line in lines:
            a = line.split()
            Latitude.append(a[-2])
            area.append(line[40:40+4])
        
            t = []
            if ' ' in line[:6]:
                a = line.split(' ')
                y = int(a[0])
                if float(a[1]) <= 10:
                    m = int(a[1])
                    d = math.floor(float(a[2]))
                else:
                    m = int(float(a[1])//100)
                    d = math.floor(float(a[1]) % 100)
                t.append(y)
                t.append(m)
                t.append(d)
                time = np.concatenate((time, np.array(t).reshape(-1, 3)), axis=0)
            else:
                y = int(int(line[:6]) // 100)
                m = int(int(line[:6]) % 100)
                t.append(y)
                t.append(m)
                if ' ' in line[:8]:
                    d = math.floor(float(a[1]))
                else:
                    d = int(float(line[:8]) % 100)
                t.append(d)
                time = np.concatenate((time, np.array(t).reshape(-1, 3)), axis=0)
        f.close()

time = np.array(time, dtype=np.int).reshape(-1, 3)
area = np.array(area, dtype=np.float32).reshape(-1, 1)
Latitude = np.array(Latitude, dtype=np.float32).reshape(-1, 1)
np.savetxt(file_location+r'\RGO_NOAA1874_2021\merge_data.txt', 
    np.concatenate((time, area, Latitude), axis=1), fmt='%.1f', delimiter=' ')

print('\tEnd merge_catalog.py\n')

