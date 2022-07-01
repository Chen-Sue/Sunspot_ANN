import os
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
# import utils 

start_time = time.time()
np.random.seed(config.seed)

file_location = os.getcwd()
font = FontProperties(fname=r"C:\WINDOWS\Fonts\simsun.ttc", size=18)
initial_julian = julian.to_jd(datetime.datetime(1740,1,1,0,0,0), fmt='jd')

data = np.loadtxt(file_location+r'\SN_y_tot_V2.0.txt')
sunspots1 =np.array(data[:, 1], dtype=np.float32)
year = np.array(data[:, 0], dtype=np.int)
month = np.ones_like(year)
day = np.ones_like(year)
date = pd.DataFrame({'year':year,'month':month,'day':day})
date = pd.to_datetime(date, format='%Y%m%d', errors='ignore')

jul_date1 = []
for i in np.arange(len(data)): 
    t = datetime.datetime.strptime(str(date[i]), '%Y-%m-%d %H:%M:%S')
    t_jd = julian.to_jd(t, fmt='jd')-initial_julian
    jul_date1.append(t_jd)
np.savetxt(file_location+r'\sn_y_jul_date.txt', \
    jul_date1, fmt='%.1f', delimiter=' ')

x_label2, times = [], []
for t in np.arange(1700, 2030+1, config.step):
    t_jd = julian.to_jd(datetime.datetime(t,1,1,0,0,0), fmt='jd')-initial_julian
    x_label2.append(t_jd)
    times.append(t)


initial_julian = julian.to_jd(datetime.datetime(1740,1,1,0,0,0), fmt='jd')
# file_location = r'D:\sunspot\number\month\1'
data = np.loadtxt(file_location+r'\SN_m_tot_V2.0.txt')
sunspots =np.array(data[:, 3], dtype=np.float32)
year = np.array(data[:, 0], dtype=np.int)
month = np.array(data[:, 1], dtype=np.int)
day = np.ones_like(month)
date = pd.DataFrame({'year':year,'month':month,'day':day})
date = pd.to_datetime(date, format='%Y%m%d', errors='ignore')

jul_data = []
for i in np.arange(len(data)): 
    t = datetime.datetime.strptime(str(date[i]), '%Y-%m-%d %H:%M:%S')
    t_jd = julian.to_jd(t, fmt='jd')-initial_julian
    jul_data.append(t_jd)
np.savetxt(file_location+r'\sn_m_jul_data.txt', \
    jul_data, fmt='%.1f', delimiter=' ')
    
fig, ax = plt.subplots(figsize=(14,4))
ax.grid(True, linestyle='--', linewidth=1)
ax.plot(jul_data, sunspots, marker='o', markersize=4, 
    color="dodgerblue", markerfacecolor='white', alpha=0.75, 
    linewidth=2, label=u'月平均太阳黑子数')
# markerline, stemlines, baseline = plt.stem(jul_data, sunspots, 
#     linefmt='dodgerblue', basefmt='-', markerfmt='')
# plt.setp(markerline, color='dodgerblue')
ax.plot(jul_date1, sunspots1, marker='s', markersize=4, 
    color="indianred", markerfacecolor='white', alpha=0.75, 
    linewidth=2, label=u'年平均太阳黑子数')
ax.set_xlabel('日期', fontproperties=font, fontsize=18)
ax.set_ylabel('太阳黑子数', fontproperties=font, fontsize=18)
ax.set_xticks(x_label2) 
ax.set_xticklabels(times, fontproperties=font, fontsize=18, rotation=45)
[y_label.set_fontname('STSong') for y_label in ax.get_yticklabels()]
ax.tick_params(labelsize=18)
x_min = [1756,1766,1775,1786,1799,
         1813,1824,1834,1844,1855,
         1866,1877,1889,1901,1912,
         1922,1932,1943,1953,1964,
         1976,1985,1996,2008,2019]
for key, value in enumerate(x_min):
    jul = julian.to_jd(datetime.datetime(value+3,1,1,0,0,0), fmt='jd')-initial_julian
    ax.text(jul, 5, '{}'.format(1+key), fontproperties=font, fontsize=20, c='k')
ax.set_xlim(min(x_label2)-1000, max(x_label2)+1000)
ax.legend(loc='upper left', ncol=1, prop=FontProperties(fname=r"C:\WINDOWS\Fonts\simsun.ttc", size=14)) 
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
plt.tight_layout()
plt.savefig(r'D:\sunspot\figure\fig1.png', dpi=600)
plt.savefig(r'D:\sunspot\figure\fig1.eps', dpi=600)
plt.show()