
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
import os
from matplotlib.lines import Line2D

import config

start_time = time.time()
np.random.seed(config.seed)

file_location = os.getcwd()
font = FontProperties(fname=r"C:\WINDOWS\Fonts\simsun.ttc", size=18)

# 1700
julian_1700 = julian.to_jd(datetime.datetime(1700,1,1,0,0,0), fmt='jd')
x_jul_1700, x_tick_time_1700= [], []
for x_tick in np.arange(1700, 2030+1, config.step): 
    t = datetime.datetime(x_tick,1,1,0,0,0)
    t_jd = julian.to_jd(t, fmt='jd')-julian_1700
    x_jul_1700.append(t_jd)
    x_tick_time_1700.append(x_tick)
data = np.loadtxt(file_location+r'\SN_y_tot_V2.0.txt')
sunspots_year =np.array(data[:, 1], dtype=np.float32)
year = np.array(data[:, 0], dtype=np.int)
month = np.ones_like(year)
day = np.ones_like(year)
date = pd.DataFrame({'year':year,'month':month,'day':day})
date = pd.to_datetime(date, format='%Y%m%d', errors='ignore')
jul_date_1700 = []
for i in np.arange(len(data)): 
    t = datetime.datetime.strptime(str(date[i]), '%Y-%m-%d %H:%M:%S')
    t_jd = julian.to_jd(t, fmt='jd')-julian_1700
    jul_date_1700.append(t_jd)
np.savetxt(file_location+r'\sn_y_jul_date_1700.txt', jul_date_1700, fmt='%.1f', delimiter=' ')

# 1740
julian_1740 = julian.to_jd(datetime.datetime(1740,1,1,0,0,0), fmt='jd')
x_jul_1740, x_tick_time_1740= [], []
for x_tick in np.arange(1740, 2030+1, config.step): 
    t = datetime.datetime(x_tick,1,1,0,0,0)
    t_jd = julian.to_jd(t, fmt='jd')-julian_1700
    x_jul_1740.append(t_jd)
    x_tick_time_1740.append(x_tick)
data = np.loadtxt(file_location+r'\SN_m_tot_V2.0.txt')
sunspots_month = data[:, 3].reshape(-1,1)
year = np.array(data[:, 0],dtype=np.int)
month = np.array(data[:, 1],dtype=np.int)
day = np.ones_like(month)
date = pd.DataFrame({'year':year,'month':month,'day':day})
date = pd.to_datetime(date,format='%Y%m%d',errors='ignore')
jul_date_1740 = []
for i in np.arange(len(data)): 
    t = datetime.datetime.strptime(str(date[i]), '%Y-%m-%d %H:%M:%S')
    t_jd = julian.to_jd(t,fmt='jd')-julian.to_jd(datetime.datetime(1700,1,1,0,0,0), fmt='jd')
    jul_date_1740.append(t_jd)
np.savetxt(file_location+r'\sn_m_jul_date_1740.txt', jul_date_1740, fmt='%.1f', delimiter=' ')

# 1870
julian_1870 = julian.to_jd(datetime.datetime(1870,1,1,0,0,0), fmt='jd')
x_jul_1870, x_tick_time_1870= [], []
for x_tick in np.arange(1870, 2030+1, config.step): 
    t = datetime.datetime(x_tick,1,1,0,0,0)
    t_jd = julian.to_jd(t, fmt='jd')-julian_1870
    x_jul_1740.append(t_jd)
    x_tick_time_1740.append(x_tick)
data = np.loadtxt(file_location+r'\sa_merge_data.txt')
area = data[:,-2].reshape(-1,1)
year = np.array(data[:, 0],dtype=np.int)
month = np.array(data[:, 1],dtype=np.int)
day = np.array(data[:, 2],dtype=np.int)
date = pd.DataFrame({'year':year,'month':month,'day':day})
date = pd.to_datetime(date,format='%Y%m%d',errors='ignore')
# with open(file_location+r'\mean.txt', mode='w+', encoding='utf-8') as fout:
#     for y in np.arange(config.min_year, config.max_year+1):          
#         print('year:', y)
#         for m in np.arange(1, 12+1):
#             result = []
#             if y==config.min_year and m<5: continue
#             elif y==config.max_year and m>5: break
#             else:
#                 date_begin = datetime.datetime(y,m,1,0,0,0)
#                 date_end = datetime.datetime(y,m,1,0,0,0)+relativedelta(months=1)
#                 days = (date_end - date_begin).days
#                 flag_year = np.logical_and(date>=date_begin, date<date_end)
#                 # print('\t date_begin:', date_begin, 'date_end', date_end)
#                 lat = area[flag_year]
#                 days = (date_end - date_begin).days
#                 result.append(np.sum(lat)/days)
#             fout.write('{} {} 1 {}\n'.format(y, m, result)) 
# with open(file_location+r'\mean.txt', 'r') as f: content = f.read()
# with open(file_location+r'\mean.txt', 'w') as f:
#     content = content.replace('[', '')
#     content = content.replace(']', '')
#     content = content.replace(',', '')
#     content = content.replace('array(', '')
#     content = content.replace(')', '') 
#     content = content.replace('dtype=float32', '')
#     f.write(content)
data = np.loadtxt(file_location+r'\mean.txt')
area = data[:,-1]
year = np.array(data[:, 0],dtype=np.int)
month = np.array(data[:, 1],dtype=np.int)
day = np.array(data[:, 2],dtype=np.int)
date = pd.DataFrame({'year':year,'month':month,'day':day})
date = pd.to_datetime(date,format='%Y%m%d',errors='ignore')
jul_date_1870 = []
for i in np.arange(len(date)): 
    t = datetime.datetime.strptime(str(date[i]), '%Y-%m-%d %H:%M:%S')
    t_jd = julian.to_jd(t,fmt='jd')-julian_1700
    jul_date_1870.append(t_jd)


fig = plt.figure(figsize=(14, 4))
ax1 = fig.add_subplot(111)
ax1.grid(True, linestyle='--', linewidth=1)
# h1 = ax1.plot(jul_date_1700, sunspots_year, marker='s', markersize=6, 
#     color="dodgerblue", markerfacecolor='white', alpha=0.75, 
#     linewidth=2, label=u'年均太阳黑子数')
# h2 = ax1.plot(jul_date_1740, sunspots_month, marker='o', markersize=4, 
#     color="darkviolet", markerfacecolor='white', alpha=0.75, 
#     linewidth=2, label=u'月均太阳黑子数')
h1 = ax1.plot(jul_date_1700, sunspots_year, 
    color="dodgerblue", alpha=0.75, marker='s', 
    linewidth=2, label=u'年均太阳黑子数')
h2 = ax1.plot(jul_date_1740, sunspots_month, 
    color="darkviolet", alpha=0.75, marker='o', 
    linewidth=2, label=u'月均太阳黑子数')
ax1.set_ylabel('太阳黑子数', fontproperties=font, fontsize=18)
ax1.set_xlabel('日期', fontproperties=font, fontsize=18)
ax1.set_ylim(0, 510)
# ax1.set_xlim(0, 1000)
ax1.tick_params(labelsize=18)
ax1.set_xticks(x_jul_1700) 
ax1.set_xticklabels(x_tick_time_1700, fontproperties=font, fontsize=18, rotation=90)
[y_label.set_fontname('STSong') for y_label in ax1.get_yticklabels()]
ax1.set_xlim(min(x_jul_1700)-500, max(x_jul_1700)+500)
# ax1.legend(loc='upper left', prop=font)  
ax2 = ax1.twinx()
# h3 = ax2.plot(jul_date_1870, area, marker='^', markersize=6, 
#     color="indianred", markerfacecolor='white', alpha=0.75, 
#     linewidth=2, label=u'月均太阳黑子面积')
h3 = ax2.plot(jul_date_1870, area, 
    color="indianred", alpha=0.75, marker='^', 
    linewidth=2, label=u'月均太阳黑子面积')
ax2.tick_params(labelsize=18)
[y_label.set_fontname('STSong') for y_label in ax2.get_yticklabels()]
ax2.set_ylabel('太阳黑子面积', fontproperties=font, fontsize=18)
# ax2.legend(loc='upper right', prop=font)
ax2.set_ylim(0, 6100)
x_min = [1756,1766,1775,1786,1799,
         1812,1824,1834,1844,1855,
         1866,1877,1889,1901,1912,
         1922,1933,1943,1953,1964,
         1975,1985,1996,2008,2019]
for key, value in enumerate(x_min):
    jul = julian.to_jd(datetime.datetime(value+3,1,1,0,0,0), fmt='jd')-julian_1700
    plt.text(jul, 100, '{}'.format(1+key), fontproperties=font, fontsize=20, c='k')
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.yaxis.set_major_locator(plt.MultipleLocator(1000))
plt.legend([\
    Line2D([0],[0],color='dodgerblue',lw=1,markersize=6, marker='s'), 
    Line2D([0],[0],color='darkviolet',lw=1,markersize=4, marker='o'), 
    Line2D([0],[0],color='indianred',lw=1,markersize=6, marker='^')], 
    [u'年均太阳黑子数', u'月均太阳黑子数', u'月均太阳黑子面积'], prop=FontProperties(fname=r"C:\WINDOWS\Fonts\simsun.ttc", size=16), loc='upper center', ncol=3)
plt.tight_layout()
plt.savefig(r'fig1.jpg', dpi=100)
# plt.savefig(r'D:\sunspot\figure\fig1.jpg', dpi=100)
plt.show()


print('\ End gen_i_o.py')
