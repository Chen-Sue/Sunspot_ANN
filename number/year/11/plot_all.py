
from numpy.core.fromnumeric import reshape
from numpy.lib import utils
import config
from utils import checkpoints, series_to_supervised, save_data, \
    colorsys, color, ncolors

import time
import numpy as np
import pandas as pd
import math
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608),
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608)])
from sklearn.preprocessing import MinMaxScaler
import julian
import datetime
from matplotlib.font_manager import FontProperties
from dateutil.relativedelta import relativedelta

split_ratio = config.split_ratio
epochs = config.epochs
learning_rate = config.learning_rate
file_name = config.file_name
file_location = config.file_location
sun_epoch = config.sun_epoch
in_cycle = config.in_cycle
model_name = config.model_name

start_time = time.time()

initial_julian = julian.to_jd(datetime.datetime(1700,1,1,0,0,0),fmt='jd')
end_julian = julian.to_jd(datetime.datetime(2040,1,1,0,0,0),fmt='jd')
model = tf.keras.models.load_model(file_location+\
    r'\{}\{}'.format(model_name, file_name)+\
    # r'\00839-0.034034-0.127491-0.177210-0.026642-0.115822-0.154958.h5')

    # r'\00466-0.023245-0.152464-0.019761-0.140574.h5') # cnn 2
    # r'\00490-0.022467-0.149889-0.020106-0.141796.h5') # cnn 3
    # r'\00322-0.035125-0.187417-0.029152-0.170741.h5') # cnn 4

    # r'\00540-0.025405-0.159173-0.019910-0.140858.h5') # lstm 2
    # r'\00498-0.026972-0.163609-0.020232-0.141521.h5') # lstm 3
    # r'\00639-0.027143-0.163891-0.020780-0.143166.h5') # lstm 4


    # r'\00163-0.035548-0.188541-0.028597-0.169106.h5') # cnn-lstm 3
    # r'\00176-0.034190-0.184905-0.031249-0.176773.h5') # cnn-lstm 4
    r'\00177-0.024573-0.156757-0.019268-0.138808.h5') # cnn-lstm 4
    # r'\00247-0.022425-0.149749-0.021418-0.146350.h5') # cnn-lstm 5
    # r'\00209-0.034132-0.184748-0.032392-0.179977.h5') # cnn-lstm 5
    # r'\00145-0.045925-0.214300-0.040489-0.201220.h5') # cnn-lstm 5

    
    # r'\00186-0.040607-0.201512-0.039489-0.198719.h5') # cnn 22
    # r'\00203-0.025599-0.159997-0.040692-0.201723.h5') # cnn 33


    # r'\00499-0.022347-0.148124-0.024734-0.155970.h5') # lstm 22
    # r'\00575-0.016213-0.125208-0.020473-0.141194.h5') # lstm 33


    # r'\00172-0.017306-0.131553-0.023052-0.151830.h5') # cnn-lstm 22
    # r'\00154-0.019844-0.140868-0.023006-0.151676.h5') # cnn-lstm 33

font = FontProperties(fname=r"C:\WINDOWS\Fonts\simsun.ttc", size=18)

jul_date = np.loadtxt(
    file_location+r'\data1700_2021\sn_train_jul_date{}.txt'.format(in_cycle*sun_epoch))

x_jul, x_tick_time = [], []
for x_tick in np.arange(1700,2040+1,config.step): 
    t = datetime.datetime(x_tick,1,1,0,0,0)
    t_jd = julian.to_jd(t,fmt='jd')-initial_julian
    x_jul.append(t_jd)
    x_tick_time.append(x_tick)

sunspot = np.loadtxt(file_location+r'\data1700_2021\sn_train{}.txt'.format(in_cycle*sun_epoch))
input_data = np.array(sunspot[:,3:3+in_cycle*sun_epoch], dtype=np.float32)
output_data = np.array(sunspot[:,3+in_cycle*sun_epoch:], dtype=np.float32)      
print('\n\tinput_data=',input_data.shape,'output_data=',output_data.shape)

sunspot_pre = np.loadtxt(
    file_location+r'\data1700_2021\sn_pre{}.txt'.format(in_cycle*sun_epoch))
input_data_pre = np.array(
    sunspot_pre[:,3:3+sun_epoch], dtype=np.float32)
print('*'*50, input_data_pre)
jul_data0 = []
for i in np.arange(len(sunspot_pre)): 
    t = julian.to_jd(datetime.datetime(int(sunspot_pre[i,0])+10,1,1,0,0,0), fmt='jd') \
         -initial_julian
    jul_data0.append(t)
jul_date_pre = jul_data0

x_scaler = MinMaxScaler(feature_range=(0, 1))
input_data = x_scaler.fit_transform(input_data)
input_data_pre = x_scaler.fit_transform(input_data_pre)
y_scaler = MinMaxScaler(feature_range=(0, 1)) 
output_data = y_scaler.fit_transform(output_data)

num1 = int(len(input_data) * split_ratio)
num2 = int(len(input_data) * (1-split_ratio)/2)
x_train, y_train = input_data[:num1, :], output_data[:num1, :]
x_val, y_val = input_data[num1:num1+num2, :], output_data[num1:num1+num2, :]
x_test, y_test = input_data[num1+num2:, :], output_data[num1+num2:, :]

x_train = np.expand_dims(x_train,axis=1)
x_val = np.expand_dims(x_val,axis=1)
x_test = np.expand_dims(x_test,axis=1)
input_data_pre = np.expand_dims(input_data_pre,axis=1)
print('\n\tshape: ', x_train.shape, y_train.shape,
    x_val.shape, y_val.shape, x_test.shape, y_test.shape)

train_loss, train_rmse = model.evaluate(x_train,y_train,verbose=2)
val_loss, val_rmse = model.evaluate(x_val,y_val,verbose=2)
test_loss, test_rmse = model.evaluate(x_test,y_test,verbose=2)
print('\tEvaluate on train set:MSE={0:.4f},RMSE={1:.4f}'.\
    format(train_loss, train_rmse))  
print('\tEvaluate on validation set:MSE={0:.4f},RMSE={1:.4f}'.\
    format(val_loss, val_rmse))  
print('\tEvaluate on test set:MSE={0:.4f},RMSE={1:.4f}'.\
    format(test_loss, test_rmse))     



y_train = y_scaler.inverse_transform(y_train)
y_train_pre = model.predict(x_train)
y_train_pre = y_scaler.inverse_transform(y_train_pre)
y_val = y_scaler.inverse_transform(y_val)
y_val_pre = model.predict(x_val)
y_val_pre = y_scaler.inverse_transform(y_val_pre)    
y_test = y_scaler.inverse_transform(y_test)
y_test_pre = model.predict(x_test)
y_test_pre = y_scaler.inverse_transform(y_test_pre)  

output_data_pre = model.predict(input_data_pre)
output_data_pre = y_scaler.inverse_transform(output_data_pre)

y_train_diff = (y_train_pre-y_train)
y_val_diff = (y_val_pre-y_val)
y_test_diff = (y_test_pre-y_test)
y_all_diff = np.concatenate((y_train_diff, y_val_diff, y_test_diff), axis=0)

y_all = np.concatenate((y_train, y_val, y_test), axis=0)
y_all_pre = np.concatenate((y_train_pre, y_val_pre, y_test_pre), axis=0)
y_train_val = np.concatenate((y_train, y_val), axis=0)
y_train_val_pre = np.concatenate((y_train_pre, y_val_pre), axis=0)

fig, ax = plt.subplots(figsize=(14,4))
ax.grid(True, linestyle='--', linewidth=1.0) 
data = np.loadtxt(file_location+r'\data1700_2021\SN_y_tot_V2.0.txt')
sunspots =np.array(data[:, 1], dtype=np.float32)
year = np.array(data[:, 0], dtype=np.int)
month = np.ones_like(year)
day = np.ones_like(year)
date = pd.DataFrame({'year':year, 'month':month, 'day':day})
date = pd.to_datetime(date, format='%Y%m%d', errors='ignore')
jul_data = []
for i in np.arange(len(data)): 
    t = datetime.datetime.strptime(str(date[i]), '%Y-%m-%d %H:%M:%S')
    t_jd = julian.to_jd(t,fmt='jd')-initial_julian
    jul_data.append(t_jd)
plt.plot(jul_data, sunspots, markersize=4, linestyle='dotted', 
    color="grey", markerfacecolor='white', alpha=0.75, 
    linewidth=2, marker='o', label='观测值')
print(max(output_data_pre[-3, :]))
plt.plot(jul_date_pre, output_data_pre[-3, :], markersize=4, 
    color="indianred", markerfacecolor='white', alpha=0.75, 
    linewidth=2, marker='s', label='预测值')
ax.set_ylim(-10, 390)
x_min = [1756,1766,1775,1786,1799,
         1813,1824,1834,1844,1855,
         1866,1877,1888,1900,1911,
         1919,1930,1942,1952,1963,
         1975,1985,1995,2007,2019, 2029]
for key, value in enumerate(x_min):
    jul = julian.to_jd(datetime.datetime(value+3,1,1,0,0,0), fmt='jd')-initial_julian
    ax.text(jul, 5, '{}'.format(1+key), fontproperties=font, 
        fontsize=18, c='k')
ax.set_xlabel('日期', fontproperties=font, fontsize=18)         
ax.set_ylabel('年均太阳黑子数', fontproperties=font, fontsize=18)  
# ax.set_yticks(size=18)
# ax.set_xticks(x_jul, x_tick_time, size=18, rotation=45)
ax.set_xticks(x_jul) 
ax.set_xticklabels(x_tick_time, fontproperties=font, fontsize=18, rotation=90)
[y_label.set_fontname('STSong') for y_label in ax.get_yticklabels()]
ax.tick_params(labelsize=18)
ax.legend(loc='upper center', ncol=4, prop=FontProperties(fname=r"C:\WINDOWS\Fonts\simsun.ttc", size=16)) 
# plt.legend(loc='upper left', ncol=1, bbox_to_anchor=(1, 1), \
#     prop=FontProperties(fname=r"C:\WINDOWS\Fonts\simyou.ttf", size=16)) 
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
plt.tight_layout()
plt.savefig(r'D:\sunspot\figure\fig4(b).pdf', dpi=100)
plt.savefig(r'D:\sunspot\figure\fig4(b).jpg', dpi=100)
plt.show()

# plt.figure(figsize=(18, 5))
# plt.grid(True, linestyle='--', linewidth=1.0) 
# plt.plot(jul_date[:num1], y_train,
#     marker='*', c='grey', label='观测值 (训练集)')
# plt.plot(jul_date[num1:num1+num2], y_val, 
#     marker='x', c='grey', label='观测值 (验证集)')
# plt.plot(jul_date[num1+num2:], y_test, 
#     marker='+', c='grey', label='观测值 (测试集)')
# plt.plot(jul_date[:num1], y_train_pre,
#     marker='*', c='dodgerblue', label='预测值 (训练集)')    
# plt.plot(jul_date[num1:num1+num2], y_val_pre, 
#     marker='x', c='darkviolet', label='预测值 (验证集)')
# plt.plot(jul_date[num1+num2:], y_test_pre, 
#     marker='+', c='indianred', label='预测值 (测试集)')
# x_min = [1756,1766,1775,1786,1799,1813,1824,1834,1844,1855,1867,1878,1890,1902,\
#     1913,1923,1933,1944,1954,1964,1976,1986,1996,2008,2019]
# for key, value in enumerate(x_min):
#     jul = julian.to_jd(datetime.datetime(value+3,1,1,0,0,0), fmt='jd')-initial_julian
#     plt.text(jul, 25, '{}'.format(1+key), fontproperties=font, fontsize=18, c='k')
# plt.xlabel('日期', fontproperties=font, fontsize=20)         
# plt.ylabel('按年平均太阳黑子数(YSSN)', fontproperties=font, fontsize=20)  
# plt.yticks(size=14)
# plt.xticks(x_jul, x_tick_time, size=14, rotation=30)
# plt.legend(loc='upper left', prop=font, ncol=2) 
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(1.5)
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['top'].set_linewidth(1.5)
# ax.spines['right'].set_linewidth(1.5)
# plt.tight_layout()
# plt.show()

# fig = plt.figure(figsize=(6, 6))  
# fig.add_subplot(1,1,1)
# plt.grid(True, linestyle='--', linewidth=1.0)
# plt.hist(y_test_diff.reshape(-1,), bins=11, range=(-45, 45), 
#     color='dodgerblue', stacked=True, label=u'测试集')
# plt.xlabel(u'预测值-观测值', fontproperties=font, fontsize=20)  
# plt.ylabel(u'频度', fontproperties=font, fontsize=20) 
# plt.xticks(size=14)
# plt.yticks(size=14)
# plt.legend(loc='upper left', prop=font, fontsize=18)
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(1.5)
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['top'].set_linewidth(1.5)
# ax.spines['right'].set_linewidth(1.5)
# plt.tight_layout()     
# plt.show()

# fig = plt.figure(figsize=(6, 6))   
# plt.grid(True, linestyle='--', linewidth=1.0)
# plt.scatter(y_train, y_train_pre, marker='*', c='dodgerblue', label=u'训练集', s=20)    
# plt.scatter(y_val, y_val_pre, marker='x', c='darkviolet', label=u'验证集', s=25)    
# plt.scatter(y_test, y_test_pre, marker='+', c='indianred', label=u'测试集', s=30) 
# plt.plot(y_all, y_all, c='grey', linewidth=1.5)
# plt.text(255, 245, 'y=x', fontproperties=config.font, fontsize=18)
# plt.xlabel(u'观测值', fontproperties=font, fontsize=20)  
# plt.ylabel(u'预测值', fontproperties=font, fontsize=20)  
# plt.xticks(size=14)
# plt.yticks(size=14)
# plt.legend(loc='upper left', prop=font, fontsize=18)
# plt.tight_layout() 
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(1.5)
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['top'].set_linewidth(1.5)
# ax.spines['right'].set_linewidth(1.5)
# ax.set_aspect(aspect='equal')
# plt.show()

# fig = plt.figure(figsize=(18, 5))   
# plt.subplot(1,1,1) 
# plt.grid(True, linestyle='--', linewidth=1)
# plt.scatter(jul_date[:len(y_train)], y_train_diff, 
#     label=u'训练集', c='dodgerblue', marker='*', s=20)
# plt.scatter(jul_date[len(y_train):len(y_train)+len(y_val)], y_val_diff, 
#     label=u'验证集', c='darkviolet', marker='x', s=25)
# plt.scatter(jul_date[len(y_train_val):], y_test_diff, 
#     label=u'测试集', c='indianred', marker='+', s=30)
# plt.xlabel(u'日期', fontproperties=font, fontsize=20)  
# plt.ylabel(u'预测值-观测值', fontproperties=font, fontsize=20)  
# plt.xticks(x_jul, x_tick_time, size=14, rotation=30)
# plt.yticks(size=14)
# plt.legend(loc='upper left', prop=font, fontsize=18)
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(1.5)
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['top'].set_linewidth(1.5)
# ax.spines['right'].set_linewidth(1.5)
# plt.tight_layout()
# plt.show()
