
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

config_font = {#'font.family':'serif', 'font.serif':['SimSun'], 
    'mathtext.fontset':'stix', 'font.size':16}
matplotlib.rcParams.update(config_font)
plt.rcParams['font.sans-serif']=['simsun'] 
plt.rcParams['axes.unicode_minus']=False 

blocks = config.blocks 
n_splits = config.n_splits
split_ratio = config.split_ratio
epochs = config.epochs
learning_rate = config.learning_rate
file_name = config.file_name
file_location = config.file_location
sun_epoch = config.sun_epoch
in_cycle = config.in_cycle
model_name = config.model_name
font = FontProperties(fname=r"C:\WINDOWS\Fonts\simsun.ttc", size=18)

# font = {'family':'serif', 'color':'darkred', 'weight':'normal', 'size':12}
colors = list(map(lambda output_data: color(tuple(output_data)), ncolors(blocks)))
markers = ['.', ',', '+', 'x', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 
    'H', 'D', 'd', 'P', 'X', '1', '2', '3',]
lat_sep = config.lat_sep

start_time = time.time()

# model = tf.keras.models.load_model(file_location+r'\model_lstm\{}.h5'.format(file_name))
model = tf.keras.models.load_model(file_location+\
    r'\{}\{}'.format(model_name, file_name)+\
    r'\00096-0.006532-0.076437-0.006995-0.079428.h5')

sunspot = np.loadtxt(file_location+r'\RGO_NOAA1874_2021\sa_train.txt')
print(sunspot.shape)
month = np.array(sunspot[:, 1], dtype=np.int)
day = np.array(sunspot[:, 2], dtype=np.int)
initial_julian = julian.to_jd(datetime.datetime(1870,1,1,0,0,0), fmt='jd')
end_julian = julian.to_jd(datetime.datetime(2030,1,1,0,0,0), fmt='jd')

jul_date = np.loadtxt(file_location+r'\RGO_NOAA1874_2021\sa_train_jul_date.txt').reshape(-1,1)
print(jul_date.shape)
# jul_date = jul_date.reshape(-1, len(sunspot)).T
jul_date = np.repeat(jul_date, blocks, axis=1)

x_jul, x_tick_time = [], [] # 设置横轴标签
for x_tick in np.arange(1870, 2030+1, config.step): 
    t = datetime.datetime(x_tick,1,1,0,0,0)
    t_jd = julian.to_jd(t, fmt='jd')-initial_julian
    x_jul.append(t_jd)
    x_tick_time.append(x_tick)

input_data = np.array(sunspot[:,3:3+blocks*in_cycle*sun_epoch*12])
output_data = np.array(sunspot[:,3+blocks*in_cycle*sun_epoch*12:])   
print('\n  input_data=', input_data.shape, 'output_data=', output_data.shape)

# from sklearn.decomposition import PCA
# pca = PCA(n_components=100)
# input_data = pca.fit_transform(input_data)

x_scaler = MinMaxScaler(feature_range=(0, 1))
input_data = x_scaler.fit_transform(input_data)
y_scaler = MinMaxScaler(feature_range=(0, 1)) 
output_data = y_scaler.fit_transform(output_data)

num1 = int(len(input_data) * split_ratio)
num2 = int(len(input_data) * (1-split_ratio)/2)
x_train, y_train = input_data[:num1, :], output_data[:num1, :]
x_val, y_val = input_data[num1:num1+num2, :], output_data[num1:num1+num2, :]
x_test, y_test = input_data[num1+num2:, :], output_data[num1+num2:, :]

x_train = np.expand_dims(x_train, axis=1)
x_val = np.expand_dims(x_val, axis=1)
x_test = np.expand_dims(x_test, axis=1)
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

y_train_diff = (y_train_pre-y_train).reshape(-1, blocks)
y_val_diff = (y_val_pre-y_val).reshape(-1, blocks)
y_test_diff = (y_test_pre-y_test).reshape(-1, blocks)
y_all_diff = np.concatenate((y_train_diff, y_val_diff, y_test_diff), axis=0)

y_all = np.concatenate((y_train, y_val, y_test), axis=0)
y_all_pre = np.concatenate((y_train_pre, y_val_pre, y_test_pre), axis=0)
y_train_val = np.concatenate((y_train, y_val), axis=0)
y_train_val_pre = np.concatenate((y_train_pre, y_val_pre), axis=0)

y_label = []
for key, value in enumerate(lat_sep):
    if key == 0 :
        y_label.append(r'({:.2f}N,90N]'.format(-lat_sep[1]))
    elif key == len(lat_sep)-1:
        y_label.append(r'[{:.2f}N,90N]'.format(lat_sep[-1]))
    elif key == len(lat_sep)//2:
        y_label.append(r'[{:.2f}S,{:.2f}N)'.format(lat_sep[key+1], -lat_sep[key]))
    else:
        if value < 0:
            y_label.append(r'[{:.2f}S,{:.2f}S)'.format(-lat_sep[key+1], -lat_sep[key]))
        else:
            y_label.append(r'[{:.2f}N,{:.2f}N)'.format(lat_sep[key], lat_sep[key+1]))

y_lat = []
y_lat_all = np.empty(shape=(0, blocks))
for key, value in enumerate(lat_sep):
    y_lat.append(key-len(lat_sep)//2)

for key in np.arange(len(y_all)):
    y_lat_all = np.concatenate((y_lat_all, np.array(y_lat).reshape(1,-1)), axis=0)
# y_lat_all = np.repeat(y_lat_all, features, axis=1)

# norm = matplotlib.colors.Normalize(vmin=0, vmax=600)
norm = matplotlib.colors.Normalize(vmin=0, vmax=600)

# for i in np.arange(blocks):
#     plt.figure(figsize=(6.5, 3.5))
#     plt.grid(True, linestyle='--', linewidth=1.0) 
#     plt.scatter(jul_date[:, 0], y_all[:,i], c='grey', 
#         marker='o', label='{}'.format('block'+str(i+1)+' observed')) 
#     plt.scatter(jul_date[:, 0], y_all_pre[:,i], c='dodgerblue', 
#         marker='*', label='{}'.format('block'+str(i+1)+' predicted'))  
#     plt.title(f'All Set', fontproperties='Arial', fontsize=20, color='red')
#     plt.xlabel('Date', fontproperties='Arial', fontsize=16)         
#     plt.ylabel('Sunspots Number', fontproperties='Arial', fontsize=16)  
#     plt.xticks(size=14)
#     plt.yticks(size=14)
#     plt.xticks(x_jul, x_tick_time, size=14, rotation=60)
#     plt.xlim(julian.to_jd(datetime.datetime(1870,1,1,0,0,0), fmt='jd')-initial_julian, 
#         julian.to_jd(datetime.datetime(2020,1,1,0,0,0), fmt='jd')-initial_julian)
#     plt.legend(prop=font, fontsize=9, ncol=1) 
#     ax = plt.gca()
#     ax.spines['bottom'].set_linewidth(1.5)
#     ax.spines['left'].set_linewidth(1.5)
#     ax.spines['top'].set_linewidth(1.5)
#     ax.spines['right'].set_linewidth(1.5)
#     plt.tight_layout()
#     plt.show()


fig = plt.figure(figsize=(14, 3.5))   
ax1 = plt.subplot(1,1,1) 
ax1.grid(True, axis='x', linestyle='--', linewidth=1.0) 
h1 = ax1.scatter(jul_date[:num1,:].reshape(-1,), 
    y_lat_all[:num1,:].reshape(-1,), 
    c=y_train.reshape(-1,), marker='|', s=200, cmap='CMRmap_r', norm=norm) 
ax1.scatter(jul_date[num1:num1+num2,:].reshape(-1,), 
    y_lat_all[num1:num1+num2,:].reshape(-1,), 
    c=y_val.reshape(-1,), marker='|', s=200, cmap='CMRmap_r', norm=norm) 
ax1.scatter(jul_date[num1+num2:,:].reshape(-1,), 
    y_lat_all[num1+num2:,:].reshape(-1,), 
    c=y_test.reshape(-1,), marker='|', s=200, cmap='CMRmap_r', norm=norm) 
# ax1.axvline(x=jul_date[num1,0], c="black", ls="--", lw=1.5)
# ax1.axvline(x=jul_date[num1+num2,0], c="black", ls="--", lw=1.5)
# ax1.text(julian.to_jd(datetime.datetime(1874,1,1,0,0,0), fmt='jd')-initial_julian,
    # len(lat_sep)//2-1, '(a)', fontproperties=font)
cbar = plt.colorbar(h1, extend='max')
cbar.ax.tick_params(labelsize=16)
ax1.set_xlabel(u'日期', fontproperties=font, fontsize=18)         
ax1.set_ylabel(u'纬度', fontproperties=font, fontsize=18)       
# ax.set_yticks(y_lat, y_label)
ax1.set_yticks(y_lat) 
ax1.set_yticklabels(y_label, fontdict={'fontsize':18})
ax1.tick_params(labelsize=14)
# ax.set_xticks(x_jul, x_tick_time, size=18)
ax1.set_xticks(x_jul) 
ax1.set_xticklabels(x_tick_time, fontdict={'fontsize':18})
x_min = [1878,1890,1902,\
    1913,1923,1933,1944,1954,1964,1976,1986,1996,2008,2020]
for key, value in enumerate(x_min):
    jul = julian.to_jd(datetime.datetime(value+3,1,1,0,0,0), fmt='jd')-initial_julian
    ax1.text(jul, -6.5, '{}'.format(12+key), fontproperties=font, fontsize=18)
[y_label.set_fontname('STSong') for y_label in ax1.get_yticklabels()]
[x_label.set_fontname('STSong') for x_label in ax1.get_xticklabels()]
ax1.set_xlim(0, julian.to_jd(datetime.datetime(2030,1,1,0,0,0), fmt='jd')-initial_julian)
ax1.set_ylim(-len(lat_sep)//2, len(lat_sep)//2+1)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
plt.tight_layout()
plt.savefig(r'D:\sunspot\figure\fig6(a).pdf', dpi=100, rasterized=True)
plt.savefig(r'D:\sunspot\figure\fig6(a).jpg', dpi=100, rasterized=True)
plt.show()

fig = plt.figure(figsize=(14, 3.5))   
ax2 = plt.subplot(1,1,1) 
ax2.grid(True, axis='x', linestyle='--', linewidth=1.0) 
h2 = ax2.scatter(jul_date[:num1,:].reshape(-1,), 
    y_lat_all[:num1,:].reshape(-1,), 
    c=y_train_pre.reshape(-1,), marker='|', s=200, cmap='CMRmap_r', norm=norm) 
ax2.scatter(jul_date[num1:num1+num2,:].reshape(-1,), 
    y_lat_all[num1:num1+num2,:].reshape(-1,), 
    c=y_val_pre.reshape(-1,), marker='|', s=200, cmap='CMRmap_r', norm=norm) 
ax2.scatter(jul_date[num1+num2:,:].reshape(-1,), 
    y_lat_all[num1+num2:,:].reshape(-1,), 
    c=y_test_pre.reshape(-1,), marker='|', s=200, cmap='CMRmap_r', norm=norm) 
# ax2.axvline(x=jul_date[num1,0], c="black", ls="--", lw=1.5)
# ax2.axvline(x=jul_date[num1+num2,0], c="black", ls="--", lw=1.5)
# ax2.text(julian.to_jd(datetime.datetime(1874,1,1,0,0,0), fmt='jd')-initial_julian,
#     len(lat_sep)//2-1, '(b)', fontproperties=font)
cbar = plt.colorbar(h2, extend='max')
cbar.ax.tick_params(labelsize=16)
# [x_label.set_fontname('STSong') for x_label in cbar.get_xticklabels()]
plt.tight_layout()
ax2.set_xlabel(u'日期', fontproperties=font, fontsize=18)         
ax2.set_ylabel(u'纬度', fontproperties=font, fontsize=18)       
# ax.set_yticks(y_lat, y_label)
ax2.set_yticks(y_lat) 
ax2.set_yticklabels(y_label, fontdict={'fontsize':18})
x_min = [1878,1890,1902,\
    1913,1923,1933,1944,1954,1964,1976,1986,1996,2008,2020]
for key, value in enumerate(x_min):
    jul = julian.to_jd(datetime.datetime(value+3,1,1,0,0,0), fmt='jd')-initial_julian
    ax1.text(jul, -6.5, '{}'.format(12+key), fontproperties=font, fontsize=18)
ax2.tick_params(labelsize=14)
# plt.xticks(x_jul, x_tick_time, size=18)
ax2.set_xticks(x_jul) 
ax2.set_xticklabels(x_tick_time, fontdict={'fontsize':18})
[x_label.set_fontname('STSong') for x_label in ax2.get_xticklabels()]
[y_label.set_fontname('STSong') for y_label in ax2.get_yticklabels()]
ax2.set_xlim(0, julian.to_jd(datetime.datetime(2030,1,1,0,0,0), fmt='jd')-initial_julian)
ax2.set_ylim(-len(lat_sep)//2, len(lat_sep)//2+1)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
plt.tight_layout()
plt.savefig(r'D:\sunspot\figure\fig6(b).pdf', dpi=100, rasterized=True)
plt.savefig(r'D:\sunspot\figure\fig6(b).jpg', dpi=100, rasterized=True)
plt.show()

x_min = [1877,1889,1901,1913,
         1922,1933,1943,1954,1965,
         1976,1986,1996,2010,2020]

fig, ax = plt.subplots(figsize=(14,4))
ax.grid(True, linestyle='--', linewidth=1.0) 
ax.plot(jul_date[:num1,0], np.sum(y_train, axis=1), markersize=4, linestyle='dotted',
    color="grey", markerfacecolor='white', alpha=0.75, 
    linewidth=2, marker='o', label='观测值')
ax.plot(jul_date[num1:num1+num2,0], np.sum(y_val, axis=1), markersize=4, linestyle='dotted', \
    color="grey", markerfacecolor='white', alpha=0.75, 
    linewidth=2, marker='s')
ax.plot(jul_date[-len(y_test):,0], np.sum(y_test, axis=1), markersize=4, linestyle='dotted', \
    color="grey", markerfacecolor='white', alpha=0.75, 
    linewidth=2, marker='^')
ax.plot(jul_date[:num1,0], np.sum(y_train_pre, axis=1), markersize=4, \
    color="dodgerblue", markerfacecolor='white', alpha=0.75, 
    linewidth=2, marker='o', label='预测值(训练集)')    
ax.plot(jul_date[num1:num1+num2,0], np.sum(y_val_pre, axis=1), markersize=4, \
    color="darkviolet", markerfacecolor='white', alpha=0.75, 
    linewidth=2, marker='s', label='预测值(验证集)')
ax.plot(jul_date[-len(y_test):,0], np.sum(y_test_pre, axis=1), markersize=4, \
    color="indianred", markerfacecolor='white', alpha=0.75, 
    linewidth=2, marker='^', label='预测值(测试集)')
# ax.scatter(jul_date_pre, output_data_pre, 
#     marker='*', c='k', label='2021年6月预测值', s=100)
# ax.annotate(r'{:.1f}'.format(output_data_pre[0,0]), 
#     xy=(jul_date_pre, output_data_pre), 
#     xytext=(-12, 20), textcoords='offset points', fontproperties=font, fontsize=18, 
#     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='k', lw=1, alpha=0.5))
for key, value in enumerate(x_min):
    jul = julian.to_jd(datetime.datetime(value+3,1,1,0,0,0), fmt='jd')-initial_julian
    ax.text(jul, 5, '{}'.format(12+key), fontproperties=font, fontsize=18)
ax.set_xlabel('日期', fontproperties=font, fontsize=18)         
ax.set_ylabel('太阳黑子面积(纬度加总)', fontproperties=font, fontsize=18)  
# ax.tick_params(size=18)
ax.set_xticks(x_jul) 
ax.set_xticklabels(x_tick_time, fontproperties=font, fontsize=18, rotation=0)
[y_label.set_fontname('STSong') for y_label in ax.get_yticklabels()]
ax.tick_params(labelsize=18)
ax.set_ylim(-100, 5600)
ax.legend(loc='upper center', ncol=5, prop=FontProperties(fname=r"C:\WINDOWS\Fonts\simsun.ttc", size=16)) 
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
plt.tight_layout()
plt.savefig(r'D:\sunspot\figure\fig7(b).pdf', dpi=100, rasterized=True)
plt.savefig(r'D:\sunspot\figure\fig7(b).jpg', dpi=100, rasterized=True)
plt.show()





# fig = plt.figure(figsize=(10, 5))   
# plt.subplot(1,1,1) 
# plt.grid(True, linestyle='--', linewidth=1)
# plt.scatter(jul_date[:num1,:], y_train_diff, 
#     label=u'训练集', c='dodgerblue', marker='*', s=20)
# plt.scatter(jul_date[num1:num1+num2,:], y_val_diff, 
#     label=u'验证集', c='darkviolet', marker='x', s=25)
# plt.scatter(jul_date[num1+num2:,:], y_test_diff, 
#     label=u'测试集', c='indianred', marker='+', s=30)
# plt.xlabel(u'日期', fontproperties=font, fontsize=18)  
# plt.ylabel(u'预测值-观测值', fontproperties=font, fontsize=18)  
# plt.xticks(x_jul, x_tick_time, size=14)
# plt.yticks(size=14)
# plt.xlim(julian.to_jd(datetime.datetime(1870,1,1,0,0,0), fmt='jd')-initial_julian, 
#     julian.to_jd(datetime.datetime(2030,1,1,0,0,0), fmt='jd')-initial_julian)
# plt.ylim(-500, 300)
# plt.legend(loc='upper left', prop=font, fontsize=16) 
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(1.5)
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['top'].set_linewidth(1.5)
# ax.spines['right'].set_linewidth(1.5)
# plt.tight_layout()
# plt.savefig(r'.\figure\{}-DiffPredictedObserved.png'.format(file_name))
# plt.show()


# fig = plt.figure(figsize=(6, 6))  
# fig.add_subplot(1,1,1)
# plt.grid(True, linestyle='--', linewidth=1.0)
# plt.hist(y_test_diff.reshape(-1,), bins=11, range=(-10, 10), 
#     color='dodgerblue', stacked=True)
# plt.xlabel('Predicted - Observed', fontdict=font, fontsize=18)  
# plt.ylabel('Frequency', fontdict=font, fontsize=18)  
# plt.title(r'Absolute Error (Testing Set)', fontdict=font, fontsize=20) 
# plt.xticks(size=14)
# plt.yticks(size=14)
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(1.5)
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['top'].set_linewidth(1.5)
# ax.spines['right'].set_linewidth(1.5)
# plt.tight_layout()     
# plt.savefig(r'.\figure\{}-Frequency-test.png'.format(file_name))
# plt.show()

# fig = plt.figure(figsize=(6, 6))   
# plt.grid(True, linestyle='--', linewidth=1.0)
# plt.scatter(y_train, y_train_pre, marker='o', c='dodgerblue', 
#     label='Training Set', s=10)    
# plt.scatter(y_val, y_val_pre, marker='s', c='darkviolet', 
#     label='Validation Set', s=10)    
# plt.scatter(y_test, y_test_pre, marker='*', c='indianred', 
#     label='Testing Set', s=20) 
# plt.plot(y_all, y_all, c='grey', linewidth=1.5)
# plt.text(-2, -2, 'y=x', fontdict=font, fontsize=16)
# plt.xlabel('Observed', fontproperties='Arial', fontsize=18)  
# plt.ylabel('Predicted', fontproperties='Arial', fontsize=18)  
# plt.xticks(size=14)
# plt.yticks(size=14)
# plt.legend(loc='best', prop=font, fontsize=15)
# plt.title(r'All Set', fontdict=font, fontsize=20, color='red')
# plt.tight_layout() 
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(1.5)
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['top'].set_linewidth(1.5)
# ax.spines['right'].set_linewidth(1.5)
# ax.set_aspect(aspect='equal')
# plt.savefig(r'.\figure\{}-PredictedObserved.png'.format(file_name))
# plt.show()