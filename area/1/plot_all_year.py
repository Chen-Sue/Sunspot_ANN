
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

from numpy.lib import utils
import config
from utils import checkpoints, series_to_supervised, save_data, \
    colorsys, color, ncolors

blocks = config.blocks 
features = config.features
n_splits = config.n_splits
split_ratio = config.split_ratio
epochs = config.epochs
learning_rate = config.learning_rate
file_name = config.file_name
file_location = config.file_location
fontcn = config.fontcn
fonten = config.fonten
font = {'family':'serif', 'color':'darkred', 'weight':'normal', 'size':12}
colors = list(map(lambda output_data: color(tuple(output_data)), ncolors(blocks)))
markers = ['.', ',', '+', 'x', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 
    'H', 'D', 'd', 'P', 'X', '1', '2', '3',]
lat_sep = config.lat_sep

start_time = time.time()

model = tf.keras.models.load_model(
    file_location+r'\model_lstm\{}.h5'.format(file_name))
data = np.loadtxt(file_location+r'\RGO_NOAA1874_2016\sunspot_year{}.txt'.format(
    config.blocks), delimiter=' ')

month = np.array(data[:, 12], dtype=np.int)
day = np.array(data[:, 13], dtype=np.int)
initial_julian = julian.to_jd(datetime.datetime(1870,1,1,0,0,0), fmt='jd')
end_julian = julian.to_jd(datetime.datetime(2020,1,1,0,0,0), fmt='jd')

# jul_date = [] # 横轴为儒略日
# for j in np.arange(features):
#     for i in np.arange(len(data)): 
#         names = globals()
#         names['year_'+str(j+1)] = np.array(data[:, j], dtype=np.int)
#         names['date_'+str(j+1)] = pd.DataFrame({'year':names['year_'+str(j+1)], 'month':month, 'day':day})
#         names['date_'+str(j+1)] = pd.to_datetime(names['date_'+str(j+1)], format='%Y%m%d', errors='ignore')
#         t = datetime.datetime.strptime(str(names['date_'+str(j+1)][i]), '%Y-%m-%d %H:%M:%S')
#         t_jd = julian.to_jd(t, fmt='jd')-initial_julian
#         jul_date.append(t_jd)
#         jul_date0.append(julian.from_jd(t_jd+initial_julian, fmt='jd'))
# np.savetxt(file_location+r'\RGO_NOAA1874_2016\jul_date.txt', 
#     jul_date, fmt='%.1f', delimiter=' ')

jul_date = np.loadtxt(file_location+r'\RGO_NOAA1874_2016\jul_date_year{}.txt'.format(
    config.blocks), delimiter=' ')
jul_date = jul_date.reshape(-1, len(data)).T


x_jul, x_tick_time = [], [] # 设置横轴标签
for x_tick in np.arange(1870, 2020+1, config.step): 
    t = datetime.datetime(x_tick,1,1,0,0,0)
    t_jd = julian.to_jd(t, fmt='jd')-initial_julian
    x_jul.append(t_jd)
    x_tick_time.append(x_tick)

input_data = np.array(data[:, 14:14+blocks*features]).reshape(-1, blocks*features)
output_data = np.array(data[:, 14+blocks*features:]).reshape(-1, features)  
print('\n  input_data=', input_data.shape, 'output_data=', output_data.shape)

x_scaler = MinMaxScaler(feature_range=(0, 1))
input_data = x_scaler.fit_transform(input_data).reshape(-1, blocks*features)
y_scaler = MinMaxScaler(feature_range=(0, 1)) 
output_data = y_scaler.fit_transform(output_data).reshape(-1, features)  

num1 = int(len(input_data) * split_ratio)
num2 = int(len(input_data) * (1-split_ratio)/2)
num = num1 + num2
x_train_val, y_train_val = input_data[:num, :], output_data[:num, :]
x_test, y_test = input_data[num:, :], output_data[num:, :]

x_train_val = np.expand_dims(x_train_val, axis=1)
x_test = np.expand_dims(x_test, axis=1)
input_data = np.expand_dims(input_data, axis=1)
print('\n\tshape: ', x_train_val.shape, y_train_val.shape, x_test.shape, y_test.shape)

y_train_val = y_scaler.inverse_transform(y_train_val)
y_train_val_pre = model.predict(x_train_val)
y_train_val_pre = y_scaler.inverse_transform(y_train_val_pre)
y_test = y_scaler.inverse_transform(y_test)
y_test_pre = model.predict(x_test)
y_test_pre = y_scaler.inverse_transform(y_test_pre)
y_all = np.concatenate((y_train_val, y_test), axis=0)
y_all_pre = np.concatenate((y_train_val_pre, y_test_pre), axis=0)

y_train_val_diff = (y_train_val_pre-y_train_val).reshape(-1, features)
y_test_diff = (y_test_pre-y_test).reshape(-1, features)
y_all_diff = np.concatenate((y_train_val_diff, y_test_diff), axis=0)

y_label = []
for key, value in enumerate(lat_sep):
    if key == 0 :
        y_label.append(r'<{:.2f}'.format(lat_sep[1]))
    elif key == len(lat_sep)-1:
        y_label.append(r'>={:.2f}'.format(lat_sep[-1]))
    else:
        y_label.append(r'[{:.2f},{:.2f})'.format(lat_sep[key], lat_sep[key+1]))

# y_lat = []
# y_lat_all = np.empty(shape=(0, features))
# for key, value in enumerate(lat_sep):
#     if value!=lat_sep[-1]:
#         y_lat.append((lat_sep[key]+lat_sep[key+1])/2)
#     else:
#         y_lat.append(-(lat_sep[0]+lat_sep[1])/2)
# y_lat = np.around(np.array(y_lat, dtype=np.float32), 2)
# for key in np.arange(len(y_all)):
#     y_lat_all = np.concatenate((y_lat_all, np.array(y_lat).reshape(1,-1)), axis=0)
# # y_lat_all = np.repeat(y_lat_all, features, axis=1)

norm = matplotlib.colors.Normalize(vmin=-3, 
    vmax=max(np.max(np.log10(y_all)), np.max(np.log10(y_all_pre))))

# for i in np.arange(features):
#     plt.figure(figsize=(6.5, 3.5))
#     plt.grid(True, linestyle='--', linewidth=1.0) 
#     plt.scatter(jul_date[:, i], y_all[:,i], c='', edgecolors='grey', 
#         marker='o', label='{}'.format('future '+str(i+1)+' year observed')) 
#     plt.scatter(jul_date[:, i], y_all_pre[:,i], c='dodgerblue', 
#         marker='*', label='{}'.format('future '+str(i+1)+' year predicted'))  
#     plt.title(f'All Set', fontproperties='Arial', fontsize=20, color='red')
#     plt.xlabel('Date', fontproperties='Arial', fontsize=16)         
#     plt.ylabel('Sunspots Number', fontproperties='Arial', fontsize=16)  
#     plt.xticks(x_jul, x_tick_time, size=14, rotation=60)
#     plt.yticks(size=14)
#     plt.xlim(0, end_julian-initial_julian)
#     plt.legend(prop=fonten, fontsize=9, ncol=1, loc='upper left') 
#     ax = plt.gca()
#     ax.spines['bottom'].set_linewidth(1.5)
#     ax.spines['left'].set_linewidth(1.5)
#     ax.spines['top'].set_linewidth(1.5)
#     ax.spines['right'].set_linewidth(1.5)
#     plt.tight_layout()
#     plt.show()

# print(len(jul_date[:len(y_train_val),:].reshape(-1,)), 
#     len(y_lat_all[:len(y_train_val),:].reshape(-1,)), 
#     len(y_train_val.reshape(-1,)))
# fig = plt.figure(figsize=(16, 9))
# plt.subplot(2,1,1)
# plt.grid(True, linestyle='--', linewidth=1.0) 
# plt.scatter(jul_date[:len(y_train_val),:].reshape(-1,), 
#     y_lat_all[:len(y_train_val),:].reshape(-1,), 
#     c=y_train_val.reshape(-1,), marker='|', s=250, cmap='viridis', norm=norm) 
# h1 = plt.contourf(y_train_val, cmap='viridis', norm=norm)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.title(f'Training and Validation Set (Observed)', fontproperties='Arial', \
#     fontsize=20, color='red')
# plt.xlabel('Date', fontproperties='Arial', fontsize=18)         
# plt.ylabel('Latitude (deg)', fontproperties='Arial', fontsize=18)  
# plt.xticks(x_jul, x_tick_time, size=14)
# plt.yticks(y_lat, y_label, size=14)
# plt.xlim(0, julian.to_jd(datetime.datetime(2020,1,1,0,0,0), fmt='jd')-initial_julian)
# plt.ylim(-90, 90)
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(1.5)
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['top'].set_linewidth(1.5)
# ax.spines['right'].set_linewidth(1.5)
# plt.tight_layout()
# plt.subplot(2,1,2)
# plt.grid(True, linestyle='--', linewidth=1.0) 
# plt.scatter(jul_date[:len(y_train_val),:].reshape(-1,), 
#     y_lat_all[:len(y_train_val),:].reshape(-1,), 
#     c=y_train_val_pre.reshape(-1,), marker='|', s=250, cmap='viridis', norm=norm) 
# h2 = plt.contourf(y_train_val_pre, cmap='viridis', norm=norm)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.title(f'Training and Validation Set (Predicted)', fontproperties='Arial', fontsize=20, color='red')
# plt.xlabel('Date', fontproperties='Arial', fontsize=18)         
# plt.ylabel('Latitude (deg)', fontproperties='Arial', fontsize=18)  
# plt.xticks(x_jul, x_tick_time, size=14)
# plt.yticks(y_lat, y_label, size=14)
# plt.xlim(0, julian.to_jd(datetime.datetime(2020,1,1,0,0,0), fmt='jd')-initial_julian)
# plt.ylim(-90, 90)
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(1.5)
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['top'].set_linewidth(1.5)
# ax.spines['right'].set_linewidth(1.5)
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
# cax = plt.axes([0.9, 0.1, 0.01, 0.8])
# cb = plt.colorbar(h1, cax=cax)
# cb.set_label('Sunspots Number', fontdict=font) 
# plt.show()

# plt.figure(figsize=(9, 9))
# plt.subplot(2,1,1)
# plt.grid(True, linestyle='--', linewidth=1.0) 
# plt.scatter(jul_date[len(y_train_val):,:].reshape(-1,), 
#     y_lat_all[len(y_train_val):,:].reshape(-1,), 
#     c=y_test.reshape(-1,), marker='|', s=250, cmap='viridis', norm=norm) 
# h1 = plt.contourf(y_test, cmap='viridis', norm=norm)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.title(f'Testing Set (Observed)', fontproperties='Arial', fontsize=20, color='red')
# plt.xlabel('Date', fontproperties='Arial', fontsize=18)         
# plt.ylabel('Latitude (deg)', fontproperties='Arial', fontsize=18)  
# plt.xticks(x_jul, x_tick_time, size=14)
# plt.yticks(y_lat, y_label, size=14)
# plt.xlim(julian.to_jd(datetime.datetime(2000,1,1,0,0,0), fmt='jd')-initial_julian, 
#     end_julian-initial_julian)
# plt.ylim(-90, 90)
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(1.5)
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['top'].set_linewidth(1.5)
# ax.spines['right'].set_linewidth(1.5)
# plt.tight_layout()
# plt.subplot(2,1,2)
# plt.grid(True, linestyle='--', linewidth=1.0) 
# plt.scatter(jul_date[len(y_train_val):,:].reshape(-1,), 
#     y_lat_all[len(y_train_val):,:].reshape(-1,), 
#     c=y_test_pre.reshape(-1,), marker='|', s=250, cmap='viridis', norm=norm) 
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.title(f'Testing Set (Predicted)', fontproperties='Arial', fontsize=20, color='red')
# plt.xlabel('Date', fontproperties='Arial', fontsize=18)         
# plt.ylabel('Latitude (deg)', fontproperties='Arial', fontsize=18)  
# plt.xticks(x_jul, x_tick_time, size=14)
# plt.yticks(y_lat, y_label, size=14)
# plt.xlim(julian.to_jd(datetime.datetime(2000,1,1,0,0,0), fmt='jd')-initial_julian, 
#     end_julian-initial_julian)
# plt.ylim(-90, 90)
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(1.5)
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['top'].set_linewidth(1.5)
# ax.spines['right'].set_linewidth(1.5)
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
# cax = plt.axes([0.9, 0.1, 0.01, 0.8])
# cb = plt.colorbar(h1, cax=cax)
# cb.set_label('Sunspots Number', fontdict=font) 
# plt.show()

fig = plt.figure(figsize=(16, 9))
plt.subplot(2,1,1)
plt.grid(True, linestyle='--', linewidth=1.0) 
plt.scatter(jul_date[:num,:], np.log10(y_train_val), 
    c='', edgecolors='grey', marker='s', s=15, cmap='viridis', label='observed') 
plt.scatter(jul_date[:num,:], np.log10(y_train_val_pre), 
    c='', edgecolors='dodgerblue', marker='s', s=15, cmap='viridis', label='predicted') 
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title(f'Training and Validation Set', fontproperties='Arial', \
    fontsize=20, color='red')
plt.xlabel('Date', fontproperties='Arial', fontsize=18)         
plt.ylabel('Sunspots Number (log10)', fontproperties='Arial', fontsize=18)  
plt.xticks(x_jul, x_tick_time, size=14)
plt.yticks(size=14)
plt.xlim(0,julian.to_jd(datetime.datetime(2020,1,1,0,0,0), fmt='jd')-initial_julian)
plt.legend(loc='upper left', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
plt.tight_layout()
plt.subplot(2,1,2)
plt.grid(True, linestyle='--', linewidth=1.0) 
plt.scatter(jul_date[:num,:], y_train_val, 
    c='', edgecolors='grey', marker='s', s=15, cmap='viridis', label='observed') 
plt.scatter(jul_date[:num,:], y_train_val_pre, 
    c='', edgecolors='dodgerblue', marker='s', s=15, cmap='viridis', label='predicted') 
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title(f'Training and Validation Set', fontproperties='Arial', fontsize=20, color='red')
plt.xlabel('Date', fontproperties='Arial', fontsize=18)         
plt.ylabel('Sunspots Number', fontproperties='Arial', fontsize=18)  
plt.xticks(x_jul, x_tick_time, size=14)
plt.yticks(size=14)
plt.xlim(0, julian.to_jd(datetime.datetime(2020,1,1,0,0,0), fmt='jd')-initial_julian)
plt.legend(loc='upper left', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
plt.tight_layout()
plt.show()

plt.figure(figsize=(16, 9))
plt.subplot(2,1,1)
plt.grid(True, linestyle='--', linewidth=1.0) 
plt.scatter(jul_date[num:,], np.log10(y_test), 
    c='', edgecolors='grey', marker='s', s=15, cmap='viridis', label='observed') 
plt.scatter(jul_date[num:,:], np.log10(y_test_pre), 
    c='', edgecolors='dodgerblue', marker='s', s=15, cmap='viridis', label='predicted') 
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title(f'Testing Set', fontproperties='Arial', fontsize=20, color='red')
plt.xlabel('Date', fontproperties='Arial', fontsize=18)         
plt.ylabel('Sunspots Number (log10)', fontproperties='Arial', fontsize=18)  
plt.xticks(x_jul, x_tick_time, size=14)
plt.yticks(size=14)
plt.xlim(julian.to_jd(datetime.datetime(1990,1,1,0,0,0), fmt='jd')-initial_julian, 
    end_julian-initial_julian)
plt.legend(loc='upper left', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
plt.tight_layout()
plt.subplot(2,1,2)
plt.grid(True, linestyle='--', linewidth=1.0) 
plt.scatter(jul_date[num:,:], y_test, 
    c='', edgecolors='grey', marker='s', s=15, cmap='viridis', label='observed') 
plt.scatter(jul_date[num:,:], y_test_pre, 
    c='', edgecolors='dodgerblue', marker='s', s=15, cmap='viridis', label='predicted') 
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title(f'Testing Set', fontproperties='Arial', fontsize=20, color='red')
plt.xlabel('Date', fontproperties='Arial', fontsize=18)         
plt.ylabel('Sunspots Number', fontproperties='Arial', fontsize=18)  
plt.xticks(x_jul, x_tick_time, size=14)
plt.yticks(size=14)
plt.xlim(julian.to_jd(datetime.datetime(1990,1,1,0,0,0), fmt='jd')-initial_julian, 
    end_julian-initial_julian)
plt.legend(loc='upper left', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
plt.grid(True, linestyle='--', linewidth=1.0)
plt.scatter(jul_date[:num,:], y_train_val, s=5, color='grey', label='Observed')
plt.scatter(jul_date[num:,:], y_test, s=5, color='grey')
plt.scatter(jul_date[:num,:], y_train_val_pre,  
    s=5, color='dodgerblue', label='Predicted (Training Set)')    
plt.scatter(jul_date[num:,:], y_test_pre, 
    s=5, color='indianred', label='Predicted (Testing Set)') 
plt.title(f'All Set', fontproperties='Arial', fontsize=20, color='red')
plt.xlabel('Date', fontproperties='Arial', fontsize=18)         
plt.ylabel('Total Sunspots Number', fontproperties='Arial', fontsize=18)  
plt.xticks(x_jul, x_tick_time, size=14)
plt.yticks(size=14)
plt.xlim(julian.to_jd(datetime.datetime(1870,1,1,0,0,0), fmt='jd')-initial_julian, 
    julian.to_jd(datetime.datetime(2020,1,1,0,0,0), fmt='jd')-initial_julian)
plt.legend(loc='best', prop=fonten, fontsize=15) 
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(6, 6))  
plt.grid(True, linestyle='--', linewidth=1.0)
plt.hist((y_test_diff/y_test).reshape(-1,), bins=21, range=(-1, 1), color='dodgerblue', stacked=True)
plt.xlabel('Predicted - Observed', fontdict=fonten, fontsize=18)  
plt.ylabel('Frequency', fontdict=fonten, fontsize=18)  
plt.title(r'Absolute Error (Testing Set)', fontdict=fonten, fontsize=20) 
plt.xticks(size=14)
plt.yticks(size=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
plt.tight_layout()     
plt.savefig(r'.\figure\{}-Frequency-test.png'.format(file_name))
plt.show()

fig = plt.figure(figsize=(6, 6))   
plt.grid(True, linestyle='--', linewidth=1.0)
plt.scatter(y_train_val, y_train_val_pre, marker='o', c='', edgecolors='dodgerblue', label='Training Set', s=10)    
plt.scatter(y_test, y_test_pre, marker='*', c='indianred', label='Testing Set', s=20) 
plt.plot(y_all, y_all, c='grey', linewidth=1.5)
plt.text(900, 900, 'y=x', fontdict=fonten, fontsize=16)
plt.xlabel('Observed', fontproperties='Arial', fontsize=18)  
plt.ylabel('Predicted', fontproperties='Arial', fontsize=18)  
plt.xticks(size=14)
plt.yticks(size=14)
plt.legend(loc='best', prop=fonten, fontsize=15)
plt.title(r'All Set', fontdict=fonten, fontsize=20, color='red')
plt.tight_layout() 
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.set_aspect(aspect='equal')
plt.savefig(r'.\figure\{}-PredictedObserved.png'.format(file_name))
plt.show()

fig = plt.figure(figsize=(10, 5))   
plt.subplot(1,1,1) 
plt.grid(True, linestyle='--', linewidth=1)
plt.scatter(jul_date[:len(y_train_val),:], y_train_val_diff, 
    label='Training Set', c='', edgecolors='dodgerblue', marker='o', s=10)
plt.scatter(jul_date[len(y_train_val):,:], y_test_diff, 
    label='Testing Set', c='', edgecolors='indianred', marker='o', s=10)
plt.xlabel('Date', fontproperties='Arial', fontsize=18)  
plt.ylabel('Predicted - Observed', fontproperties='Arial', fontsize=18)  
plt.title(r'All Set', fontdict=fonten, fontsize=20, color='red')
plt.xticks(x_jul, x_tick_time, size=14)
plt.yticks(size=14)
plt.xlim(0, end_julian-initial_julian)
plt.legend(loc='best', prop=fonten, fontsize=15)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
plt.tight_layout()
plt.savefig(r'.\figure\{}-DiffPredictedObserved.png'.format(file_name))
plt.show()


# plt.figure(figsize=(16, 8))
# plt.grid(True, linestyle='--', linewidth=1.0) 
# for i in np.arange(blocks):
#     plt.scatter(np.arange(len(y_test[:,i])), y_test[:,i], c=colors[i], alpha=0.2, \
#         marker=markers[i], label='{}'.format('block'+str(i+1)+'observed')) 
# for i in np.arange(blocks):
#     plt.scatter(np.arange(len(y_test[:,i])), y_test_pre[:,i], c=colors[i], \
#         marker=markers[i], label='{}'.format('block'+str(i+1)+'predicted'))  
# plt.title(f'Testing Set', fontproperties='Arial', fontsize=20, color='red')
# plt.xlabel('Date', fontproperties='Arial', fontsize=18)         
# plt.ylabel('Sunspots Number', fontproperties='Arial', fontsize=18)  
# plt.xticks(size=14)
# plt.yticks(size=14)
# plt.legend(prop=fonten, fontsize=9, ncol=2) 
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(1.5)
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['top'].set_linewidth(1.5)
# ax.spines['right'].set_linewidth(1.5)
# plt.tight_layout()
# plt.show()
