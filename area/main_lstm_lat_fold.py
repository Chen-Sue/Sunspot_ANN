
import config
from ANN import stateless_lstm
from utils import checkpoints, series_to_supervised, save_data
import time
import os
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608),
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608)])
import numpy as np
import pandas as pd
from tensorflow.keras import optimizers
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import julian
import datetime

file_name = config.file_name
file_location = config.file_location
split_ratio = config.split_ratio
blocks = config.blocks
features = config.features
n_splits = config.n_splits

epochs = config.epochs
learning_rate = config.learning_rate
fonten = config.fonten
fontcn = config.fontcn
lat_sep = config.lat_sep

start_time = time.time()
np.random.seed(config.seed)
tf.random.set_seed(config.seed)

if not os.path.exists(file_location+r'\figure'):
    os.mkdir(file_location+r'\figure')
if not os.path.exists(file_location+r'\loss'):
    os.mkdir(file_location+r'\loss')

data = np.loadtxt(file_location+r'\RGO_NOAA1874_2016\sunspot_lat{}.txt'.format(
    config.blocks), delimiter=' ')

initial_julian = julian.to_jd(datetime.datetime(1870,1,1,0,0,0), fmt='jd')
end_julian = julian.to_jd(datetime.datetime(2020,1,1,0,0,0), fmt='jd')
jul_date = np.loadtxt(file_location+r'\RGO_NOAA1874_2016\jul_date_lat{}.txt'.format(
    config.blocks), delimiter=' ')
jul_date = jul_date.reshape(-1, len(data)).T

x_jul, x_tick_time = [], []
for x_tick in np.arange(1870, 2020+1, config.step): 
    t = datetime.datetime(x_tick,1,1,0,0,0)
    t_jd = julian.to_jd(t, fmt='jd')-initial_julian
    x_jul.append(t_jd)
    x_tick_time.append(x_tick)

input_data = np.array(data[:, 3:3+blocks*features]).reshape(-1, blocks*features)
output_data = np.array(data[:, 3+blocks*features:]).reshape(-1, blocks)       
print('\n\tinput_data=', input_data.shape, 'output_data=', output_data.shape)

x_scaler = MinMaxScaler(feature_range=(0, 1))
input_data = x_scaler.fit_transform(input_data)
y_scaler = MinMaxScaler(feature_range=(0, 1)) 
output_data = y_scaler.fit_transform(output_data)

num1 = int(len(input_data) * split_ratio)
num2 = int(len(input_data) * (1-split_ratio)/2)
x_train, y_train = input_data[:num1, :], output_data[:num1, :]
x_val, y_val = input_data[num1:num1+num2, :], output_data[num1:num1+num2, :]
x_test, y_test = input_data[num1+num2:, :], output_data[num1+num2:, :]
x_train_val, y_train_val = input_data[:num1+num2, :], output_data[:num1+num2, :]

x_train = np.expand_dims(x_train, axis=1)
x_val = np.expand_dims(x_val, axis=1)
x_test = np.expand_dims(x_test, axis=1)
x_train_val = np.expand_dims(x_train_val, axis=1)
print('\n\tshape: ', x_train.shape, y_train.shape,
    x_val.shape, y_val.shape, x_test.shape, y_test.shape)

fold = TimeSeriesSplit(n_splits=n_splits, max_train_size=None)

fig = plt.figure(figsize=(10, 8))
for i, (train_index, val_index) in enumerate(fold.split(x_train_val, y_train_val)):
    l1 = plt.scatter(train_index, [i+1]*len(train_index), 
        c='dodgerblue', marker='_', lw=14)
    l2 = plt.scatter(val_index, [i+1]*len(val_index), 
        c='darkviolet', marker='_', lw=14)
    # plt.legend([Patch(color='dodgerblue'), Patch(color='indianred')],
    #   ['Training Set', 'Validation Set'],
    #   prop=fonten, loc=(0.55, 0.8), fontsize=16)
    plt.legend([l1, l2], ['Training set', 'Validation set'], \
        prop=fonten, loc='upper right', fontsize=16)
    plt.xlabel('Sample Index (Month)', fontproperties='Arial', fontsize=18)  
    plt.ylabel('CV Iteration', fontproperties='Arial', fontsize=18)  
    plt.title('Time Series Split', fontproperties='Arial', fontsize=20)    # Blocking
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.text(1.2, i+1.15, '{} | {}'.format(len(train_index), len(val_index)), 
        fontproperties='Arial', fontsize=16)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.set(ylim=[n_splits+0.9, 0.1])
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    plt.tight_layout()
plt.show()

val_loss_per_fold, val_mae_per_fold, val_rmse_per_fold = [], [], []
test_loss_per_fold, test_mae_per_fold, test_rmse_per_fold = [], [], []

model = stateless_lstm(x_train, output_node=blocks, 
        layer=config.layer, layer_size=config.layer_size, 
        rate=config.rate, weight=config.weight)
print('\tmodel.summary(): \n{} '.format(model.summary()))
print('\tlayer nums:', len(model.layers))       

for fold_no, (train_index, val_index) in enumerate(fold.split(x_train_val, y_train_val)):
    x_train, y_train = x_train_val[train_index], y_train_val[train_index]
    x_val, y_val = x_train_val[val_index], y_train_val[val_index]
    optimizer = optimizers.Adam(learning_rate=learning_rate) 
    model.compile(optimizer=optimizer, loss='mean_squared_error',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    print(f'\n\n\nTraining for Fold {fold_no+1} ...')
    history = model.fit(x_train, y_train, verbose=2, epochs=epochs, 
        batch_size=len(x_train), validation_data=(x_val, y_val), 
        callbacks=checkpoints(), shuffle=False)    
    val_loss, val_mae, val_rmse = model.evaluate(x_val, y_val,
        batch_size=len(x_val), verbose=2)
    test_loss, test_mae, test_rmse = model.evaluate(x_test, y_test,
        batch_size=len(x_test), verbose=2)
    val_loss_per_fold.append(val_loss)
    val_mae_per_fold.append(val_mae)
    val_rmse_per_fold.append(val_rmse)
    test_loss_per_fold.append(test_loss)
    test_mae_per_fold.append(test_mae)
    test_rmse_per_fold.append(test_rmse)
   
    y_train = y_scaler.inverse_transform(y_train)
    y_train_pre = model.predict(x_train)
    y_train_pre = y_scaler.inverse_transform(y_train_pre)
    y_val = y_scaler.inverse_transform(y_val)
    y_val_pre = model.predict(x_val)
    y_val_pre = y_scaler.inverse_transform(y_val_pre)    

    if fold_no+1 == config.n_splits:
        y_test = y_scaler.inverse_transform(y_test)
        y_test_pre = model.predict(x_test)
        y_test_pre = y_scaler.inverse_transform(y_test_pre)    
        val_loss, val_mae, val_rmse = model.evaluate(x_test, y_test,
            batch_size=len(x_test), verbose=2)
        print('Evaluate on testing set: ' + \
            'val_mse={0:.4f}, val_mae={1:.4f}, val_rmse={2:.4f}'.\
            format(val_loss, val_mae, val_rmse))    
        plt.figure(figsize=(10, 6))
        plt.grid(True, linestyle='--', linewidth=1.0) 
        for i in np.arange(blocks):
            if i==0:
                plt.scatter(jul_date[:len(y_train),i], y_train[:,i],
                    marker='o', c='', edgecolors='grey', label='Observed', s=10)
                plt.scatter(jul_date[len(y_train):len(y_train)+len(y_val),i], y_val[:,i],
                    marker='o', c='', edgecolors='grey', s=10)
                plt.scatter(jul_date[:len(y_train),i], y_train_pre[:,i],
                    marker='+', c='dodgerblue', label='Predicted (Training Set)', s=30)    
                plt.scatter(jul_date[len(y_train):len(y_train)+len(y_val),i], y_val_pre[:,i],
                    marker='x', c='darkviolet', label='Predicted (Validation Set)', s=25)
            else:
                plt.scatter(jul_date[:len(y_train),i], y_train[:,i],
                    marker='o', c='', edgecolors='grey', s=10)
                plt.scatter(jul_date[len(y_train):len(y_train)+len(y_val),i], y_val[:,i],
                    marker='o', c='', edgecolors='grey', s=10)
                plt.scatter(jul_date[:len(y_train),i], y_train_pre[:,i],
                    marker='+', c='dodgerblue', s=30)    
                plt.scatter(jul_date[len(y_train):len(y_train)+len(y_val),i], y_val_pre[:,i],
                    marker='x', c='darkviolet', s=25)
        plt.title('Training', fontproperties='Arial', fontsize=20, color='red')
        plt.xlabel('Sample Index', fontproperties='Arial', fontsize=18)         
        plt.ylabel('Sunspots Number', fontproperties='Arial', fontsize=18)  
        plt.xticks(x_jul, x_tick_time, size=14)
        plt.yticks(size=14)
        plt.legend(loc='best', prop=fonten, fontsize=15) 
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.grid(True, linestyle='--', linewidth=1.0) 
        for i in np.arange(blocks):
            if i==0:
                plt.scatter(jul_date[len(y_train)+len(y_val):,i], y_test[:,i],
                    marker='o', c='', edgecolors='grey', label='Observed', s=10)
                plt.scatter(jul_date[len(y_train)+len(y_val):,i], y_test_pre[:,i],
                    marker='*', c='indianred', label='Predicted', s=20)
            else:
                plt.scatter(jul_date[len(y_train)+len(y_val):,i], y_test[:,i],
                    marker='o', c='', edgecolors='grey', s=10)
                plt.scatter(jul_date[len(y_train)+len(y_val):,i], y_test_pre[:,i],
                    marker='*', c='indianred', s=20)
        plt.title('Testing Set', fontproperties='Arial', fontsize=20, color='red')
        plt.xlabel('Sample Index', fontproperties='Arial', fontsize=18)         
        plt.ylabel('Sunspots Number', fontproperties='Arial', fontsize=18)  
        plt.xticks(x_jul, x_tick_time, size=14)
        plt.yticks(size=14)
        plt.legend(loc='best', prop=fonten, fontsize=15) 
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.grid(True, linestyle='--', linewidth=1.0) 
        for i in np.arange(blocks):
            if i==0:
                plt.scatter(jul_date[:len(y_train),i], y_train[:,i],
                    marker='o', c='', edgecolors='grey', label='Observed', s=10)
                plt.scatter(jul_date[len(y_train):len(y_train)+len(y_val),i], y_val[:,i], 
                    marker='o', c='', edgecolors='grey', s=10)
                plt.scatter(jul_date[len(y_train)+len(y_val):,i], y_test[:,i], 
                    marker='o', c='', edgecolors='grey', s=10)
                plt.scatter(jul_date[:len(y_train),i], y_train_pre[:,i],
                    marker='+', c='dodgerblue', label='Predicted (Training Set)', s=30)    
                plt.scatter(jul_date[len(y_train):len(y_train)+len(y_val),i], y_val_pre[:,i], 
                    marker='x', c='darkviolet', label='Predicted (Validation Set)', s=25)
                plt.scatter(jul_date[len(y_train)+len(y_val):,i], y_test_pre[:,i], 
                    marker='*', c='indianred', label='Predicted (Testing Set)', s=20)
            else:
                plt.scatter(jul_date[:len(y_train),i], y_train[:,i],
                    marker='o', c='', edgecolors='grey', s=10)
                plt.scatter(jul_date[len(y_train):len(y_train)+len(y_val),i], y_val[:,i], 
                    marker='o', c='', edgecolors='grey', s=10)
                plt.scatter(jul_date[len(y_train)+len(y_val):,i], y_test[:,i], 
                    marker='o', c='', edgecolors='grey', s=10)
                plt.scatter(jul_date[:len(y_train),i], y_train_pre[:,i],
                    marker='+', c='dodgerblue', s=30)    
                plt.scatter(jul_date[len(y_train):len(y_train)+len(y_val),i], y_val_pre[:,i], 
                    marker='x', c='darkviolet', s=25)
                plt.scatter(jul_date[len(y_train)+len(y_val):,i], y_test_pre[:,i], 
                    marker='*', c='indianred', s=20)
        plt.title(f'All Set', fontproperties='Arial', fontsize=20, color='red')
        plt.xlabel('Sample Index', fontproperties='Arial', fontsize=18)         
        plt.ylabel('Sunspots Number', fontproperties='Arial', fontsize=18)  
        plt.xlim(julian.to_jd(datetime.datetime(1870,1,1,0,0,0), fmt='jd')-initial_julian, 
            julian.to_jd(datetime.datetime(1990,1,1,0,0,0), fmt='jd')-initial_julian)
        plt.xticks(x_jul, x_tick_time, size=14)
        plt.yticks(size=14)
        plt.legend(loc='best', prop=fonten, fontsize=15) 
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        plt.tight_layout()
        plt.show()

        for i in np.arange(blocks):
            plt.figure(figsize=(6, 5))
            plt.grid(True, linestyle='--', linewidth=1.0) 
            plt.scatter(jul_date[:len(y_train)][:,i], y_train[:,i],
                marker='o', c='', edgecolors='grey', label='Observed', s=10)
            plt.scatter(jul_date[len(y_train):len(y_train)+len(y_val)][:,i], y_val[:,i], 
                marker='o', c='', edgecolors='grey', s=10)
            plt.scatter(jul_date[len(y_train)+len(y_val):][:,i], y_test[:,i], 
                marker='o', c='', edgecolors='grey', s=10)
            plt.scatter(jul_date[:len(y_train)][:,i], y_train_pre[:,i],
                marker='+', c='dodgerblue', label='Predicted (Training Set)', s=30)    
            plt.scatter(jul_date[len(y_train):len(y_train)+len(y_val)][:,i], y_val_pre[:,i], 
                marker='x', c='darkviolet', label='Predicted (Validation Set)', s=25)
            plt.scatter(jul_date[len(y_train)+len(y_val):][:,i], y_test_pre[:,i], 
                marker='*', c='indianred', label='Predicted (Testing Set)', s=20)
            plt.title('All Set (block {})'.format(i+1), fontproperties='Arial', fontsize=20, color='red')
            plt.xlabel('Sample Index', fontproperties='Arial', fontsize=18)         
            plt.ylabel('Sunspots Number', fontproperties='Arial', fontsize=18)  
            plt.xlim(julian.to_jd(datetime.datetime(1870,1,1,0,0,0), fmt='jd')-initial_julian, 
                julian.to_jd(datetime.datetime(1990,1,1,0,0,0), fmt='jd')-initial_julian)
            plt.xticks(x_jul, x_tick_time, size=14, rotation=45)
            plt.yticks(size=14)
            plt.legend(loc='upper left', prop=fonten, fontsize=12) 
            ax = plt.gca()
            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['top'].set_linewidth(1.5)
            ax.spines['right'].set_linewidth(1.5)
            plt.tight_layout()
            plt.show()
    
model.save(file_location+r'\model_lstm\{}.h5'.format(file_name))

fig = plt.figure(figsize=(3.5, 3))  
plt.text(0.01, 0.85, 'Validation set Scores:', va='bottom', fontsize=14)
plt.text(0.01, 0.75, f' MSE:{np.mean(val_loss_per_fold):.4f}', va='bottom', fontsize=14)
plt.text(0.01, 0.65, f' MAE:{np.mean(val_mae_per_fold):.4f}(+/-{np.std(val_mae_per_fold):.4f})', 
    va='bottom', fontsize=14)
plt.text(0.01, 0.55, f' RMSE:{np.mean(val_rmse_per_fold):.4f}(+/-{np.std(val_rmse_per_fold):.4f})', 
    va='bottom', fontsize=14)
plt.text(0.01, 0.35, 'Testing set Scores:', va='bottom', fontsize=14)
plt.text(0.01, 0.25, f' MSE:{np.mean(test_loss_per_fold):.4f}', va='bottom', fontsize=14)
plt.text(0.01, 0.15, f' MAE:{np.mean(test_mae_per_fold):.4f}(+/-{np.std(test_mae_per_fold):.4f})', 
    va='bottom', fontsize=14)
plt.text(0.01, 0.05, f' RMSE:{np.mean(test_rmse_per_fold):.4f}(+/-{np.std(test_rmse_per_fold):.4f})', 
    va='bottom', fontsize=14)
ax = plt.gca()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
plt.tight_layout()
plt.show()

print((time.time()-start_time)/60, ' minutes')

# fig = plt.figure(figsize=(10, 8))
# for i, (train_index, val_index) in enumerate(fold.split(x_train_val, y_train_val)):
#     l1 = plt.scatter(train_index, [i+1]*len(train_index), 
#         c='dodgerblue', marker='_', lw=14)
#     l2 = plt.scatter(val_index, [i+1]*len(val_index), 
#         c='darkviolet', marker='_', lw=14)
#     # plt.legend([Patch(color='dodgerblue'), Patch(color='indianred')],
#     #   ['Training Set', 'Validation Set'],
#     #   prop=fonten, loc=(0.55, 0.8), fontsize=16)
#     plt.legend([l1, l2], ['Training set', 'Validation set'], \
#         prop=fonten, loc='upper right', fontsize=16)
#     plt.xlabel('Sample Index (Month)', fontproperties='Arial', fontsize=18)  
#     plt.ylabel('CV Iteration', fontproperties='Arial', fontsize=18)  
#     plt.title('Time Series Split', fontproperties='Arial', fontsize=20)    # Blocking
#     plt.xticks(size=14)
#     plt.yticks(size=14)
#     plt.text(1.2, i+1.15, '{} | {}'.format(len(train_index), len(val_index)), 
#         fontproperties='Arial', fontsize=16)
#     ax = plt.gca()
#     ax.spines['bottom'].set_linewidth(1.5)
#     ax.spines['left'].set_linewidth(1.5)
#     ax.spines['top'].set_linewidth(1.5)
#     ax.spines['right'].set_linewidth(1.5)
#     ax.set(ylim=[n_splits+0.9, 0.1])
#     ax.yaxis.set_major_locator(plt.MultipleLocator(1))
#     plt.tight_layout()
# plt.show()

# save_data(file_location=file_location+r'\loss', 
#     name='loss-{}-{}'.format(file_name, fold_no+1), value=history.history['loss'])    
# save_data(file_location=file_location+r'\loss', 
#     name='mae-{}-{}'.format(file_name, fold_no+1), value=history.history['mae'])    
# save_data(file_location=file_location+r'\loss', 
#     name='rmse-{}-{}'.format(file_name, fold_no+1), value=history.history['rmse'])  
# save_data(file_location=file_location+r'\loss', 
#     name='val_loss-{}-{}'.format(file_name, fold_no+1), value=history.history['val_loss'])    
# save_data(file_location=file_location+r'\loss', 
#     name='val_mae-{}-{}'.format(file_name, fold_no+1), value=history.history['val_mae'])    
# save_data(file_location=file_location+r'\loss', 
#     name='val_rmse-{}-{}'.format(file_name, fold_no+1), value=history.history['val_rmse'])  
