
import config
from ANN import stateless_lstm,stateless_convlstm, cnn_lstm
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
sun_epoch = config.sun_epoch

epochs = config.epochs
learning_rate = config.learning_rate
batch_size = config.batch_size
fonten = config.fonten
fontcn = config.fontcn


start_time = time.time()
np.random.seed(config.seed)
tf.random.set_seed(config.seed)

if not os.path.exists(file_location+r'\figure'):
    os.mkdir(file_location+r'\figure')
if not os.path.exists(file_location+r'\loss'):
    os.mkdir(file_location+r'\loss')
initial_julian = julian.to_jd(datetime.datetime(1700,1,1,0,0,0),fmt='jd')
end_julian = julian.to_jd(datetime.datetime(2030,1,1,0,0,0),fmt='jd')

jul_date = np.loadtxt(file_location+r'\data1700_2021\sn_train_jul_date.txt').reshape(-1,1)

x_jul, x_tick_time = [], []
for x_tick in np.arange(1700,2030+1,config.step): 
    t = datetime.datetime(x_tick,1,1,0,0,0)
    t_jd = julian.to_jd(t,fmt='jd')-initial_julian
    x_jul.append(t_jd)
    x_tick_time.append(x_tick)

sunspot = np.loadtxt(file_location+r'\data1700_2021\sn_train.txt')
print(sunspot.shape)
input_data = np.array(sunspot[:,3:3+sun_epoch], dtype=np.float32)
output_data = np.array(sunspot[:,3+sun_epoch:], dtype=np.float32)      
print('\n\tinput_data=',input_data.shape,'output_data=',output_data.shape)

x_scaler = MinMaxScaler(feature_range=(0,1))
input_data = x_scaler.fit_transform(input_data)
y_scaler = MinMaxScaler(feature_range=(0,1)) 
output_data = y_scaler.fit_transform(output_data)

split_ratio2 = (1-split_ratio)/2
num1 = int(len(input_data) * split_ratio)
num2 = int(len(input_data) * split_ratio2)
x, y = input_data[:num1, :], output_data[:num1, :]
x_test, y_test = input_data[num1:, :], output_data[num1:, :]
x = np.expand_dims(x,axis=1)
x_test = np.expand_dims(x_test,axis=1)
# x_train = np.expand_dims(x_train,axis=2)
# x_val = np.expand_dims(x_val,axis=2)
# x_test = np.expand_dims(x_test,axis=2)
print('\n\tshape: ', x.shape, y.shape,
     x_test.shape, y_test.shape)

# model = stateless_lstm(x_train, output_node=output_data.shape[1], 
#         layer=config.layer, layer_size=config.layer_size, 
#         rate=config.rate, weight=config.weight)
# model = cnn_lstm(x_train, output_node=output_data.shape[1], 
#         layer=config.layer, layer_size=config.layer_size, 
#         rate=config.rate, weight=config.weight)
# print('\tmodel.summary(): \n{} '.format(model.summary()))
# print('\tlayer nums:', len(model.layers))       

fold = TimeSeriesSplit(n_splits=config.n_splits, max_train_size=None)

# fig = plt.figure(figsize=(10, 10))
# for i, (train_index, val_index) in enumerate(fold.split(x, y)):
#     l1 = plt.scatter(train_index, [i+1]*len(train_index), 
#         c='dodgerblue', marker='_', lw=14)
#     l2 = plt.scatter(val_index, [i+1]*len(val_index), 
#         c='darkviolet', marker='_', lw=14)
#     # plt.legend([Patch(color='dodgerblue'), Patch(color='indianred')],
#     #     ['Training Set', 'Validation Set'],
#         # prop=fonten, loc=(0.55, 0.8), fontsize=16)
#     plt.legend([l1, l2], ['Training set', 'Validation set'], \
#         prop=fonten, loc='upper right', fontsize=16)
#     plt.xlabel('Sample Index (Month)', fontproperties='Arial', fontsize=18)  
#     plt.ylabel('CV Iteration', fontproperties='Arial', fontsize=18)  
#     plt.title('Time Series Split', fontproperties='Arial', fontsize=20)    # Blocking
#     plt.xticks(size=14)
#     plt.yticks(size=14)
#     # plt.axvline(x=444, ls=':', c='green')
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
# plt.savefig(file_location+r'\figure\{}-CV.png'.format(filename))
# plt.show()

loss_per_fold = []
mae_per_fold = []
rmse_per_fold = []

# model = stateless_lstm(x, output_node=blocks, 
#         layer=config.layer, layer_size=config.layer_size, 
#         rate=config.rate, weight=config.weight)
# print('\tmodel.summary(): \n{} '.format(model.summary()))
# print('\tlayer nums:', len(model.layers))       

for fold_no, (train_index, val_index) in enumerate(fold.split(x, y)):
    model = cnn_lstm(x_train, output_node=output_data.shape[1], 
        layer=config.layer, layer_size=config.layer_size, 
        rate=config.rate, weight=config.weight)
    print('\tmodel.summary(): \n{} '.format(model.summary()))
    print('\tlayer nums:', len(model.layers))    
    optimizer = optimizers.Adam(learning_rate=learning_rate) 
    model.compile(optimizer=optimizer, loss='mean_squared_error',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    print(f'\n\n\n\nTraining for Fold {fold_no+1} ...')
    x_train, y_train = x[train_index], y[train_index]
    x_val, y_val = x[val_index], y[val_index]
    # history = model.fit(x_train, y_train, 
    #     verbose=2, epochs=epochs, 
    #     batch_size=batch_size, 
    #     validation_data=(x_val, y_val), 
    #     callbacks=checkpoints(), 
    #     shuffle=False)  
    train_size = (x_train.shape[0] // batch_size) * batch_size
    val_size = (x_val.shape[0] // batch_size) * batch_size
    test_size = (x_test.shape[0] // batch_size) * batch_size
    x_train, y_train = x_train[0:train_size,:,:], y_train[0:train_size,:]
    x_val, y_val = x_val[0:val_size,:,:], y_val[0:val_size,:]
    x_test, y_test = x_test[0:test_size,:,:], y_test[0:test_size,:]
    for i in np.arange(100):
        print("\n\n Epoch {:d}/{:d}".format(i+1, 100))
        history = model.fit(x_train, y_train, 
            validation_data=(x_val, y_val),
            batch_size=batch_size, verbose=2, epochs=1, shuffle=False,
            callbacks=checkpoints()
            )
        model.reset_states()

    val_loss, val_mae, val_rmse = model.evaluate(x_val, y_val,
        batch_size=batch_size, verbose=2)
    print('Evaluate on validation data: ' + \
        'val_mse={0:.2f}%, val_mae={1:.2f}%, val_rmse={2:.2f}%'.\
        format(val_loss*100, val_mae*100, val_rmse*100))    
    loss_per_fold.append(val_loss)
    mae_per_fold.append(val_mae)
    rmse_per_fold.append(val_rmse)

    y_train = y_scaler.inverse_transform(y_train)
    y_train_pre = model.predict(x_train)
    y_train_pre = y_scaler.inverse_transform(y_train_pre)
    y_val = y_scaler.inverse_transform(y_val)
    y_val_pre = model.predict(x_val)
    y_val_pre = y_scaler.inverse_transform(y_val_pre)  
    
    len_train = len(y_train.reshape(-1, ))
    len_val = len(y_val.reshape(-1, ))   
    len_test = len(y_test.reshape(-1, ))    

    # if fold_no+1 != config.n_splits:
    #     plt.figure(figsize=(10, 6))
    #     plt.grid(True, linestyle='--', linewidth=1.0) 
    #     plt.scatter(np.arange(len_train), y_train, 
    #         marker='o', c='', edgecolors='grey', label='Observed', s=10)
    #     plt.scatter(np.arange(len_train, len_train+len_val, 1), y_val, 
    #         marker='o', c='', edgecolors='grey', s=10)
    #     plt.scatter(np.arange(len_train), y_train_pre, 
    #         marker='+', c='dodgerblue', label='Predicted (Training Set)', s=30)    
    #     plt.scatter(np.arange(len_train, len_train+len_val, 1), y_val_pre, 
    #         marker='x', c='darkviolet', label='Predicted (Validation Set)', s=25)
    #     plt.title(f'Training for Fold {fold_no+1}', fontproperties='Arial', fontsize=20)
    #     plt.xlabel('Sample Index', fontproperties='Arial', fontsize=18)         
    #     plt.ylabel('Predicted max. Magnitude', fontproperties='Arial', fontsize=18)  
    #     plt.xticks(size=14)
    #     plt.yticks(size=14)
    #     plt.ylim(2.5, 8.5)
    #     plt.legend(loc='best', prop=fonten, fontsize=15) 
    #     ax = plt.gca()
    #     ax.spines['bottom'].set_linewidth(1.5)
    #     ax.spines['left'].set_linewidth(1.5)
    #     ax.spines['top'].set_linewidth(1.5)
    #     ax.spines['right'].set_linewidth(1.5)
    #     plt.savefig(file_location+r'.\figure\{}-fold{}.png'.format(filename, fold_no+1))
        # plt.show()
        # elif fold_no+1 == config.n_splits:
    if fold_no+1 == config.n_splits:
        y_test = y_scaler.inverse_transform(y_test)
        y_test_pre = model.predict(x_test)
        y_test_pre = y_scaler.inverse_transform(y_test_pre)   
        plt.figure(figsize=(10, 6))
        plt.grid(True, linestyle='--', linewidth=1.0) 
        plt.scatter(np.arange(len_train), y_train, 
            marker='o', c='', edgecolors='grey', label='Observed', s=10)
        plt.scatter(np.arange(len_train, len_train+len_val, 1), y_val, 
            marker='o', c='', edgecolors='grey', s=10)
        plt.scatter(np.arange(len_train), y_train_pre, 
            marker='+', c='dodgerblue', label='Predicted (Training Set)', s=30)    
        plt.scatter(np.arange(len_train, len_train+len_val, 1), y_val_pre, 
            marker='x', c='darkviolet', label='Predicted (Validation Set)', s=25)
        plt.title(f'Training for Fold {fold_no+1}', fontproperties='Arial', fontsize=20, color='red')
        plt.xlabel('Sample Index', fontproperties='Arial', fontsize=18)         
        plt.ylabel('Predicted max. Magnitude', fontproperties='Arial', fontsize=18)  
        plt.xticks(size=14)
        plt.yticks(size=14)
        # plt.ylim(2.5, 8.5)
        plt.legend(loc='best', prop=fonten, fontsize=15) 
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        # plt.savefig(file_location+r'.\figure\{}-fold{}.png'.format(filename, fold_no+1))
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.grid(True, linestyle='--', linewidth=1.0) 
        plt.scatter(np.arange(len_train+len_val, len_train+len_val+len_test), y_test, 
            marker='o', c='', edgecolors='grey', label='Observed', s=10)
        plt.scatter(np.arange(len_train+len_val, len_train+len_val+len_test), y_test_pre, 
            marker='*', c='indianred', label='Predicted (Testing Set)', s=20)
        plt.title('Testing Set', fontproperties='Arial', fontsize=20, color='red')
        plt.xlabel('Sample Index', fontproperties='Arial', fontsize=18)         
        plt.ylabel('Predicted max. Magnitude', fontproperties='Arial', fontsize=18)  
        plt.xticks(size=14)
        plt.yticks(size=14)
        # plt.ylim(2.5, 8.5)
        plt.legend(loc='best', prop=fonten, fontsize=15) 
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        # plt.savefig(file_location+r'.\figure\{}-fold{}.png'.format(filename, fold_no+1))
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.grid(True, linestyle='--', linewidth=1.0) 
        plt.scatter(np.arange(len_train), y_train, 
            marker='o', c='', edgecolors='grey', label='Observed', s=10)
        plt.scatter(np.arange(len_train, len_train+len_val, 1), y_val, 
            marker='o', c='', edgecolors='grey', s=10)
        plt.scatter(np.arange(len_train+len_val, len_train+len_val+len_test, 1), y_test, 
            marker='o', c='', edgecolors='grey', s=10)
        plt.scatter(np.arange(len_train), y_train_pre, 
            marker='+', c='dodgerblue', label='Predicted (Training Set)', s=30)    
        plt.scatter(np.arange(len_train, len_train+len_val, 1), y_val_pre, 
            marker='x', c='darkviolet', label='Predicted (Validation Set)', s=25)
        plt.scatter(np.arange(len_train+len_val, len_train+len_val+len_test, 1), y_test_pre, 
            marker='*', c='indianred', label='Predicted (Testing Set)', s=20)
        plt.title(f'All Set', fontproperties='Arial', fontsize=20, color='red')
        plt.xlabel('Sample Index', fontproperties='Arial', fontsize=18)         
        plt.ylabel('Predicted max. Magnitude', fontproperties='Arial', fontsize=18)  
        plt.xticks(size=14)
        plt.yticks(size=14)
        # plt.ylim(2.5, 8.5)
        plt.legend(loc='best', prop=fonten, fontsize=15) 
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        # plt.savefig(file_location+r'\figure\{}-fold{}.png'.format(filename, fold_no+1))
        plt.show()

    # if not os.path.exists(file_location+r'\loss'):
    #     os.mkdir(file_location+r'\loss')
    # save_data(file_location=file_location+r'\loss', 
    #     name='loss-{}-{}'.format(filename, fold_no+1), value=history.history['loss'])    
    # save_data(file_location=file_location+r'\loss', 
    #     name='mae-{}-{}'.format(filename, fold_no+1), value=history.history['mae'])    
    # save_data(file_location=file_location+r'\loss', 
    #     name='rmse-{}-{}'.format(filename, fold_no+1), value=history.history['rmse'])  
    # save_data(file_location=file_location+r'\loss', 
    #     name='val_loss-{}-{}'.format(filename, fold_no+1), value=history.history['val_loss'])    
    # save_data(file_location=file_location+r'\loss', 
    #     name='val_mae-{}-{}'.format(filename, fold_no+1), value=history.history['val_mae'])    
    # save_data(file_location=file_location+r'\loss', 
    #     name='val_rmse-{}-{}'.format(filename, fold_no+1), value=history.history['val_rmse'])  

model.save(file_location+r'\model_lstm\{}.h5'.format(filename))

fig = plt.figure(figsize=(4, 3))  
plt.text(0.01, 0.7, 'Average scores for all folds:', va='bottom', fontsize=14)
plt.text(0.01, 0.5, f' MSE:{np.mean(loss_per_fold):.4f}', va='bottom', fontsize=14)
plt.text(0.01, 0.3, f' MAE:{np.mean(mae_per_fold):.4f}(+/-{np.std(mae_per_fold):.4f})', 
    va='bottom', fontsize=14)
plt.text(0.01, 0.1, f' RMSE:{np.mean(rmse_per_fold):.4f}(+/-{np.std(rmse_per_fold):.4f})', 
    va='bottom', fontsize=14)
ax = plt.gca()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
# plt.savefig(file_location+r'\figure\error-{}.png'.format(filename))
plt.show()

print((time.time()-start_time)/60, ' minutes')

# optimizer = optimizers.Adam(learning_rate=learning_rate) 
# model.compile(optimizer=optimizer, loss='mean_squared_error',
#     metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
# history = model.fit(x_train, y_train, verbose=2, epochs=epochs, 
#     batch_size=len(x_train), validation_data=(x_val, y_val), 
#     callbacks=checkpoints(), shuffle=False)    

# model.save(file_location+r'\model_lstm\{}.h5'.format(file_name))

# val_loss, val_mae, val_rmse = model.evaluate(x_val,y_val,verbose=2)
# test_loss, test_mae, test_rmse = model.evaluate(x_test,y_test,verbose=2)
# print('\tEvaluate on validation set:MSE={0:.4f},MAE={1:.4f},RMSE={2:.4f}'.\
#     format(val_loss, val_mae, val_rmse))  
# print('\tEvaluate on test set:MSE={0:.4f},MAE={1:.4f},RMSE={2:.4f}'.\
#     format(test_loss, test_mae, test_rmse))     

# val_loss_per_fold.append(val_loss)
# val_mae_per_fold.append(val_mae)
# val_rmse_per_fold.append(val_rmse)
# test_loss_per_fold.append(test_loss)
# test_mae_per_fold.append(test_mae)
# test_rmse_per_fold.append(test_rmse)

# y_train = y_scaler.inverse_transform(y_train)
# y_train_pre = model.predict(x_train)
# y_train_pre = y_scaler.inverse_transform(y_train_pre)
# y_val = y_scaler.inverse_transform(y_val)
# y_val_pre = model.predict(x_val)
# y_val_pre = y_scaler.inverse_transform(y_val_pre)    
# y_test = y_scaler.inverse_transform(y_test)
# y_test_pre = model.predict(x_test)
# y_test_pre = y_scaler.inverse_transform(y_test_pre)  

# plt.figure(figsize=(6, 5))
# plt.grid(True, linestyle='--', linewidth=1.0) 
# plt.scatter(jul_date[num1+num2:], y_test, 
#     marker='+', c='grey', label='观测值 (测试集)', s=30)
# plt.scatter(jul_date[num1+num2:], y_test_pre, 
#     marker='+', c='indianred', label='预测值 (测试集)', s=30)
# plt.xlabel('日期', fontproperties=fontcn, fontsize=20)         
# plt.ylabel('太阳黑子数', fontproperties=fontcn, fontsize=20)   
# plt.xticks(x_jul, x_tick_time, size=16)
# plt.yticks(size=16)
# plt.xlim(julian.to_jd(datetime.datetime(1960,1,1,0,0,0), fmt='jd')-initial_julian, 
#     julian.to_jd(datetime.datetime(2030,1,1,0,0,0), fmt='jd')-initial_julian)
# plt.legend(loc='upper left', prop=fontcn, fontsize=14, ncol=1) 
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(1.5)
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['top'].set_linewidth(1.5)
# ax.spines['right'].set_linewidth(1.5)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(18, 5))
# plt.grid(True, linestyle='--', linewidth=1.0) 
# plt.scatter(jul_date[:num1], y_train,
#     marker='*', c='grey', label='观测值 (训练集)', s=20)
# plt.scatter(jul_date[num1:num1+num2], y_val, 
#     marker='x', c='grey', label='观测值 (验证集)', s=25)
# plt.scatter(jul_date[num1+num2:], y_test, 
#     marker='+', c='grey', label='观测值 (测试集)', s=30)
# plt.scatter(jul_date[:num1], y_train_pre,
#     marker='*', c='dodgerblue', label='预测值 (训练集)', s=20)    
# plt.scatter(jul_date[num1:num1+num2], y_val_pre, 
#     marker='x', c='darkviolet', label='预测值 (验证集)', s=25)
# plt.scatter(jul_date[num1+num2:], y_test_pre, 
#     marker='+', c='indianred', label='预测值 (测试集)', s=30)
# plt.xlabel('日期', fontproperties=fontcn, fontsize=20)         
# plt.ylabel('太阳黑子数', fontproperties=fontcn, fontsize=20)  
# plt.xlim(julian.to_jd(datetime.datetime(1700,1,1,0,0,0), fmt='jd')-initial_julian, 
#     julian.to_jd(datetime.datetime(2030,1,1,0,0,0), fmt='jd')-initial_julian)
# plt.xticks(x_jul, x_tick_time, size=14)
# plt.yticks(size=16)
# plt.legend(loc='upper left', prop=fontcn, fontsize=14, ncol=2) 
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(1.5)
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['top'].set_linewidth(1.5)
# ax.spines['right'].set_linewidth(1.5)
# plt.tight_layout()
# plt.show()
    
# fig = plt.figure(figsize=(3.5, 3))  
# plt.text(0.01, 0.85, 'Validation set Scores:', va='bottom', fontsize=14)
# plt.text(0.01, 0.75, f' MSE:{np.mean(val_loss_per_fold):.4f}', va='bottom', fontsize=14)
# plt.text(0.01, 0.65, f' MAE:{np.mean(val_mae_per_fold):.4f}(+/-{np.std(val_mae_per_fold):.4f})', 
#     va='bottom', fontsize=14)
# plt.text(0.01, 0.55, f' RMSE:{np.mean(val_rmse_per_fold):.4f}(+/-{np.std(val_rmse_per_fold):.4f})', 
#     va='bottom', fontsize=14)
# plt.text(0.01, 0.35, 'Testing set Scores:', va='bottom', fontsize=14)
# plt.text(0.01, 0.25, f' MSE:{np.mean(test_loss_per_fold):.4f}', va='bottom', fontsize=14)
# plt.text(0.01, 0.15, f' MAE:{np.mean(test_mae_per_fold):.4f}(+/-{np.std(test_mae_per_fold):.4f})', 
#     va='bottom', fontsize=14)
# plt.text(0.01, 0.05, f' RMSE:{np.mean(test_rmse_per_fold):.4f}(+/-{np.std(test_rmse_per_fold):.4f})', 
#     va='bottom', fontsize=14)
# ax = plt.gca()
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# plt.tight_layout()
# plt.show()

# print((time.time()-start_time)/60, ' minutes')


# save_data(file_location=file_location+r'\loss', 
#     name='loss-{}'.format(file_name), value=history.history['loss'])    
# save_data(file_location=file_location+r'\loss', 
#     name='mae-{}'.format(file_name), value=history.history['mae'])    
# save_data(file_location=file_location+r'\loss', 
#     name='rmse-{}'.format(file_name), value=history.history['rmse'])  
# save_data(file_location=file_location+r'\loss', 
#     name='val_loss-{}'.format(file_name), value=history.history['val_loss'])    
# save_data(file_location=file_location+r'\loss', 
#     name='val_mae-{}'.format(file_name), value=history.history['val_mae'])    
# save_data(file_location=file_location+r'\loss', 
#     name='val_rmse-{}'.format(file_name), value=history.history['val_rmse'])  
