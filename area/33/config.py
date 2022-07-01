
import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # gpu
import tensorflow as tf
# assert tf.__version__.startswith("2.") 
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608),
#     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608)])
import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np
import math

file_location = os.getcwd()
seed = 12345
min_year, max_year = 1874, 2021
split_ratio = 0.6

input_window = 1 # 4
sun_epoch = 11
in_cycle = 1

# lat_sep = []
# for key, value in zip(np.arange(-5, 6, 1), np.arange(-45, 55, 10)): # arc sin ((2*N+1)sin1°）
#     temp = (2*key+1) * math.sin(5*math.pi/180)
#     temp = math.asin(temp)*180/math.pi
#     lat_sep.append(temp)
# lat_sep = np.around(np.array(lat_sep, dtype=np.float32), 2)
# blocks = len(lat_sep)
# print('间隔：', lat_sep, '区块数：', blocks)

lat_sep = []
# for key, value in enumerate(np.arange(-33, 33, 2)): # arc sin ((2*N+1)sin1°）
#     temp = value*math.sin(1/180*math.pi)
#     temp = math.asin(temp)*180/math.pi
#     lat_sep.append(temp)
# for key, value in enumerate(np.arange(-19, 19, 2)): # arc sin ((2*N+1)sin1°）
#     temp = value*math.sin(2/180*math.pi)
#     temp = math.asin(temp)*180/math.pi
#     lat_sep.append(temp)
# for key, value in enumerate(np.arange(-15, 15, 2)): # arc sin ((2*N+1)sin1°）
#     temp = value*math.sin(2.5/180*math.pi)
#     temp = math.asin(temp)*180/math.pi
#     lat_sep.append(temp)
for key, value in enumerate(np.arange(-13, 13, 2)): # arc sin ((2*N+1)sin1°）
    temp = value*math.sin(3/180*math.pi)
    temp = math.asin(temp)*180/math.pi
    lat_sep.append(temp)
lat_sep = np.around(np.array(lat_sep, dtype=np.float32), 2)
blocks = len(lat_sep)
print('间隔：', lat_sep, '区块数：', blocks)

n_splits = 5
layer = 4
layer_size = 128 
batch_size = 256
epochs = 1*10**3
learning_rate = 1e-3
learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    decay_steps=epochs, initial_learning_rate=learning_rate, 
    decay_rate=0.999, staircase=False)
rate = 0.2
weight = 1e-6
file_name = f'len{in_cycle*sun_epoch:.0f}-split_ratio{split_ratio:.2f}-' +\
    f'layer{layer:.0f}-layer_size{layer_size:.0f}'

fontcn = {'family':'SimSun', 'size':15} 
fonten = {'family':'Arial', 'size':15}

step = 10

# model_name = "conv1d"
# model_name = "lstm"
model_name = "cnn_lstm"