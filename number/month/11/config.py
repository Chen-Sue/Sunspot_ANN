
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # gpu
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
assert tf.__version__.startswith("2.") 
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608),
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608)])

file_location = os.getcwd()
seed = 12345
min_year, max_year = 1749, 2021
split_ratio = 0.6
sun_epoch = 11
in_cycle = 1

layer = 4
layer_size = 128 
epochs = 1*10**3
learning_rate = 1e-3
learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    decay_steps=epochs, initial_learning_rate=learning_rate, 
    decay_rate=0.999, staircase=False)
rate = 0.2
weight = 1e-6
file_name = f'len{in_cycle*sun_epoch:.0f}-split_ratio{split_ratio:.2f}-' +\
    f'layer{layer:.0f}-layer_size{layer_size:.0f}'

fontcn = {'family':'Simyou', 'size':18} 
fonten = {'family':'Arial', 'size':18}

step = 10

# model_name = "conv1d"
# model_name = "lstm"
model_name = "cnn_lstm"
