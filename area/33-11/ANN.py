import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608),
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608)])
from tensorflow.keras import optimizers, layers, initializers, regularizers
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import ConvLSTM2D

import config

def cnn_lstm(x, output_node, layer=4, layer_size=128, rate=0.20, weight=1e-6):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(x.shape[1], x.shape[2])))
    # model.add(layers.Dropout(rate))
    model.add(layers.Conv1D(filters=64, kernel_size=2, strides=1, 
        kernel_regularizer=regularizers.l2(weight), 
        padding='valid', activation='relu', data_format="channels_first"))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.LSTM(128, 
        # kernel_regularizer=regularizers.l2(weight), 
        activation='relu', return_sequences=True))
    # model.add(layers.Dropout(rate))
    model.add(layers.LSTM(64, 
        # kernel_regularizer=regularizers.l2(weight), 
        activation='relu', return_sequences=False))
    # model.add(layers.Dropout(rate))
    model.add(layers.Dense(output_node, activation='relu',
        # kernel_regularizer=regularizers.l2(weight)
        ))
    return model