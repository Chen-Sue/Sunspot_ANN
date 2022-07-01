import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import size
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608),
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608)])
from tensorflow.keras import layers, initializers, regularizers

def stateless_lstm(x, output_node, layer=2, layer_size=128, rate=0.20, weight=1e-6):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(x.shape[1], x.shape[2])))
    for i in np.arange(1, layer, 1):
        if i == layer-1:
            return_sequences = False
        else:
            return_sequences = True
        units = int(layer_size/(2**(i-1)))
        model.add(layers.LSTM(units=units,
            activation='relu',
            # kernel_initializer=initializers.he_normal(),
            # kernel_initializer=initializers.glorot_normal(),
            kernel_regularizer=regularizers.l2(weight), 
            return_sequences=return_sequences, 
            name='hidden_layer{}'.format(i)))
        model.add(layers.Dropout(rate)),
        # model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=output_node, 
        activation='relu',
        # kernel_initializer=initializers.he_normal(),
        # kernel_initializer=initializers.glorot_normal(),
        kernel_regularizer=regularizers.l2(weight), 
        name='output_layer'))
    return model

def cnn1d(x, output_node, layer=2, layer_size=128, rate=0.20, weight=1e-6):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(x.shape[1], x.shape[2])))
    for i in np.arange(1, layer, 1):
        units = int(layer_size/(2**(i-1)))
        model.add(layers.Conv1D(filters=units, kernel_size=3, strides=1, 
            activation='relu', data_format="channels_first"))
        model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(output_node, activation='relu', ))
    return model

def cnn_lstm(x, output_node, layer=2, layer_size=128, rate=0.20, weight=1e-6):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(x.shape[1], x.shape[2])))
    model.add(layers.Conv1D(filters=128, kernel_size=3, strides=1, 
        activation='relu', data_format="channels_first"))
    model.add(layers.MaxPooling1D(pool_size=2))
    # model.add(layers.Conv1D(filters=64, kernel_size=3, strides=1, 
    #     activation='relu', data_format="channels_first"))
    # model.add(layers.MaxPooling1D(pool_size=2))
    # model.add(layers.Conv1D(filters=32, kernel_size=3, strides=1, 
    #     activation='relu', data_format="channels_first"))
    # model.add(layers.MaxPooling1D(pool_size=2))
    # model.add(layers.LSTM(128, activation='relu', return_sequences=True))
    model.add(layers.LSTM(128, activation='relu', return_sequences=True))
    model.add(layers.LSTM(64, activation='relu', return_sequences=False))
    model.add(layers.Flatten())
    model.add(layers.Dense(output_node, activation='relu', ))
    return model