
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
step = 10