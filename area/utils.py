
import config
import colorsys
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
import random
import datetime 
import os

def save_data(file_location, name, value):
    with h5py.File(config.file_location + r'\{}.h5'.format(name),'w') as hf:
        hf.create_dataset('elem', data=value, compression='gzip', compression_opts=9)
        hf.close()

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def checkpoints():
	# current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
	current_time = config.file_name
	model_dir = os.path.join('model_lstm')
	if not os.path.exists(model_dir):
		os.mkdir(model_dir)
	model_dir = os.path.join(model_dir, current_time)
	if not os.path.exists(model_dir):
		os.mkdir(model_dir)
	filepath = os.path.join(model_dir, 
		'{epoch:05d}-{loss:.6f}-{mae:.6f}-{rmse:.6f}-' +
		'{val_loss:.6f}-{val_mae:.6f}-{val_rmse:.6f}.h5')
	checkpoint = [tf.keras.callbacks.ModelCheckpoint(filepath, verbose=2,  
			monitor='val_loss', save_best_only=True, save_weights_only=False),
		tf.keras.callbacks.EarlyStopping(monitor='loss', 
			patience=10000, verbose=True),
		tf.keras.callbacks.TensorBoard(log_dir=model_dir, histogram_freq=1)]				 
	return checkpoint

def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step
    return hls_colors

def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])
    return rgb_colors

def color(value):
    digit = list(map(str, range(10))) + list('ABCDEF')
    if isinstance(value, tuple):
        string = '#'
        for i in value:
            a1 = i // 16
            a2 = i % 16
            string += digit[a1] + digit[a2]
        return string
    elif isinstance(value, str):
        a1 = digit.index(value[1]) * 16 + digit.index(value[2])
        a2 = digit.index(value[3]) * 16 + digit.index(value[4])
        a3 = digit.index(value[5]) * 16 + digit.index(value[6])
        return (a1, a2, a3)

def get_peaks_troughs(h,rangesize):
    peaks = list()
    troughs = list()
    S = 0
    for x in range(1,len(h)-1):
        if S == 0:
            if h[x] > h[x+1]:
                S = 1 ## down
            else:
                S = 2 ## up
        elif S == 1:
            if h[x] < h[x+1]:
                S = 2
                ## from down to up
                if len(troughs):
                    ## check if need merge
                    (prev_x,prev_trough) = troughs[-1]
                    if x - prev_x < rangesize:
                        if prev_trough > h[x]:
                            troughs[-1] = (x,h[x])
                    else:
                        troughs.append((x,h[x]))
                else:
                    troughs.append((x,h[x]))                
        elif S == 2:
            if h[x] > h[x+1]:
                S = 1
                ## from up to down
                if len(peaks):
                    prev_x,prev_peak = peaks[-1]
                    if x - prev_x < rangesize:
                        if prev_peak < h[x]:
                            peaks[-1] = (x,h[x])
                    else:
                        peaks.append((x,h[x]))
                else:
                    peaks.append((x,h[x]))
    return peaks,troughs