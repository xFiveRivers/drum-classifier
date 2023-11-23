import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from python_speech_features import mfcc, logfbank

def calc_fft(signal, rate):
    n = len(signal)
    freq = np.fft.rfftfreq(n, d=1/rate)
    mag = abs(np.fft.rfft(signal) / n)
    return (mag, freq)

def envelope(signal, rate, thresh):
    mask = []
    y = pd.series(signal).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > thresh:
            mask.append(True)
        else:
            mask.append(False)
    return mask

def plot_signals(signals):
    rows, cols = 3, 5
    fig, axes = plt.subplots(
        nrows = rows,
        ncols = cols,
        sharex = False,
        sharey = True,
        figsize = (20,5)
    )
    fig.suptitle('Time Series Examples', size = 16)
    
    i = 0
    for x in range(rows):
        for y in range(cols):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fft(fft):
    rows, cols = 3, 5
    fig, axes = plt.subplots(
        nrows = rows,
        ncols = cols,
        sharex = False,
        sharey = True,
        figsize = (20,5)
    )
    fig.suptitle('Fourier Transform Examples', size = 16)

    i = 0
    for x in range(rows):
        for y in range(cols):
            data = list(fft.values())[i]
            mag, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, mag)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fbank(fbank):
    rows, cols = 3, 5
    fig, axes = plt.subplots(
        nrows = rows,
        ncols = cols,
        sharex = False,
        sharey = True,
        figsize = (20,5),
        gridspec_kw={'height_ratios': [1, 1, 1]}
    )
    fig.suptitle('Filter Bank Coefficient Examples', size = 16)

    i = 0
    for x in range(rows):
        for y in range(cols):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(
                list(fbank.values())[i],
                cmap = 'hot',
                interpolation = 'nearest'
            )
            axes[x,y].get_xaxis().set_visible(True)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_mfccs(mfccs):
    rows, cols = 3, 5
    fig, axes = plt.subplots(
        nrows = rows,
        ncols = cols,
        sharex = False,
        sharey = True,
        figsize = (20,5)
    )
    fig.suptitle('Mel Frequency Capstrum Coefficient Examples', size = 16)

    i = 0
    for x in range(rows):
        for y in range(cols):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(
                list(mfccs.values())[i],
                cmap = 'hot',
                interpolation = 'nearest'
            )
            axes[x,y].get_xaxis().set_visible(True)
            axes[x,y].get_yaxis().set_visible(False)
            print(list(mfccs.keys())[i])
            i += 1

signals = {}
fft = {}
fbank = {}
mfccs = {}
df = pd.read_csv('data/samples.csv', usecols=['file', 'class'])

for label in df['class'].unique():
    class_df = df[df['class'] == label]
    file_names = class_df['file'].head(5).tolist()
    for f in file_names:
        signal, rate = librosa.load('data/raw/'+f, sr=None)

        signals[f] = signal
        fft[f] = calc_fft(signal, rate)
        fbank[f] = logfbank(signal, rate, nfilt=26, nfft=1103).T
        mfccs[f] = mfcc(signal, rate, numcep=13, nfilt=26, nfft=1103).T

plot_signals(signals)
# plt.show()

plot_fft(fft)
# plt.show()

plot_fbank(fbank)
# plt.show()

plot_mfccs(mfccs)
plt.show()