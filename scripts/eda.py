import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from python_speech_features import mfcc, logfbank

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
            axes[x,y].get_xaxis().set_visable(False)
            axes[x,y].get_yaxis().set_visable(False)
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
            axes[x,y].get_xaxis().set_visable(False)
            axes[x,y].get_yaxis().set_visable(False)
            i += 1

def plot_fbank(fbank):
    rows, cols = 3, 5
    fig, axes = plt.subplots(
        nrows = rows,
        ncols = cols,
        sharex = False,
        sharey = True,
        figsize = (20,5)
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
            axes[x,y].get_xaxis().set_visable(False)
            axes[x,y].get_yaxis().set_visable(False)

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
            axes[x,y].set_title(list(mfcc.keys())[i])
            axes[x,y].imshow(
                list(mfcc.values())[i],
                cmap = 'hot',
                interpolation = 'nearest'
            )
            axes[x,y].get_xaxis().set_visable(False)
            axes[x,y].get_yaxis().set_visable(False)

signals = {}
df = pd.read_csv('data/samples.csv', usecols=['file', 'class'])

for label in df['class'].unique():
    class_df = df[df['class'] == label]
    file_names = class_df['file'].head(5).tolist()
    for f in file_names:
        signal, _ = librosa.load('data/raw/'+f, sr=None)
        signals[f] = signal

