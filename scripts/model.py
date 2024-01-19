import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from scipy.io import wavfile
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc

class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=551, rate=16000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/100)

def build_rand_feat():
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df['class']==rand_class].index)
        signal, rate = librosa.load('data/clean/'+file)
        rand_index = np.random.randint(0, signal.shape[0]-config.step)
        sample = signal[rand_index:rand_index+config.step]
        X_sample = mfcc(
            sample, 
            config.rate, 
            numcep = config.nfeat, 
            nfilt = config.nfilt,
            nfft = config.nfft
        ).T
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample if config.mode == 'conv' else X_sample.T)
        y.append(classes.index(rand_class))
    X, y = np.array(X), np.array(y)
    X = (X - _min) / (_max - _min)
    if config.mode == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    elif config.mode == 'time':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    y = to_categorical(y, num_classes=3)
    return X, y

# Load Data
df = pd.read_csv('data/samples.csv')
df.set_index('file', inplace=True)

# Find distribution of classes based on sample lengths
classes = list(df['class'].unique())
class_dist = df.groupby(['class'])['length'].mean()

# Find the total number of samples we can use
# Based on using 1/10 of a second
n_samples = 2 * int(df['length'].sum() / 0.01)

# Build probability distribution to sample from
prob_dist = class_dist / class_dist.sum()
choices = np.random.choice(class_dist.index, p=prob_dist)

config = Config(mode='conv')

if config.mode == 'conv':
    X, y = build_rand_feat()
elif config.mode == 'time':
    X, y = build_rand_feat()

print(X.shape)