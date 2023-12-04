import librosa
import os
import pandas as pd
from scipy.io import wavfile
from tqdm import tqdm

df = pd.read_csv('data/samples.csv', usecols=['file', 'class'])

if len(os.listdir('data/clean')) == 0:
    for f in tqdm(df.file):
        signal, rate = librosa.load('data/raw/'+f, sr=16000)
        wavfile.write(filename='data/clean/'+f, rate=rate, data=signal)