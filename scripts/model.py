import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc

df = pd.read_csv('data/samples.csv')
df.set_index('file', inplace=True)

classes = df['class'].unique()
class_dist = df.groupby(['class'])['length'].mean()