import csv
import os
import re
import librosa

data_folder = 'data/wavefiles'
wav_files = os.listdir(data_folder)

concat_result = ' '.join(wav_files)
class_list = re.findall(r'(\S+)_\d+\.wav', concat_result)
unique_classes = list(set(class_list))

signal_lengths = []

for f in wav_files:
    signal, rate = librosa.load('data/wavefiles/'+f, sr=None)
    signal_lengths.append(signal.shape[0] / rate)

with open('data/samples.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['file', 'length', 'rate', 'class'])
    writer.writerows(zip(wav_files, signal_lengths, class_list))