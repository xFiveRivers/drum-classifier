import csv
import os
import re
import librosa

# Get list of wav files
data_folder = 'data/raw'
wav_files = [file for file in os.listdir('data/raw') if file.endswith('.wav')]
wav_files.sort()
# wav_files = os.listdir(data_folder)

# Create a list of classes for the wav files
concat_result = ' '.join(wav_files)
class_list = re.findall(r'(\S+)_\d+\.wav', concat_result)
unique_classes = list(set(class_list))

# Create dictionary to encode class ids
class_ids = {}
for id, label in enumerate(unique_classes):
    class_ids[label] = id
print(class_ids)

# Create a list of class ids for the wav files
class_id_list = [class_ids[label] for label in class_list]
    
# Create empty list to store signal lengths
signal_lengths = []

# Get signal length of each file
for f in wav_files:
    signal, rate = librosa.load('data/raw/'+f, sr=None)
    signal_lengths.append(round(signal.shape[0] / rate, 4))

# Create CSV file
with open('data/samples.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['file', 'length', 'class', 'class_id'])
    writer.writerows(zip(wav_files, signal_lengths, class_list, class_id_list))