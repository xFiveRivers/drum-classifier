import csv
import librosa
import os
import re
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T



class DataGenerator(nn.Module):
    def __init__(self, raw_dir: str, clean_dir: str, 
                 input_freq: int = 41000, resample_freq: int = 16000):
        super().__init__()

        self.raw_dir = raw_dir
        self.clean_dir = clean_dir

        self.resample = T.Resample(orig_freq=input_freq, new_freq=resample_freq)

    def forward(self):
        self._create_csv(self.raw_dir)

    def _create_csv(self, dir: str):
        print(f'Generating CSV for [{dir}]...')
        # Get list of wav files
        wav_files = [file for file in os.listdir(dir) if file.endswith('.wav')]
        wav_files.sort()

        # Create a list of classes for the wav files
        concat_result = ' '.join(wav_files)
        class_list = re.findall(r'(\S+)_\d+\.wav', concat_result)
        unique_classes = list(set(class_list))

        # Create dictionary to encode class ids
        class_ids = {}
        for id, label in enumerate(unique_classes):
            class_ids[label] = id
        class_id_list = [class_ids[label] for label in class_list]

        # Create empty list to store signal lengths
        signal_lengths = []

        # Get signal length of each file
        for f in wav_files:
            signal, rate = librosa.load(dir+f, sr=None)
            signal_lengths.append(round(librosa.get_duration(signal, rate), 4))

        # Create CSV file
        with open('data/samples.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['file', 'length', 'class', 'class_id'])
            writer.writerows(zip(
                wav_files, signal_lengths, class_list, class_id_list
            ))

        print(f'CSV for [{dir}] generated.')