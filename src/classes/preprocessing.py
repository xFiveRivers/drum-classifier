import csv
import librosa
import os
import pandas as pd
import re
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T


class Preprocessing(nn.Module):
    def __init__(self, raw_dir: str, clean_dir: str, chunk_len: int = 100,
                 input_freq: int = 41000, resample_freq: int = 16000):
        super().__init__()

        self.raw_dir = raw_dir
        self.clean_dir = clean_dir
        self.chunk_len = chunk_len
        self.unique_classes = None

        self.resample = T.Resample(orig_freq=input_freq, new_freq=resample_freq)


    def forward(self):
        # Generate csv file for raw directory
        self._create_csv(self.raw_dir)

        # Load csv file
        df = pd.read_csv(self.raw_dir + 'samples.csv', index_col=None)

        # Loop through each unique class
        for unique_class in self.unique_classes:
            # Print progress statement
            print(f'Processing {unique_class} files...')

            # Initialize counter for file names
            class_counter = 1

            # Initialize df for class
            class_df = df.loc[df['class'] == unique_class]

            # Loop through each file in the current class
            for file in list(class_df['file']):
                sample = torchaudio.load(self.raw_dir + file)   # Load sample
                sample = self.resample(sample)                  # Downsample
                sample = self._stereo_to_mono(sample)           # Stereo to mono
                chunks = torch.split(sample, self.chunk_len)    # Get chunks

                # Loop through each chunk (skip last chunk)
                for chunk in chunks[:-1]:
                    # Save chunk as WAV file
                    torchaudio.save(
                        src = chunk,
                        uri = self.clean_dir + unique_class + \
                            str(class_counter).zfill(3) + '.wav',
                        sample_rate = self.resample_freq
                    )

                    # Increment counter for naming files
                    class_counter += 1

        # Generate csv file for clean directory
        self._create_csv(self.raw_dir)


    def _stereo_to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """Converts a stereo waveform to mono

        Parameters
        ----------
        waveform : torch.Tensor
            Waveform to convert to mono.

        Returns
        -------
        torch.Tensor
            Waveform converted to mono.
        """

        if waveform.shape[0] > 1:
            return(torch.mean(waveform, dim=0, keepdim=True))


    def _create_csv(self, dir: str):
        print(f'Generating CSV for [{dir}]...')
        # Get list of wav files
        wav_files = [file for file in os.listdir(dir) if file.endswith('.wav')]
        wav_files.sort()

        # Create a list of classes for the wav files
        concat_result = ' '.join(wav_files)
        class_list = re.findall(r'(\S+)_\d+\.wav', concat_result)
        self.unique_classes = list(set(class_list))

        # Create dictionary to encode class ids
        class_ids = {}
        for id, label in enumerate(self.unique_classes):
            class_ids[label] = id
        class_id_list = [class_ids[label] for label in class_list]

        # Create empty list to store signal lengths
        signal_lengths = []

        # Get signal length of each file
        for f in wav_files:
            signal, rate = librosa.load(dir+f, sr=None)
            signal_lengths.append(round(librosa.get_duration(signal, rate), 4))

        # Create CSV file
        with open(dir + 'samples.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['file', 'length', 'class', 'class_id'])
            writer.writerows(zip(
                wav_files, signal_lengths, class_list, class_id_list
            ))

        print(f'CSV for [{dir}] generated.')