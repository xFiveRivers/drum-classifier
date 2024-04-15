import csv
import librosa
import os
import pandas as pd
import re
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T


class Preprocessor():
    def __init__(self, raw_dir: str = 'data/raw/', clean_dir: str = 'data/clean/',
                 chunk_len: int = 160, resample_freq: int = 16000):

        self.raw_dir = raw_dir
        self.clean_dir = clean_dir
        self.chunk_len = chunk_len
        self.input_freq = 41000
        self.resample_freq = resample_freq
        self.db_cutoff = -80
        self.unique_classes = None
        self.csv_name = '_samples.csv'

        self.resample = T.Resample(
            orig_freq = self.input_freq, 
            new_freq = self.resample_freq
        )

        self.power_to_db = T.AmplitudeToDB(stype="amplitude", top_db=80)


    def forward(self):
        # Generate csv file for raw directory
        self._create_csv(self.raw_dir)

        # Load csv file
        df = pd.read_csv(self.raw_dir + self.csv_name, index_col=None)

        # Loop through each unique class
        for unique_class in self.unique_classes:
            # Print progress statement
            print(f'Processing {unique_class} files...')

            # Initialize df for class
            class_df = df.loc[df['class'] == unique_class]

            # Loop through each file in the current class
            self._process_class(class_df, unique_class)        

        # Generate csv file for clean directory
        self._create_csv(self.clean_dir)

        # Print complete statement
        print('Preprocessing complete!')


    def _process_class(self, class_df: pd.DataFrame, unique_class: str,
                       counter: int = 1):
        # Loop through each file in the current class
        for file in list(class_df['file']):
            sample, self.input_freq = torchaudio.load(self.raw_dir + file)
            sample = self._stereo_to_mono(sample)                   # Mono
            sample = self.resample(sample)                          # Downsample
            chunks = torch.split(sample.squeeze(0), self.chunk_len) # Get chunks

            # Loop through each chunk (skip last chunk)
            for chunk in chunks[:-1]:
                avg_db = self._get_avg_db(chunk)

                if avg_db > self.db_cutoff:
                    # Save chunk as WAV file
                    torchaudio.save(
                        src = chunk.unsqueeze(0),
                        uri = self.clean_dir + unique_class + '_' + \
                            str(counter).zfill(3) + '.wav',
                        sample_rate = self.resample_freq
                    )

                # Increment counter for naming files
                counter += 1


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
        
    
    def _get_avg_db(self, signal: torch.Tensor) -> float:
        """Returns the average decibel level of a signal.

        Parameters
        ----------
        signal : torch.Tensor
            Signal in power amplitude domain.

        Returns
        -------
        float
            Average decibel level of signal.
        """

        if len(signal.shape) == 1:
            signal = signal.unsqueeze(0)

        signal_db = self.power_to_db(signal)

        return(round(float(torch.mean(signal_db.squeeze(0))), 4))


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
        lengths = []
        avg_dbs = []

        # Get signal length and average db level of each file
        for f in wav_files:
            signal, rate = torchaudio.load(dir + f)
            lengths.append(round(signal.shape[1] / rate, 4))
            avg_dbs.append(self._get_avg_db(signal))

        # Create CSV file
        with open(dir + self.csv_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['file', 'length', 'avg_db', 'class', 'class_id'])
            writer.writerows(zip(
                wav_files, lengths, avg_dbs, class_list, class_id_list
            ))