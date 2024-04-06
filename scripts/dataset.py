import librosa
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset



class DrumTrackerDataset(Dataset):
    def __init__(self, csv_file: str, file_dir: str, transform):
        self.sample_df = pd.read_csv(csv_file, index_col=False)
        self.file_dir = file_dir
        self.transform = transform

    def __len__(self):
        return len(self.sample_df)
    
    def __getitem__(self, idx):
        # Get filename of indexed sample
        file = self.sample_df.iloc[idx, self.sample_df.columns.get_loc('file')]
        file = str(file)

        # Get sample class
        label = self.sample_df.iloc[idx, self.sample_df.columns.get_loc('class')]

        # Load signal
        signal, _ = torchaudio.load(self.file_dir+file)

        # Transform signal
        mfcc = self.transform(signal)

        return mfcc, label



class PreProcPipeline(nn.Module):
    def __init__(self, input_freq=41000, resample_freq=32000, n_mfcc=512,
                 n_fft=1024, hop_len=512, n_mels=512):
        super().__init__()

        # Resample Transformation
        self.resample = T.Resample(
            orig_freq=input_freq, 
            new_freq=resample_freq
        )

        # MFCC Transformation
        self.mfcc = T.MFCC(
            sample_rate = resample_freq, 
            n_mfcc = n_mfcc,
            melkwargs = {
                'n_fft': n_fft,
                'hop_length': hop_len,
                'n_mels': n_mels
            }
        )

    def _right_pad():
        pass

    def _right_trim():
        pass
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # If not mono, mixdown to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        print(waveform.dtype)
        
        # Downsample signal
        resampled = self.resample(waveform)

        # Extract MFCCs
        mfcc = self.mfcc(resampled)

        return mfcc