import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset


class DrumTrackerDataset(Dataset):
    def __init__(self, 
                 data_dir: str = 'data/clean/', 
                 csv_filename: str = '_samples.csv'):
        self.samples_df = pd.read_csv(data_dir+csv_filename, index_col=False)
        self.data_dir = data_dir
        
        SAMPLE_FREQ = 16000
        # N_MFCC = 256
        N_FFT = 256
        HOP_LEN = N_FFT // 8
        N_MELS = 256

        # self.mfcc_transform = T.MFCC(
        #     sample_rate = SAMPLE_FREQ,
        #     n_mfcc = N_MFCC,
        #     melkwargs = {
        #         'n_fft': N_FFT,
        #         'hop_length': HOP_LEN,
        #         'n_mels': N_MELS
        #     }
        # )

        self.mel_spec_trans = T.MelSpectrogram(
            sample_rate = SAMPLE_FREQ,
            n_fft = N_FFT,
            hop_length = HOP_LEN,
            n_mels = N_MELS
        )


    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in dataset.
        """

        return len(self.samples_df)
    

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        """Gets a sample from the dataset based on its index.

        Parameters
        ----------
        idx : int
            Index of desired item.

        Returns
        -------
        tuple[torch.Tensor, str]
            MFCCs and label of sample.
        """

        # Get filename of indexed sample
        file = self.samples_df.iloc[
            idx, self.samples_df.columns.get_loc('file')
        ]
        file = str(file)

        # Get sample class
        label = self.samples_df.iloc[
            idx, self.samples_df.columns.get_loc('class')
        ]

        # Load signal
        signal, _ = torchaudio.load(self.file_dir+file)

        # Get MFCCs
        # mfcc = self.mfcc_transform(signal)
        mel_spec = self.mel_spec_trans(signal)

        return mel_spec, label