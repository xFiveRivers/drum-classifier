import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


class EDA():

    def __init__(self):
        pass


    def plot_class_counts(self, df: pd.DataFrame, data_type: str):
        """Plots the count of samples in each class.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing sample metadata.
        data_type : str
            Type of data being used, either 'raw' or 'clean'.
        """

        class_counts = pd.DataFrame(df['class'].value_counts()).reset_index()

        sns.barplot(
            data = class_counts,
            x = 'count',
            y = 'class',
            orient = 'h',
            hue = 'class'
        ).set_title(
            f'Class Distribution for {data_type.capitalize()} Data'
        )


    def plot_sample_len_dist(self, df: pd.DataFrame, data_type: str):
        """Plots the distribution of sample lengths for each class.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing sample metadata.
        data_type : str
            Type of data being used, either 'raw' or 'clean'.
        """

        sns.violinplot(
        data = df,
        y = 'class',
        x = 'length',
        inner = 'quart'
        ).set_title(
            f'Distribution of Sample Lengths by Class '
            f'for {data_type.capitalize()} Data'
        )

    
    def plot_waveform(self, signal: torch.Tensor, rate: int = 44100, 
                      title: str = 'Waveform', ax: np.ndarray|None = None):
        """Plots the waveform of a signal.

        Parameters
        ----------
        signal : torch.Tensor
            Tensor containing signal values.
        rate : int, optional
            Sample rate of signal.
        title : str, optional
            Title of the plot, by default 'Waveform'.
        ax : np.ndarray | None, optional
            Axes to plot on, by default None
        """

        signal = signal.numpy()
        _, n_frames = signal.shape
        time_axis = torch.arange(0, n_frames) / rate

        if ax == None:
            ax = plt.gca()
        ax.plot(time_axis, signal[0], linewidth=1)
        ax.grid(True)
        ax.set_xlim([0, time_axis[-1]])
        ax.set_title(title)
    

    def plot_spectrogram(self, spec: torch.Tensor, title: str = 'Spectrogram',
                          ax: np.ndarray|None = None):
        """Plots the spectrogram of a signal.

        Parameters
        ----------
        spec : torch.Tensor
            Tensor containing the spectogram of a signal
        title : str, optional
            Title of the plot, by default 'Spectrogram'
        ax : np.ndarray | None, optional
            Axes to plot on, by default None
        """

        if ax == None:
            ax = plt.gca()
        ax.imshow(
            librosa.power_to_db(spec),
            origin='lower', 
            aspect='auto'
            interpolation='nearest'
        )
        ax.set_title(title)