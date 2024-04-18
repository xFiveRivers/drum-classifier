import torch
import torch.nn as nn


class CNN_Model(nn.Module):

    def __init__(self, n_input: int = 1, n_output: int = 128,
                 kernal_size: int = 16):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = n_input,
                out_channels = 16,
                kernel_size = kernal_size,
                stride = 1,
                padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernal_size = 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels = 16,
                out_channels = 32,
                kernel_size = kernal_size,
                stride = 1,
                padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernal_size = 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels = 32,
                out_channels = 64,
                kernel_size = kernal_size,
                stride = 1,
                padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernal_size = 2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels = 64
                out_channels = n_output,
                kernel_size = kernal_size,
                stride = 1,
                padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernal_size = 2)
        )

        self.flatten = nn.Flatten()
        self.linear = nn.Linear()


    def forward(self, X):
        pass