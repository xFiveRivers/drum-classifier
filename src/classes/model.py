import torch
import torch.nn as nn


class CNN_Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
    

    def forward(self, X):
        pass