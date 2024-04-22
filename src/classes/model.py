import torch
import torch.nn as nn
from torchsummary import summary


class CNN_Model(nn.Module):

    def __init__(self, n_input: int = 1, n_classes: int = 3):
        super().__init__()

        # === HYPERPARAMETERS === #
        N_INPUT = n_input
        CONV_1_N_OUT = 16
        CONV_2_N_OUT = 32
        CONV_3_N_OUT = 64
        N_OUT = 64

        CONV_KERNAL_SIZE = 3
        STRIDE = 1
        PADDING = 2
        POOL_KERNAL_SIZE = 2

        LINEAR_1_IN = 128 * 17 * 2
        self.LINEAR_1_OUT = 128
        self.linear1 = nn.Linear(LINEAR_1_IN, self.LINEAR_1_OUT)
        LINEAR_2_OUT = 32
        LINEAR_3_OUT = n_classes

        # === NETWORK LAYERS === #
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = N_INPUT,
                out_channels = CONV_1_N_OUT,
                kernel_size = CONV_KERNAL_SIZE,
                stride = STRIDE,
                padding = PADDING
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = POOL_KERNAL_SIZE)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels = CONV_1_N_OUT,
                out_channels = CONV_2_N_OUT,
                kernel_size = CONV_KERNAL_SIZE,
                stride =  STRIDE,
                padding = PADDING
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = POOL_KERNAL_SIZE)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels = CONV_2_N_OUT,
                out_channels = CONV_3_N_OUT,
                kernel_size = CONV_KERNAL_SIZE,
                stride =  STRIDE,
                padding = PADDING
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = POOL_KERNAL_SIZE)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels = CONV_3_N_OUT,
                out_channels = N_OUT,
                kernel_size = CONV_KERNAL_SIZE,
                stride =  STRIDE,
                padding = PADDING
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = POOL_KERNAL_SIZE)
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            self.linear1,
            nn.Linear(self.LINEAR_1_OUT, LINEAR_2_OUT),
            nn.Linear(LINEAR_2_OUT, n_classes)
        )

        self.softmax = nn.Softmax(dim=1)


    def forward(self, X):
        output = self.conv1(X)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)

        linear_input = output.size()[1] * output.size()[2] * output.size()[3]
        self.linear1 = nn.Linear(linear_input, self.LINEAR_1_OUT)

        logits = self.dense(output)
        pred = self.softmax(logits)

if __name__ == '__main__':
    cnn = CNN_Model()
    summary(cnn.cuda(), (1, 256, 6))