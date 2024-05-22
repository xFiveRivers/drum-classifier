import torch
import torch.nn as nn
from torchsummary import summary


class Model_00(nn.Module):

    def __init__(self, n_input: int = 1, n_classes: int = 3):
        super().__init__()

        # === HYPERPARAMETERS === #
        N_INPUT = n_input
        CONV_1_N_OUT = 16
        CONV_2_N_OUT = 32
        CONV_3_N_OUT = 64
        N_OUT = 128

        CONV_KERNAL_SIZE = 3
        STRIDE = 1
        PADDING = 2
        POOL_KERNAL_SIZE = 2

        LINEAR_1_IN = 118272
        LINEAR_1_OUT = 128
        LINEAR_2_OUT = 32

        # === NETWORK LAYERS === #
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = N_INPUT,
                out_channels = CONV_1_N_OUT,
                kernel_size = CONV_KERNAL_SIZE,
                stride = STRIDE,
                padding = PADDING
            ),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels = CONV_1_N_OUT,
                out_channels = CONV_2_N_OUT,
                kernel_size = CONV_KERNAL_SIZE,
                stride =  STRIDE,
                padding = PADDING
            ),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels = CONV_2_N_OUT,
                out_channels = CONV_3_N_OUT,
                kernel_size = CONV_KERNAL_SIZE,
                stride =  STRIDE,
                padding = PADDING
            ),
            nn.ReLU()
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

        self.dropout = nn.Dropout(p=0.5)

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(LINEAR_1_IN, LINEAR_1_OUT),
            nn.Linear(LINEAR_1_OUT, LINEAR_2_OUT),
            nn.Linear(LINEAR_2_OUT, n_classes)
        )


    def forward(self, X):
        output = self.dropout(self.conv4(self.conv3(self.conv2(self.conv1(X)))))
        logits = self.dense(output)

        return logits
    

    def get_name(self):
        return('model_00')


if __name__ == '__main__':
    model = Model_00()
    summary(model.cuda(), (1, 256, 6))