import numpy as np
import torch.nn as nn
import torch as t

#build the model for 1D Time_series
class CNN_1D_Series(nn.Module):
    def __init__(self):
        super(CNN_1D_Series, self).__init__()
        self.conv1 = nn.Sequential(
            #the first conv
            nn.Conv1d(
                in_channels=1,
                out_channels=8,
                kernel_size=3
            ),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            #the second conv
            nn.Conv1d(
                in_channels=8,
                out_channels=16,
                kernel_size=3
            ),
            nn.MaxPool1d(kernel_size=2)
        )
        self.fc = nn.Linear(16,1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        print(x.shape)
        x = x.view(x.size(0),-1)
        out = self.fc(x)
        return out

class CNN_2D(nn.Module):
    def __init__(self):
        super(CNN_2D, self).__init__()
        self.conv1 = nn.Sequential(
            #the first conv
            nn.Conv1d(
                in_channels=96,
                out_channels=128,
                kernel_size=3
            ),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            #the second conv
            nn.Conv1d(
                in_channels=8,
                out_channels=16,
                kernel_size=3
            ),
            nn.MaxPool1d(kernel_size=2)
        )
        self.fc = nn.Linear(160,1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        out = self.fc(x)
        return out
