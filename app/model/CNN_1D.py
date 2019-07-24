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


#build the CNN model #(64,1,8,12) we set the batch_size = 64

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 8, 12) or (1,4,24)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=32,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),
            # output shape (32, 10, 14) is (8*12) Image or the shape (16,4,12) is (4*12)
            nn.BatchNorm2d(32),
            nn.ReLU(),                      # activation
            # nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (32, 4, 6) or
        )
        #local-conv
        self.conv2 = nn.Sequential(         # input shape (32, 10, 14)
            nn.Conv2d(32, 32, 1, 1, 0), # output shape (32, 10, 14)
            nn.BatchNorm2d(32),
            nn.ReLU(),                       # activation
            # nn.Dropout(0.5)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32,64,3,1,0),     #input shape (32,10,14)
            nn.BatchNorm2d(64),             #output shape(64,8,12)
            nn.ReLU(),
            # nn.Dropout(0.5)

        )
        #local-conv
        self.conv4 = nn.Sequential(         # input shape (64, 8, 12)
            nn.Conv2d(64, 64, 1, 1, 0), # output shape (64, 8, 12)
            nn.BatchNorm2d(64),
            nn.ReLU(),                       # activation
            # nn.Dropout(0.5)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(64,64,(3,3),1,2),     #input shape (64,8,12)
            nn.BatchNorm2d(64),             #output shape(64,10,14)
            nn.ReLU(),
            # nn.Dropout(0.5)
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (64, 5, 7) or
        )
        self.linear1 = nn.Sequential(
            nn.Linear(2240,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.linear3 = nn.Sequential(
            nn.Linear(256,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.linear4 = nn.Sequential(
            nn.Linear(64,16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(16, 9),   # fully connected layer, output 9 classes
            nn.LogSoftmax(dim=-1)
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x1 = x
        x = x.view(x.size(0), -1)           # flatten the output of conv3 to (batch_size, 64 * 4 * 4)
        # print("x:{}".format(x.shape))       #(64,1024)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        output = self.out(x)
        return output, x1    # return x1 for visualization the image

#初始化权重

 # 1. 根据网络层的不同定义不同的初始化方式
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    #imitate the data of inputs
    inputs = t.randn(64,1,8,12)
    model = CNN()
    model.apply(weight_init)
    # parameters = list(model.parameters())
    # print(parameters)
    output,_ = model(inputs)
    print("CNN mdoel:{}".format(model))
    print("x:{}".format(output))
