import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

#reproduciable
t.manual_seed(1)

# Hyper Parameters
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001 #learning rate
DOWNLOAD_MNIST  = False

"""


#load the data mnist
train_data = torchvision.datasets.MNIST(
    root='./',
    train=True, #set this data to train_data
    transform=torchvision.transforms.ToTensor(),# 转换 PIL.Image or numpy.ndarray 成
                                                # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_MNIST
)

test_data = torchvision.datasets.MNIST(
    root='./',
    train=False
)

#(50,1,28,28)
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# test for per 2000
test_x = t.unsqueeze(test_data.test_data, dim=1).type(t.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000]

"""

#build the CNN model #(64,1,8,12) we set the batch_size = 64

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 8, 12)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),
            # output shape (16, 8, 12) is (8*12) Image
            nn.BatchNorm2d(16),
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 4, 6)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 4, 6)
            nn.Conv2d(16, 32, (5,7), 1, 2), # output shape (16, 4, 4)
            nn.BatchNorm2d(32),
            nn.ReLU(),                       # activation
            nn.Dropout(0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32,64,(5,5),1,2),     #input shape (32,4,4)
            nn.BatchNorm2d(64),             #output shape(64,4,4) = 1024
            nn.ReLU(),
            nn.Dropout(0.5)

        )
        self.linear1 = nn.Sequential(
            nn.Linear(1024,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(256,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.linear3 = nn.Sequential(
            nn.Linear(64,16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )

        self.out = nn.Sequential(
            nn.Linear(16, 9),   # fully connected layer, output 9 classes
            nn.LogSoftmax()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x1 = x
        x = x.view(x.size(0), -1)           # flatten the output of conv3 to (batch_size, 64 * 4 * 4)
        print("x:{}".format(x.shape))       #(64,1024)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        output = self.out(x)
        return output, x1    # return x1 for visualization the image


if __name__ == '__main__':
    #imitate the data of inputs
    inputs = t.randn(64,1,8,12)
    model = CNN()
    output,_ = model(inputs)
    print("CNN mdoel:{}".format(model))
    print("output:{}".format(output))
