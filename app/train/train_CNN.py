# -*- coding:utf-8 -*-

import torch as t
import torch.nn as nn
import torch.optim as optimizer
from torch.autograd import Variable
from app.model.CNN_1D import CNN
from app.utils.dataSet import train_loader,test_loader

"""
    function: define the data train 
"""

# Hyper parameter
EPOCH = 10
#model
cnn = CNN()

loss_func = nn.CrossEntropyLoss()
optimizer = optimizer.Adam(cnn.parameters(),lr=0.001,betas=(0.5,0.999))

def train():
    #begin to train

    correct = 0
    total = 0
    for epoch in range(EPOCH):
        for step,(x,y) in enumerate(train_loader):
            # 封装为自动求导类型
            x = Variable(x)
            y = Variable(y)
            # print(x.shape,y.shape)
            # 前向传播
            output,x1 = cnn(x.float())
            loss = loss_func(output,y.squeeze())
            # 梯度清空与梯度下降
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _,predict  = t.max(output,dim=1)
            print('Predict:{}'.format(predict))
            correct += predict.eq(y.data.squeeze()).cpu().sum()
            total += y.size(0)
            # caculate the accuracy
            print('Loss:{}'.format( loss.item()))
            #Accuracy
            print('Accuracy:{}'.format(100.*predict.eq(y.data.squeeze()).cpu().sum()/y.size(0)))
    accu = 100. * correct/total
    print('Accuracy:{}'.format(accu))
    return loss


def test():
    accuracy = 0

    return accuracy



if __name__ == '__main__':
    loss = train()
    t.save(cnn,'model_Down.pkl')