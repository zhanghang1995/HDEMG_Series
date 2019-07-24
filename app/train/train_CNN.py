# -*- coding:utf-8 -*-

import torch as t
import torch.nn as nn
import torch.optim as optimizer
from torch.autograd import Variable
import numpy as np

from app.model.CNN_1D import CNN,weight_init
from app.utils.dataloader import filename_get
from app.utils.dataSet import get_loader


"""
    function: define the data train 
"""

#If the GPU is available
use_cuda = t.cuda.is_available()
# Hyper parameter
EPOCH = 1000
#model
cnn = CNN()
cnn.apply(weight_init)

if use_cuda:
    cnn = cnn.cuda()

loss_func = nn.CrossEntropyLoss()
optimizer = optimizer.Adam(cnn.parameters(),lr=0.001,betas=(0.5,0.999))

def train(j):
    #begin to train
    correct = 0
    total = 0
    for epoch in range(EPOCH):
        for step,(x,y) in enumerate(train_loader):
            if use_cuda:
                x,y = x.cuda(),y.cuda()
            # 封装为自动求导类型
            x = Variable(x)
            y = Variable(y)
            # 前向传播
            output,x1 = cnn(x.float())
            loss = loss_func(output,y.squeeze())
            # 梯度清空与梯度下降
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _,predict  = t.max(output,dim=1)
            # print('Predict:{}'.format(predict))
            correct += predict.eq(y.data.squeeze()).cpu().sum()
            total += y.size(0)
            # caculate the accuracy
            print('Loss name->{}:{}'.format(j,loss.item()))
            #Accuracy
            # print('Accuracy:{}'.format(100.*predict.eq(y.data.squeeze()).cpu().sum()/y.size(0)))
    # acc = 100. * correct/total
    # print('Accuracy:{}'.format(acc))
        accuracy = test(model=cnn,name=j)
    return accuracy,loss


def test(model,name):
    correct = 0
    total = 0
    ave_loss = 0
    for step,(x,y) in enumerate(test_loader):
        if use_cuda:
            x,y,model= x.cuda(),y.cuda(),model.cuda()
        x,y = Variable(x),Variable(y)
        output,_ = model(x.float())
        loss = loss_func(output,y.squeeze())
        _,predict = t.max(output,dim=1)
        correct += predict.eq(y.data.squeeze()).cpu().sum()
        total += y.size(0)
        #smooth average
        ave_loss = ave_loss * 0.9 + loss.item() * 0.1

        if(step+1) % 10 == 0 or (step+1) == len(test_loader):
            print("==>>step:{}，test_loss:{:.6f},acc:{:.3f}".format(step+1,ave_loss,correct * 100./total))

    print("Total name->{}accuracy:{}".format(name,correct * 100./total))

    return correct * 100./total

if __name__ == '__main__':
    # trian model
    #list all filename
    import os
    filepath = '../data/trainingfiles'
    b = os.listdir(filepath)
    a = []
    for i in b:
        a.append((i.split("_")[0]))
    a = set(a)
    for j in list(a):
        if j !="trainLabel.csv":
            train_filename, test_filename = filename_get(name=j+'_')
            print(j)
            train_loader, test_loader = get_loader(train_filename,test_filename)
            accuracy,loss = train(j)
            np.save(j+"accuracy.txt",accuracy)
            t.save(cnn,j+'.pkl')
        # test model
    # model = t.load('IN.pkl')
    # test(model)