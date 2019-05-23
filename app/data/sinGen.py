# -*-coding:utf-8 -*-
import torch
from torch.autograd import Variable
import numpy as np
def SeriesGeN(N):
    x = torch.arange(1,N,0.01)
    return torch.sin(x)


def trainDataGen(seq,k):
    In_data = []
    Out_data = []
    #序列的长度
    L = len(seq)
    for i in range(L-k-1):
        indata = seq[i:i+k]
        outdata = seq[i+k+1:i+k+2]
        In_data.append(indata.numpy())
        Out_data.append(outdata.numpy())
    # print(len(In_data),len(Out_data))
    return np.array(In_data),np.array(Out_data)

def ToVariable(x):
    temp = torch.from_numpy(x)
    return Variable(temp)

def dataGen(N,k):
    x = SeriesGeN(N)
    In_data,Out_data = trainDataGen(x,k)
    return In_data,Out_data

#test
print(SeriesGeN(100).shape)