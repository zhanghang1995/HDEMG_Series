# -*- coding:utf-8 -*-
import torch.nn as nn
import torch as t


#build the RNN model
class RNN(nn.Module):
    def __init__(self,input_size,output_size,hidden_size,layers):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layers = layers

        self.lstm = nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=layers)
        self.linear = nn.Linear(hidden_size,output_size)


    def forward(self, input_x):
        x = self.lstm(input_x)





