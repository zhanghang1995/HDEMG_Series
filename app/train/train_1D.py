# -*-coding:utf-8 -*-
from app.model.CNN_1D import CNN_1D_Series
import torch.nn as nn
import torch.nn.functional as F
import torch as t
from app.data.sinGen import dataGen,ToVariable
import torch.optim as op
#train
EPOCH = 10000
# In_data,Out_data = dataGen(100,10)

# define the model and loss.,optimizer
model = CNN_1D_Series()
loss_func = nn.MSELoss()
optimizer = op.SGD(model.parameters(), lr=0.001)
def train(batch_num,batch_size):

    #begint to train
    for epoch in range(EPOCH):
        print("EPOCH:[{}/{}]".format(epoch,EPOCH))
        for batch_idx in range(batch_num):
            #load the data for each batch
            seq = In_data[batch_idx*batch_size:(batch_idx+1)*batch_size]
            out = Out_data[batch_idx*batch_size:(batch_idx+1)*batch_size]
            seq = ToVariable(seq)
            seq = seq.unsqueeze(1)
            out = ToVariable(out)
            optimizer.zero_grad()
            output = model(seq)
            loss = loss_func(output,out)
            print("Batch[{}],Loss:{}".format(batch_idx,loss.data[0]))
            loss.backward()
            optimizer.step()

def test(K):

    #begin to test
    seq_test = In_data[K:]
    out_test = Out_data[K:]
    seq_test = ToVariable(seq_test)
    seq_test = seq_test.unsqueeze(1)
    out_test = ToVariable(out_test)
    output = model(seq_test)
    loss_test = F.mse_loss(output,out_test)
    print("Test,Loss:{}".format(loss_test.data[0]))


if __name__ == "__main__":
    In_data, Out_data = dataGen(100, 10)
    # In_data = t.from_numpy(In_data)
    # Out_data = t.from_numpy(Out_data)
    train(50,10)
    test(700)
    t.save(model,'model.pkl')
