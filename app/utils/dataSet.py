# -*- coding:utf-8 -*-

from app.utils.dataloader import get_data
import torch.utils.data as Data


batch_size = 40

def get_loader(train_filename,test_filename):
    train_data, test_data, train_label, test_label = get_data(train_filename,test_filename)
    #封装成为Dataset
    train_dataSet = Data.TensorDataset(train_data.reshape(-1,1,8,12),train_label)
    test_dataSet = Data.TensorDataset(test_data.reshape(-1,1,8,12),test_label)

    #封装成为Loader
    # Data Loader for easy mini-batch return in training, the time_series batch shape will be (16, 30, 30)
    train_loader = Data.DataLoader(
        dataset=train_dataSet,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = Data.DataLoader(
        dataset=test_dataSet,
        batch_size=batch_size,
        shuffle=True
    )
    return train_loader,test_loader