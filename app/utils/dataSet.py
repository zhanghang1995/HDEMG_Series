# -*- coding:utf-8 -*-

from app.utils.dataloader import
import torch.utils.data as Data
#封装成为Dataset
train_dataSet = Data.TensorDataset(train_A_mean_data,train_A_label)
test_dataSet = Data.TensorDataset(test_A_mean_data,test_A_label)
# test_dataSet = Data.TensorDataset()

torch.manual_seed(1)    # reproducible
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