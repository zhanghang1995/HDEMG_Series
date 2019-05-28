import torch
import os
import numpy as np
import pandas as pd
from app.utils import constant


torch.manual_seed(1)    # reproducible
filename = "Down_"
train_filename = [filename+'trainData.csv',filename+'trainLabel.csv']
test_filename = [filename+'testData.csv',filename+'testLabel.csv']

if os.path.exists(os.path.join(constant.TRAIN_FIELS, filename+'test.pth') and os.path.join(constant.TEST_FILES,filename+'train.pth')):
    train_data , train_label = torch.load(os.path.join(constant.TRAIN_FIELS, filename+'train.pth'))
    test_data ,test_label = torch.load(os.path.join(constant.TEST_FILES, filename+'test.pth'))
    print('<<<the data of train and test is loaded>>>')
else:
    print('Processing the data...')
    #load the train
    train_data = np.array(pd.read_csv(os.path.join(constant.TRAIN_FIELS, train_filename[0]),header=None))
    train_label = np.array(pd.read_csv(os.path.join(constant.TRAIN_FIELS, train_filename[1]),header=None))
    #load the test
    test_data = np.array(pd.read_csv(os.path.join(constant.TEST_FILES, test_filename[0]),header=None))
    test_label = np.array(pd.read_csv(os.path.join(constant.TEST_FILES, test_filename[1]),header=None))

    #A tensor(test and train)
    train_data = torch.tensor(train_data,dtype=torch.long)
    test_data = torch.tensor(test_data, dtype=torch.long)
    #label tensor(test and train)
    train_label = torch.tensor(train_label,dtype=torch.long)
    test_label= torch.tensor(test_label,dtype=torch.long)

    #save the file train
    torch.save((train_data,train_label),os.path.join(constant.TRAIN_FIELS,filename+'train.pth'))
    #save the file test
    torch.save((test_data,test_label),os.path.join(constant.TEST_FILES,filename+'test.pth'))


