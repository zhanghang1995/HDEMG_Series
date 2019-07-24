# -*- coding:utf-8 -*-
import cv2 as cv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler,minmax_scale,Normalizer
"""
fucntion: transform the sEMG signal to Image
"""

# transform the data into(0-255)
def linearConvert(data):
    data_max = np.max(data)
    data_min = np.min(data)
    linear_data = np.round(255*(data-data_min)/(data_max-data_min) + 0)
    return linear_data

# transform the data into (0-1)
def minMaxScalar(data):
    if data.size != 0:
        scaler = MinMaxScaler()
        scaler_data = scaler.fit_transform(data)
        return scaler_data
# tranform the data into(0-1) using the z-score

def z_score(data):
    return (data - np.mean(data)) / np.std(data, ddof=1)

# we expand the data 0-1 to 0-255
def signal2Image(signal):
    pass




