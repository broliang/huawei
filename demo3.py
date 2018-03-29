# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
from keras.layers import RNN
from statsmodels.tsa.arima_model import ARMA


def get_data(data, time_steps=50):
    dataX = []
    dataY = []
    for i in range(0, len(data) - time_steps, 1):
        seq_in = data[i:i + time_steps]
        seq_out = data[i + time_steps]
        dataX.append(seq_in)
        dataY.append(seq_out)
    return dataX, dataY

def create_dataset(dataset, look_back=50):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)
my = []  #预测各虚拟机总次数
ttt = [] #训练数据各虚拟机总次数
name = 'class_flavor'
# 读数据
path = 'TrainData_2015.1.1_2015.2.19.txt'
train = pd.read_csv(path, sep='\t', names=['ID', 'class', 'time'])
train['time'] = pd.to_datetime(train['time'])
train = train.drop('ID', axis=1)
train.set_index('time', inplace=True)
train['day'] = train.index.weekday

y = np_utils.to_categorical(train['class'])