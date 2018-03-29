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
    return np.array(dataX), np.array(dataY)

def create_dataset(dataset, look_back=50):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

name = 'class_flavor'
# 读数据
path = 'TrainData_2015.1.1_2015.2.19.txt'
train = pd.read_csv(path, sep='\t', names=['ID', 'class', 'time'])
train['time'] = pd.to_datetime(train['time'])
train = train.drop('ID', axis=1)
train.set_index('time', inplace=True)
train['day'] = train.index.weekday

# 转换
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train['class'].tolist())
num_class = le.transform(train['class'].tolist())
time_steps = 20
dataX,dataY = get_data(num_class,20)
dataX = dataX.reshape(len(dataX),time_steps,1)
dataY = np_utils.to_categorical(dataY)
model = Sequential()
model.add(LSTM(32, input_shape=(dataX.shape[1], dataX.shape[2])))
model.add(Dense(dataY.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(dataX, dataY, nb_epoch=500, batch_size=10, verbose=2)





flavor__ = {'flavor1':[1, 1024],
'flavor2': [1, 2048],
'flavor3': [1, 4096],
'flavor4': [2, 2048],
'flavor5': [2, 4096],
'flavor6': [2, 8192],
'flavor7': [4, 4096],
'flavor8': [4, 8192],
'flavor9': [4, 16384],
'flavor10': [8, 8192],
'flavor11': [8, 16384],
'flavor12': [8, 32768],
'flavor13': [16, 16384],
'flavor14': [16, 32768],
'flavor16': [32, 32768],
'flavor17': [32, 65536],
'flavor18': [32, 131072],
'flavor19': [64, 65536],
'flavor20': [64, 131072],
'flavor21': [64, 262144],
'flavor22': [128, 131072],
'flavor23': [128, 262144],
'flavor15': [16, 65536],}
