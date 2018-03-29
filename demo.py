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
# train = pd.get_dummies(train)
train['day'] = train.index.weekday
# 提取类标签
import re
train['classid'] = train['class'].apply(lambda x:re.sub("\D",'',str(x)))
for i in range(5):

    class2 = train[name+str(i+1)].tolist()
    ttt.append(sum(class2))
    # 得出数据集，data[:50] -> data[51]
    # trainX,trainY = create_dataset(class2)
    # trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

    #模型1

    # model = Sequential()
    # model.add(LSTM(30, input_shape=(1, 50)))
    # model.add(Dense(1,activation = 'sigmoid'))
    # model.compile(loss='mse', optimizer='adam')
    # model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    dataX,dataY = get_data(class2)
    X = np.reshape(dataX, (len(dataX), 50, 1))
    y = np_utils.to_categorical(dataY)
    input_shape =(X.shape[1], X.shape[2])
    # 模型2
    model = Sequential()
    model.add(LSTM(32, input_shape=input_shape,stateful=True))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=50,verbose=2)
    # verbose=2
    # 预测
    trainPredict = model.predict(X)
    x = X[-1].reshape(1, 50, 1)
    ans = []
    for i in range(61):
        t = model.predict(x)[:,1]
        ans.append(t)
        x = np.delete(x,0)
        x = np.append(x,t)
        x = x.reshape(1,50,1)
    my.append(sum(ans))
print(ttt)
print(my)
print([0,13,3,0,2])


# n = 50
# flavor2 = train['class_flavor1'].tolist()
#
# # 线性自回归
# # flavor2 = data['class_flavor2'].tolist()
# x,y = get_data(flavor2,n)
# theta = np.linalg.lstsq(x,y)[0]
#
# pn = 50
# y = np.hstack((flavor2[-n:],np.zeros(pn)))
# for i in range(pn):
#     y[n+i] = np.dot(theta,y[i:n+i])
# print(sum(y[-pn:]))



