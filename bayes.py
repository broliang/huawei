import pandas as pd
import numpy as np
# from demo4 import flavor__
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ARDRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import xgboost

def get_data(data, time_steps=50): #把数据划分为0-50,51形式 用前一组预测后一组
    dataX = []
    dataY = []
    for i in range(0, len(data) - time_steps, 1):
        seq_in = data[i:i + time_steps]
        seq_out = data[i + time_steps]
        dataX.append(seq_in)
        dataY.append(seq_out)
    return np.array(dataX), np.array(dataY)
# 读数据
if __name__ == '__main__':
    data = pd.read_csv('TrainData_2015.1.1_2015.2.19.txt',sep = '\t',names = ['id','class','time'])
    data = data.drop('id',axis=1)
    data['time'] = pd.to_datetime(data['time'])
    # 每一种flavor按照原始时间刻度使用情况
    class_data = pd.get_dummies(data)
    #  time  class_flavor1  class_flavor10  class_flavor11  \
    # 0 2015-01-01 19:03:32              0               0               0
    # 1 2015-01-01 19:03:34              0               0               0
    # 2 2015-01-01 23:26:04              0               0               0
    # 3 2015-01-02 18:25:23              0               0               0
    # 4 2015-01-02 21:03:49              0               0               0
    # flavor原始时间刻度使用情况序列
    flavor_seq = data['class']
    ans = []  #输出结果
    flavor = class_data['class_flavor1']
    x, y = get_data(flavor, 50)
    y = y.reshape(-1, 1)
    br = LogisticRegression(penalty='l2', class_weight='balanced')
    br.fit(x, y)
    for i in range(1,6): #针对flavor1-5进行预测
        flavor = class_data['class_flavor'+str(i)]
        x,y = get_data(flavor,50)
        y = y.reshape(-1,1)
        # for i, j in flavor.items():
        #     x.append(i)
        #     y.append(j)
        # y = np.array(y).reshape(-1, 1)
        # x = np.array(x).reshape(-1, 1)
        # weight = {1: flavor.value_counts()[0] / flavor.count(), 0: flavor.value_counts()[1] / flavor.count()}
        # weight = np.random.exponential(5,50)
        # weight.sort()
        # weight = weight/weight.sum()
        # br = LogisticRegression(penalty='l1',class_weight='balanced')
        # # br = HyperoptEstimator()
        # br.fit(x, y)

        X = x[-1].reshape(1, -1)
        ans = []
        for i in range(61):#预测未来61次的数据，61得自己算
            t = br.predict(X)
            ans.append(t)
            X = np.delete(X, 0)
            X = np.append(X, t)
            X = X.reshape(1, -1)
        print(sum(ans))



