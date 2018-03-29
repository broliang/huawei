import pandas as pd
import datetime
import numpy as np
from statsmodels.tsa.arima_model import ARMA
data = pd.read_csv('TrainData_2015.1.1_2015.2.19.txt',sep = '\t',names = ['ID','class','time'])
data = data.drop('ID',axis = 1)
data['time'] = pd.to_datetime(data['time'])
data = pd.get_dummies(data)
data['time'] = data['time'].apply(lambda x:str(x)[:10])
data = data.groupby('time').aggregate('sum')
data['time'] = data.index
def get_data(data, time_steps=50):
    dataX = []
    dataY = []
    for i in range(0, len(data) - time_steps, 1):
        seq_in = data[i:i + time_steps]
        seq_out = data[i + time_steps]
        dataX.append(seq_in)
        dataY.append(seq_out)
    return dataX, dataY
def get_datelist(starttime,endtime):
  #get_datelist('2015-08-11','2015-09-22')
    startdate = pd.to_datetime(starttime)
    #now = datetime.datetime.now()
    delta = datetime.timedelta(days=1)
    # my_yestoday = startdate + delta
    # my_yes_time = my_yestoday.strftime('%Y%m%d')
    n = 0
    date_list = []
    while 1:
        if starttime<=endtime:
            days = (startdate  + delta*n).strftime('%Y-%m-%d')
            n = n+1
            date_list.append(days)
            if days == endtime:
                break
    return date_list
lis = get_datelist('2015-01-01','2015-02-19')
lis = pd.DataFrame({'time':lis})
data = pd.merge(lis,data,how='left').fillna(0)
data['week'] = pd.to_datetime(data['time']).apply(lambda x:x.weekday())
ans = []
n = 5
data1 = data[data['week'] != 5]
data1 = data1[data1['week'] != 6]
flavor2 = data1['class_flavor2'].tolist()

# 线性自回归
# flavor2 = data['class_flavor2'].tolist()
x,y = get_data(flavor2,n)
theta = np.linalg.lstsq(x,y)[0]

pn = 7
y = np.hstack((flavor2[-n:],np.zeros(pn)))
for i in range(pn):
    y[n+i] = np.dot(theta,y[i:n+i])
print(sum(y[-7:]))
# 平稳性检测
from statsmodels.tsa.stattools import adfuller as ADF
adf = ADF(data)
# 白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox
[[lb], [p]] = acorr_ljungbox(data, lags=1)
# 画出序列的自相关图
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

# ARMA
# for i in range(5):
#     model = ARMA(data['class_flavor'+str(i+1)].tolist(),(7,7)).fit()
#     temp = model.forecast(7)[0].sum()
#     ans.append(temp)
# print(ans)
sm.tsa.arma_order_select_ic(data['class_flavor2'],max_ar=6,max_ma=4,ic='hqic')['hqic_min_order']
sm.tsa.arma_order_select_ic(data['class_flavor2'],max_ar=6,max_ma=4,ic='bic')['bic_min_order']
sm.tsa.arma_order_select_ic(data['class_flavor2'],max_ar=6,max_ma=4,ic='aic')['aic_min_order']  # AIC

