import pandas as pd
import numpy as np
# from demo4 import flavor__


# 读数据
data = pd.read_csv('TrainData_2015.1.1_2015.2.19.txt',sep = '\t',names = ['id','class','time'])
data = data.drop('id',axis=1)
data['time'] = pd.to_datetime(data['time'])
# 每一种flavor按照原始时间刻度使用情况
class_data = pd.get_dummies(data)
# flavor原始时间刻度使用情况序列
flavor_seq = data['class']

data['cpu'] = [ flavor__[x][0] for x in data['class'].tolist()]
data['mem'] = [ flavor__[x][1] for x in data['class'].tolist()]
data['time'] = data['time'].apply(lambda x:str(x)[:10])

# 按天时间刻度的cpu和mem使用情况
data = data.groupby('time').aggregate('sum')

# 平稳性检测
from statsmodels.tsa.stattools import adfuller as ADF


# 画出序列的自相关图
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

#白噪声检测
from statsmodels.stats.diagnostic import acorr_ljungbox

#ARMA模型
from statsmodels.tsa.arima_model import ARMA
import statsmodels.api as sm

sm.tsa.arma_order_select_ic(data['class_flavor2'],max_ar=6,max_ma=4,ic='hqic')['hqic_min_order']
sm.tsa.arma_order_select_ic(data['class_flavor2'],max_ar=6,max_ma=4,ic='bic')['bic_min_order']
sm.tsa.arma_order_select_ic(data['class_flavor2'],max_ar=6,max_ma=4,ic='aic')['aic_min_order']