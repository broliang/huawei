import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hpsklearn import xgboost_regression
#打开文档
ret = []
Y = []
for root, dirs, files in os.walk('data'):
    for filespath in files:
        ret.append(os.path.join(root, filespath))
        Y.append(filespath)

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

data = []
for i in ret:
    data.append(pd.read_csv(i,sep = '\t',names = ['id','class','time']))
result = pd.concat(data)
result = result.drop('id',axis = 1)
result['cpu'] = [ flavor__[x][0] for x in result['class'].tolist()]
result['mem'] = [ flavor__[x][1] for x in result['class'].tolist()]
result['time'] = pd.to_datetime(result['time'])
result = result.set_index('time')
cpu_day = result['2015-01-01':'2015-01-31']['cpu'].resample('D').sum().fillna(0)
cpu_week = result['2015-01-01':'2015-01-31']['cpu'].resample('W').sum().fillna(0)
cpu_grow_day = []
cpu_grow_week = []
x = y = 0
for i in cpu_day.tolist():
    x = x + i
    cpu_grow_day.append(x)
for i in cpu_week.tolist():
    y = y + i
    cpu_grow_week.append(y)
fig,axes = plt.subplots(2,1,sharey=True)
print(axes)
axes[0].plot(cpu_grow_week,label = 'week')
axes[1].plot(cpu_grow_day,label = 'day')
plt.show()