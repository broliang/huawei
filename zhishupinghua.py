import pandas as pd
import numpy as np

data = pd.read_csv('data_train_file.csv')
data['time'] = pd.to_datetime(data['time'],format='%Y%m%d', errors='ignore')
data = data.set_index('time')
# data = data.resample('D',how = 'sum')
cpu = np.array(data['cpu_num'])
s1_1 = []
for m in range(0, len(cpu)):
     S1_1_empty = []
     x = 0
     for n in range(0, 3):
        x = x + int(info_data_sales[m][n])
        x = x / 3
        S1_1_empty.append(x)
        S1_1.append(S1_1_empty)