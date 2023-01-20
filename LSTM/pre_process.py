from plot import plot_comp
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

data = pd.read_pickle('price-data-XBTUSD')
print(len(list(data)))
quit()
train_size = 2000

train_data = data[:train_size]
test_data = data[train_size:]


new_train = train_data.values.reshape(-1,1)

scaler = MinMaxScaler()

for i in range(4):
    scaler.fit(new_train[i*500:(i+1)*500, :])
    new_train[i*500:(i+1)*500, :] = scaler.transform(new_train[i*500:(i+1)*500, :])

test_data = scaler.transform(test_data.values.reshape(-1,1)).reshape(-1)
train_data = new_train.reshape(-1)



ema = 0.0
g = 0.1

for i in range(2000):
    ema  = g*train_data[i]+ema*(1-g)
    train_data[i] = ema

all_mid_data = np.concatenate([train_data,test_data],axis=0)
all_mid_data = pd.Series(all_mid_data)

all_mid_data.to_pickle('all_mid_data')

train_data = pd.Series(train_data)

train_data.to_pickle('train_data')

test_data = pd.Series(test_data)

train_data.to_pickle('test_data')