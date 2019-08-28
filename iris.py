import sys
sys.path.append('~/common_data/Projects/Custom_CNN_Lib')

import pandas as pd
import numpy as np

from Model.Model import Model
from Layers.Dense import Dense
from Layers.Conv import Conv
from Dataset.Dataset import Dataset

df = pd.read_csv('Iris.csv')
# print(df.head())

df['Species'] = pd.factorize(df['Species'],sort=True)[0]
# print(df.head())
train = df.iloc[:135,:]
valid = df.iloc[135:,:]

Y = train.iloc[:,-1].values
Y = np.eye(np.max(Y)+1)[Y]

train_dataset = Dataset(train.iloc[:,1:-1].values, Y, 135)
# print(train_dataset.next())

nn = Model(4)
nn.add(Dense(6,'relu'))
nn.add(Dense(3,'sigmoid'))

nn.train(1e-2, train_dataset, 8000, 'binary_cross_entropy',None)
# nn.train(1e-2, train_dataset, 8000, 'softmax_cross_entropy_with_logits',None)



