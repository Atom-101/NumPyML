import sys
sys.path.append('~/common_data/Projects/Custom_CNN_Lib')

import pandas as pd
import numpy as np

from Model.Model import Model
from Layers.Dense import Dense
from Dataset.Dataset import Dataset

df = pd.read_csv('Iris.csv')
# print(df.head())

df['Species'] = pd.factorize(df['Species'],sort=True)[0]
# print(df.head())
train = df.iloc[:135,:]
test = df.iloc[135:,:]

Y = train.iloc[:,-1].values
Y = np.eye(np.max(Y)+1)[Y]

train_dataset = Dataset(train.iloc[:,:-2].values, Y, 135)
# print(train_dataset.next())

nn = Model(4)
nn.add(Dense(4,'relu'))
nn.add(Dense(3,'sigmoid'))

nn.train(1e-2, train_dataset, 64, 100, 'mse')



