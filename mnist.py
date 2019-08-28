import cv2
import glob
import numpy as np

from Model.Model import Model
from Layers.Dense import Dense
from Layers.Flatten import Flatten
from Layers.BatchNorm import BatchNorm
from Layers.MaxPool import MaxPool
from Layers.Conv import Conv
from Dataset.Dataset import Dataset

file_names = glob.glob('MnistTrain/*/*.jpg')[:3200]
labels = np.zeros((len(file_names),10))

for i,name in enumerate(file_names):
    labels[i] = np.eye(10)[int(name.split('/')[-2])]


def read(addresses):
    X = []
    for address in addresses:
        im = cv2.cvtColor(cv2.imread(address),cv2.COLOR_BGR2GRAY)
        im = im[:,:,np.newaxis]
        im = im/255.0
        # im = im.reshape(-1)
        X.append(im)
    
    return np.array(X)


train_ds = Dataset(file_names,labels,8,True,read)

nn = Model((28,28,1))
nn.add(Conv(5,6,1,'same','relu'))
nn.add(BatchNorm(0.9))
# nn.add(MaxPool(2))
nn.add(Conv(5,16,1,'same','relu'))

nn.add(Flatten())
nn.add(Dense(84,'relu'))
nn.add(BatchNorm(0.9))
nn.add(Dense(10,'linear'))


nn.train(1e-4,train_ds,10,'softmax_cross_entropy_with_logits')


