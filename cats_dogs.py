import cv2
import glob
import numpy as np

from Model.Model import Model
from Layers.Dense import Dense
from Layers.Flatten import Flatten
from Layers.BatchNorm import BatchNorm
from Layers.MaxPool import MaxPool
from Layers.Conv import Conv
from Layers.Activation import Activation
from Dataset.Dataset import Dataset

file_names = glob.glob('train/*.jpg')[:3200]
labels = np.zeros(len(file_names))

for i,name in enumerate(file_names):
    if 'cat' in name:
        labels[i] += 1


def read(addresses):
    X = []
    for address in addresses:
        im = cv2.resize(cv2.imread(address),(32,32),interpolation=cv2.INTER_AREA)
        # im = im.reshape(-1)
        X.append(im)
    
    return np.array(X)


train_ds = Dataset(file_names,labels,32,True,read)

nn = Model((32,32,3))
# nn.add(Conv(7,32,1,'same','relu'))
# nn.add(BatchNorm(0.9))
# nn.add(Conv(3,64,1,'same','relu'))
# nn.add(BatchNorm(0.9))
# nn.add(MaxPool(2))

# nn.add(Conv(5,128,1,'same','relu'))
# nn.add(BatchNorm(0.9))
# nn.add(Conv(3,128,1,'same','relu'))
# nn.add(BatchNorm(0.9))
# nn.add(MaxPool(2))

# nn.add(Conv(1,32,2,'valid','relu'))

# nn.add(Flatten())
# nn.add(Dense(1024,'relu'))
# nn.add(Dense(1,'sigmoid'))

nn.add(Conv(3,1,1,'same','linear'))
# nn.add(BatchNorm(0.9))
nn.add(Activation('leaky_relu'))
nn.add(Conv(3,2,1,'same','linear'))
# nn.add(BatchNorm(0.9))
nn.add(Activation('leaky_relu'))
nn.add(MaxPool(2))

# nn.add(Conv(1,2,2,'valid','relu'))

nn.add(Flatten())
nn.add(Dense(512,'linear'))
# nn.add(BatchNorm(0.9))
nn.add(Activation('leaky_relu'))
nn.add(Dense(1,'sigmoid'))


nn.train(1e-2,train_ds,100,'binary_cross_entropy','l2')


