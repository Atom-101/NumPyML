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

file_names = glob.glob('train/*.jpg')[:2000]
labels = np.zeros(len(file_names))

for i,name in enumerate(file_names):
    if 'cat' in name:
        labels[i] += 1


def read(addresses):
    X = []
    for address in addresses:
        im = cv2.cvtColor(cv2.resize(cv2.imread(address),(32,32),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
        im = im[:,:,np.newaxis]
        im = im/255.0
        # im = im.reshape(-1)
        X.append(im)
    
    return np.array(X)


train_ds = Dataset(file_names,labels,8,True,read)

nn = Model((32,32,1))
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

nn.add(Conv(5,6,1,'same','leaky_relu'))
# # nn.add(BatchNorm(0.9))
# nn.add(Conv(3,2,1,'same','leaky_relu'))
# # nn.add(BatchNorm(0.9))
nn.add(MaxPool(2))
nn.add(Conv(5,16,1,'same','leaky_relu'))
# nn.add(Conv(1,2,2,'valid','leaky_relu'))

nn.add(Flatten())
nn.add(Dense(84,'relu'))
# nn.add(Dense(256,'relu'))
# nn.add(BatchNorm(0.9))
nn.add(Dense(1,'sigmoid'))

# nn.add(Conv(5,8,2,'valid','leaky_relu'))
# nn.add(BatchNorm(0.9))
# nn.add(Flatten())
# nn.add(Dense(512,'relu'))
# nn.add(BatchNorm(0.9))
# nn.add(Dense(1,'sigmoid'))


nn.train(1e-4,train_ds,10,'binary_cross_entropy')


