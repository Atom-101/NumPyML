import numpy as np

class Dataset(object):
    def __init__(self,X,Y,batch_size):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.length = X.shape[0]

    def next(self):
        indices = np.random.choice(self.length,self.batch_size,replace=False)
        X = self.X[indices]
        Y = self.Y[indices]
        return X,Y
    

    