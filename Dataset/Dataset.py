import numpy as np

class Dataset(object):
    "This class wraps a dataset to be fed into a model object"
    def __init__(self,X,Y,batch_size,lazy_load=False,read_fn=None):
        ''' 
        Args:
        X: Array of input features or image filenames of the dataset
        Y: Array of model targets
        batch_size: Number of elements to forward and backpropagate in each iteration of model
        lazy_load: If set tor True, it indicates that the X values are not 
        in memory and have to be read from secondary storage.
        read_fn: If lazy_load is true, this function will be called at the start of every iteration
        with a batch of image filenames. The function should return a 4D numpy array, after loading
        images and doing relevant preprocessing
        '''
        self.X = np.array(X)
        self.Y = np.array(Y)
        if self.Y.ndim < 2:
            self.Y = self.Y[:,np.newaxis]
        self.lazy_load = lazy_load
        self.read_fn = read_fn
        self.batch_size = batch_size
        self.length = self.X.shape[0]

    def next(self):
        indices = np.random.choice(self.length,self.batch_size,replace=False)
        X = self.X[indices]
        Y = self.Y[indices]
        if self.lazy_load:
            X = self.read_fn(X)
        return X,Y
    

    