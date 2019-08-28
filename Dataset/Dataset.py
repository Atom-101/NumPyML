import numpy as np

class Dataset(object):
    def __init__(self,X,Y,batch_size,lazy_load=False,read_fn=None):
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
        # Per image normalization
        # X = X-np.apply_over_axes(np.mean,X,list(range(1,X.ndim)))
        # X = np.divide(X,np.apply_over_axes(np.std,X,list(range(1,X.ndim)))+1e-8)

        # Rescale
        # X = X/255.0
        return X,Y
    

    