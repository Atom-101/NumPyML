import numpy as np

class Flatten(object):
    def _forward_pass(self, X):
        self.shape = X.shape
        return np.reshape(X,(self.shape[0],-1))

    def _backward_pass(self, gradients):
        return np.reshape(gradients,self.shape)


